from tqdm import tqdm
import os
import copy
import gym
import time
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.diffusion import Diffusion_Lower, Diffusion_Upper
from agents.models import Transformer, Value, Actor, LSTMPolicy, RNNPolicy, GRUPolicy, MTPN_Policy, MambaPolicy, Critic
from agents.temporal import TemporalUnet_Lower, TemporalUnet_Upper
from datasets.normalization import DatasetNormalizer
from utils.utils import set_seed
torch.backends.cudnn.enabled = True

class Policy(object):
    def __init__(self,
                 env_name,
                 state_dim,
                 action_dim,
                 action_scale,
                 horizon_seq,
                 horizon_up,
                 horizon_low,
                 interval,
                 device,
                 discount,
                 tau,
                 n_timesteps,
                 mamba_layers,
                 outter_layers,
                 attn_layers,
                 eval_episodes,
                 random_len,
                 embedding_dim=256,
                 d_ff=512,
                 use_attention=False,
                 w=1,
                 threshold_scale=1.0,                 
                 history=None,
                 schedule = "linear",
                 lr=3e-4,
                 model='MTPN',
                 dim_mults_up=(1, 2, 4, 8),
                 dim_mults_low=(1, 2, 4, 8),
                 args=None,
                 update_every=2,
                 use_cfg=True,
                 use_adapt_art=True,
                 use_diffusion=True) -> None:
        
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.min_action = float(action_scale.low[0])
        self.max_action = float(action_scale.high[0])
        self.horizon_seq = horizon_seq
        self.horizon_up = horizon_up
        self.horizon_low = horizon_low
        self.interval = interval
        self.device = device
        self.discount = discount
        self.tau = tau
        self.n_timesteps = n_timesteps
        self.mamba_layers = mamba_layers
        self.outter_layers = outter_layers
        self.attn_layers = attn_layers
        self.eval_episodes = eval_episodes
        self.random_len = random_len
        self.history = history
        if history is None:
            self.history = horizon_seq - 1
        self.args = args
        self.sps_list = []
        self.update_every = update_every
        self.w = w
        self.threshold_scale = threshold_scale
        self.use_cfg = use_cfg
        self.up_index = [interval * i for i in range(self.horizon_up)] # [0, 3, 6, 9, 12, 15, 18, 21]
        self.break_count = 0
        self.use_adapt_art = use_adapt_art
        self.use_diffusion = use_diffusion
        self.model = model

        print(f"[ Use model ] {model}")

        #=====================================================================#
        #============================= Seq Model =============================#
        #=====================================================================#

        if model == 'Transformer':
            n_heads, n_layers = 4, 4
            self.feasible_generator = Transformer(state_dim, embedding_dim, n_heads, d_ff, n_layers, 0.1).to(device)
        elif model == 'LSTM':
            self.feasible_generator = LSTMPolicy(state_dim, length=horizon_seq).to(device)
        elif model == 'RNN':
            self.feasible_generator = RNNPolicy(state_dim, length=horizon_seq).to(device)
        elif model == 'GRU':
            self.feasible_generator = GRUPolicy(state_dim, length=horizon_seq).to(device)
        elif model == 'Mamba':
            self.feasible_generator = MambaPolicy(input_dim=state_dim, d_model=128, mamba_dim=64, mamba_layers=mamba_layers*2, dropout_rate=0.1, 
                                                  ssm_cfg={"d_state":64, "expand":2}, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, device=device).to(device)
        elif model == 'MTPN':            
            self.feasible_generator = MTPN_Policy(input_dim=state_dim, d_model=128, d_ff=d_ff, mamba_dim=64, n_heads=4, mamba_layers=mamba_layers, outter_layers=outter_layers, attn_layers=attn_layers,
                                                   dropout_rate=0.1, ssm_cfg={"d_state":64, "expand":2}, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, device=device).to(device)
          
        self.feasible_generator_optimizer = torch.optim.Adam(self.feasible_generator.parameters(), lr=lr)

        #=====================================================================#
        #=========================== Low Diffuser ============================#
        #=====================================================================#

        self.model_low = TemporalUnet_Lower(horizon=self.horizon_low, transition_dim=state_dim, attention=use_attention, dim_mults=dim_mults_low).to(device)
        self.planner_low = Diffusion_Lower(state_dim, self.model_low, horizon=self.horizon_low, n_timesteps=n_timesteps, predict_epsilon=True, beta_schedule=schedule, w=w, use_cfg=use_cfg).to(device) # predict_epsilon=False
        self.planner_low_optimizer = torch.optim.AdamW(self.planner_low.parameters(), lr=lr, weight_decay=1e-4)
        self.ema = EMA(1-tau)
        self.ema_model_low = copy.deepcopy(self.planner_low)

        #=====================================================================#
        #============================ Up Diffuser ============================#
        #=====================================================================#

        self.model_up = TemporalUnet_Upper(horizon=self.horizon_up, transition_dim=state_dim, attention=use_attention, dim_mults=dim_mults_up).to(device)
        self.planner_up = Diffusion_Upper(state_dim, self.model_up, horizon=self.horizon_up, n_timesteps=n_timesteps, predict_epsilon=True, beta_schedule=schedule, w=w, use_cfg=use_cfg).to(device)
        self.planner_up_optimizer = torch.optim.AdamW(self.planner_up.parameters(), lr=lr, weight_decay=1e-4)
        self.ema_model_up = copy.deepcopy(self.planner_up)

        #=====================================================================#
        #=============================== Actor ===============================#
        #=====================================================================#

        self.actor = Actor(state_dim, action_dim, hidden_dim=256).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        #=====================================================================#
        #=============================== Critic ==============================#
        #=====================================================================#

        self.critic = Critic(input_dim=state_dim, d_model=embedding_dim, n_heads=4, d_ff=d_ff, n_layers=2, dropout_rate=0.1).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        #=====================================================================#
        #================================ Value ==============================#
        #=====================================================================#

        self.value = Value(state_dim, hidden_dim=256).to(device)
        self.value_target = copy.deepcopy(self.value)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

        #=====================================================================#
        #============================ Lr scheduler ===========================#
        #=====================================================================#
        
        # linear decay of learning rate
        self.planner_low_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.planner_low_optimizer, 1, 1e-2, total_iters=1e5)
        self.planner_up_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.planner_up_optimizer, 1, 1e-2, total_iters=1e5)
        self.feasible_generator_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.feasible_generator_optimizer, 1, 1e-2, total_iters=1e5)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.actor_optimizer, 1, 1e-2, total_iters=1e5)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.critic_optimizer, 1, 1e-2, total_iters=1e5)
        self.value_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.value_optimizer, 1, 1e-2, total_iters=1e5)

    def model_eval(self):
        self.feasible_generator.eval()
        self.ema_model_low.eval()
        self.ema_model_up.eval()
        self.critic_target.eval()
        self.value_target.eval()
        self.actor.eval()
        
    def model_train(self):
        self.feasible_generator.train()
        self.ema_model_low.train()
        self.ema_model_up.train()
        self.critic_target.train()
        self.value_target.train()
        self.actor.train()    

    def evaluate(self, env_name, eval_episodes=10, normalizer:DatasetNormalizer=None, rtg=None, scale=1.0, use_diffusion=True, show_progress=True, seed=None):
        print('Evaluate seed: ', seed)
        set_seed(seed)

        env = gym.make(env_name)
        env.seed(seed)

        # set the model to evaluation mode and disable layer behaviors unique to training such as Dropout and BatchNorm
        self.model_eval()

        return_to_go = rtg
        scores = []

        for idx in range(eval_episodes):
            self.break_count = 0
            state, done = env.reset(), False  # directly input the seed into reset to ensure that the initial state is fixed
            history_state = []
            history_rtg = []
            episode_reward = 0
            rtg = return_to_go
            rtg = rtg / scale
            min_rtg = rtg / 3
            total_step = 0
            start_time = time.perf_counter()  

            if isinstance(state, dict):
                state = state["observation"] # shape:(state_dim,)
            state = normalizer.normalize(state, "observations")
            history_state.append(state)
            history_rtg.append(rtg)
            
            while not done:
                _state: torch.Tensor = torch.tensor(np.stack(history_state, 0), dtype=torch.float32, device=self.device).unsqueeze(0)
                _rtg: torch.Tensor = torch.tensor(np.stack(history_rtg, 0), dtype=torch.float32, device=self.device).unsqueeze(0)

                action_list, threshold_list = self.select_action(_state, _rtg, use_diffusion)
                for j in range(self.horizon_low - 1):
                    action = action_list[j]
                    state, reward, done, _ = env.step(action)
                    rtg -= reward / scale
                    if self.use_adapt_art:
                        break_flag = (rtg < self.threshold_scale*threshold_list[j])
                    rtg = np.clip(rtg, min_rtg, None)
                    episode_reward += reward

                    if isinstance(state, dict):
                        state = state["observation"] # shape:(state_dim,)
                    state = normalizer.normalize(state, "observations")
                    history_state.append(state)
                    history_rtg.append(rtg)

                    if show_progress:
                        total_step += 1
                        print(f"                                                              ", end="\r")
                        print(f"steps: {total_step} -------- rewards: {episode_reward}", end="\r")

                    if done: break

                    if self.use_adapt_art and break_flag:
                        self.break_count += 1
                        break

            end_time = time.perf_counter() 
            sps = round((end_time - start_time)/total_step, 2) 
            
            if show_progress:
                print(f"reward: {episode_reward}, normalized_scores: {env.get_normalized_score(episode_reward)}, total_step: {total_step}, sps: {sps}s, break_count: {self.break_count}")

            scores.append(episode_reward)

            if total_step == 1000:
                self.sps_list.append(sps)

        self.model_train()

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)

        sps_avg = np.mean(self.sps_list) if self.sps_list != [] else 0
        sps_min = np.min(self.sps_list) if self.sps_list != [] else 0     

        # normlize
        normalized_scores = [env.get_normalized_score(s) for s in scores]
        avg_normalized_score = env.get_normalized_score(avg_score)
        std_normalized_score = np.std(normalized_scores)
        max_normalized_score = np.max(normalized_scores)

        return {"reward/avg": avg_score,
                "reward/std": std_score,
                "reward/avg_normalized": avg_normalized_score,
                "reward/std_normalized": std_normalized_score,
                "reward/max": max_score,
                "reward/max_normalized": max_normalized_score}, {"sps/avg": sps_avg,"sps/min": sps_min}

    def train(self, batch, cnt):
        observations, next_observations, actions, rtg = batch

        observations = observations.to(self.device)
        next_observations = next_observations.to(self.device)
        actions = actions.to(self.device)
        rtg = rtg.unsqueeze(-1).to(self.device)     # [batch_size, seq] -> [batch_size, seq, 1]
        states = observations[:, :-1]
        next_states = observations[:, 1:]
        cond = {0:observations[:,0]}

        # start_time = time.perf_counter()

        #=====================================================================#
        #=============================== Value ===============================#
        #=====================================================================#

        current_rtg1, current_rtg2 = self.value.forward(observations.view(-1, self.state_dim))
        v_loss = F.mse_loss(current_rtg1, rtg.reshape(-1, 1)) + F.mse_loss(current_rtg2, rtg.reshape(-1, 1))

        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()

        #=====================================================================#
        #============================ Planner up =============================#
        #=====================================================================#

        upper_input = observations[:, self.up_index, :]
        p_loss_up:torch.Tensor = self.planner_up.loss(upper_input, rtg[:, 0], cond)

        self.planner_up_optimizer.zero_grad()
        p_loss_up.backward()
        self.planner_up_optimizer.step()

        #=====================================================================#
        #======================== Feasible generator =========================#
        #=====================================================================#
        if self.model == 'MTPN':
            state_pred, reward_pred, seq_len = self.feasible_generator(states, rtg[:,:-1], self.random_len, 'train')
            t_loss = F.mse_loss(state_pred, next_states[:, -seq_len:]) + F.mse_loss(reward_pred, rtg[:, -seq_len:])
        else:
            state_pred, reward_pred = self.feasible_generator(states, rtg[:,:-1])
            t_loss = F.mse_loss(state_pred, next_states[:, -self.horizon_seq:]) + F.mse_loss(reward_pred, rtg[:, -self.horizon_seq:])

        self.feasible_generator_optimizer.zero_grad()
        t_loss.backward()
        self.feasible_generator_optimizer.step()

        #=====================================================================#
        #============================ Planner Low ============================#
        #=====================================================================#

        p_loss_low:torch.Tensor = self.planner_low.loss(observations[:, 0:self.horizon_low], observations[:, self.interval], rtg[:, 0], cond)

        self.planner_low_optimizer.zero_grad()
        p_loss_low.backward()
        self.planner_low_optimizer.step()

        #=====================================================================#
        #=============================== Actor ===============================#
        #=====================================================================#

        s_ns_pair = torch.cat([observations, next_observations], dim=-1).view(-1, self.state_dim*2)
        pred_actions = self.actor(s_ns_pair).clamp(self.min_action, self.max_action)
        a_loss = F.mse_loss(pred_actions, actions.view(-1, self.action_dim))

        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        #=====================================================================#
        #============================== Critic ===============================#
        #=====================================================================#

        value = self.critic(observations[:, 0:self.horizon_low])
        reward = (rtg[:, 0:self.horizon_low] - rtg[:, 1:self.horizon_low+1]).squeeze(-1)
        c_loss = F.mse_loss(value, reward)

        self.critic_optimizer.zero_grad()
        c_loss.backward()
        self.critic_optimizer.step()

        # end_time = time.perf_counter()
        # print(f"p_loss time: {(end_time-start_time)*5e4 / 3600}")

        if cnt % 5e4 == 0:
            print(f"Step {cnt}: Learning Rate = {self.planner_low_optimizer.param_groups[0]['lr']:.6f}")

        self.planner_low_lr_scheduler.step()
        self.planner_up_lr_scheduler.step()
        self.feasible_generator_lr_scheduler.step()
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()
        self.value_lr_scheduler.step()

        return {
                "loss/planner_low": p_loss_low.item(),
                "loss/planner_up": p_loss_up.item(),
                "loss/model": t_loss.item(),
                "loss/critic": c_loss.item(),
                "loss/value": v_loss.item(),
                "loss/actor": a_loss.item(),
                }
    
    def save(self, path=None):
        if path is None:
            path = "./Diffusion Model/APHD/results/checkpoint.pth"
        prefix = os.path.dirname(path)
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        torch.save({
            'planner_low': self.planner_low.state_dict(),
            'planner_up': self.planner_up.state_dict(),
            'ema_model_low': self.ema_model_low.state_dict(),
            'ema_model_up': self.ema_model_up.state_dict(),
            'feasible_generator': self.feasible_generator.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'value': self.value.state_dict(),
            'value_target': self.value_target.state_dict(),            
        }, path)

    def select_action(self, state, rtg, use_diffusion=True):
        repeat = 32
        state = torch.repeat_interleave(state, repeat, dim=0)
        rtg = torch.repeat_interleave(rtg, repeat, dim=0).unsqueeze(-1)
        cond = {0:state[:,-1]}          # fix the starting point of the trajectory
        condition = rtg[:, -1]        # condition reward

        action_list = []
        threshold_list = []
        
        with torch.no_grad():

            #=====================================================================#
            #============================= Seq Model =============================#
            #=====================================================================#

            for _ in range(self.horizon_seq - 1):
                if self.model == 'MTPN':
                    pred_state, pred_rtg, _ = self.feasible_generator(state[:,-self.history:], rtg[:,-self.history:], False, 'eval')
                else:
                    pred_state, pred_rtg = self.feasible_generator(state[:,-self.history:], rtg[:,-self.history:])

                state = torch.cat([state, pred_state[:, -1:]], dim=1)
                rtg = torch.cat([rtg, pred_rtg[:,-1:]], dim=1)

            state = state[:, -self.horizon_seq:, :]

            #=====================================================================#
            #============================ Up Diffuser ============================#
            #=====================================================================#
            if use_diffusion:
                sub_goals = self.ema_model_up(state[:, self.up_index, :], condition, cond)
                sub_goals = sub_goals[:, 1]

            #=====================================================================#
            #=========================== Low Diffuser ============================#
            #=====================================================================#

                _noise_state = state[:, 0:self.horizon_low, :]
                state = self.ema_model_low(_noise_state, sub_goals, condition, cond)

            #=====================================================================#
            #=============================== Critic ==============================#
            #=====================================================================#

            values = self.critic_target(state)
            values = torch.sum(values, dim=1)      
            idx = torch.argmax(values, dim=0)
            state = state[idx].squeeze(0)

            #=====================================================================#
            #=============================== Actor ===============================#
            #=====================================================================#

            for i in range(self.horizon_low - 1):
                _cond = state[i:i+2, :]
                action = self.actor(_cond.reshape(-1, self.state_dim * 2)).squeeze()
                action_list.append(action.cpu().data.numpy().flatten())

            #=====================================================================#
            #=============================== Value ===============================#
            #=====================================================================#

                threshold = self.value_target.q_min(_cond[1]).flatten()
                threshold_list.append(threshold.squeeze().cpu().data.numpy())
    
        return action_list, threshold_list

    def reset_parameters(self):
        self.ema_model_low.load_state_dict(self.planner_low.state_dict())
        self.ema_model_up.load_state_dict(self.planner_up.state_dict())

    def step_ema(self, step, step_start = 1000):
        if step < step_start:
            self.reset_parameters()
            return
        if step % self.update_every == 0:
            self.ema.update_model_average(self.ema_model_low, self.planner_low)
            self.ema.update_model_average(self.ema_model_up, self.planner_up)

    def step_critic(self, step, step_start = 1000):
        if step < step_start:
            self.critic_target.load_state_dict(self.critic.state_dict())
            return
        if step % self.update_every == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def step_value(self, step, step_start = 1000):
        if step < step_start:
            self.value_target.load_state_dict(self.value.state_dict())
            return
        if step % self.update_every == 0:
            for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new