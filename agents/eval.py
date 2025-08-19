import gym
import time
import numpy as np
import torch
from agents.diffusion import Diffusion_Lower, Diffusion_Upper
from agents.models import Transformer, Value, Actor, LSTMPolicy, RNNPolicy, GRUPolicy, MTPN_Policy, MambaPolicy, Critic
from agents.temporal import TemporalUnet_Lower, TemporalUnet_Upper
from datasets.normalization import DatasetNormalizer
from utils.utils import set_seed
from PIL import Image
import matplotlib.pyplot as plt
import os

class Eval_Policy(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 horizon_seq,
                 horizon_up,
                 horizon_low,
                 interval,
                 device,
                 n_timesteps,
                 mamba_layers,
                 outter_layers,
                 attn_layers,
                 embedding_dim=256,
                 d_ff=512,
                 use_attention=False,
                 w=1,
                 threshold_scale=1.0,                 
                 history=None,
                 schedule = "linear",
                 model='LSAM',
                 dim_mults_up=(1, 2, 4, 8),
                 dim_mults_low=(1, 2, 4, 8),
                 use_cfg=True,
                 use_adapt_art=True,
                 use_diffusion=True) -> None:
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon_seq = horizon_seq
        self.horizon_up = horizon_up
        self.horizon_low = horizon_low
        self.interval = interval
        self.device = device
        self.n_timesteps = n_timesteps
        self.mamba_layers = mamba_layers
        self.outter_layers = outter_layers
        self.attn_layers = attn_layers
        self.history = history
        if history is None:
            self.history = horizon_seq - 1

        self.w = w
        self.threshold_scale = threshold_scale
        self.use_cfg = use_cfg
        self.use_adapt_art = use_adapt_art
        self.use_diffusion = use_diffusion
        self.up_index = [interval * i for i in range(horizon_up)] # [0, 3, 6, 9, 12, 15, 18, 21]
        self.break_count = 0
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
          

        #=====================================================================#
        #=========================== Low Diffuser ============================#
        #=====================================================================#

        self.model_low = TemporalUnet_Lower(horizon=self.horizon_low, transition_dim=state_dim, attention=use_attention, dim_mults=dim_mults_low).to(device)
        self.ema_model_low = Diffusion_Lower(state_dim, self.model_low, horizon=self.horizon_low, n_timesteps=n_timesteps, predict_epsilon=True, beta_schedule=schedule, w=w, use_cfg=use_cfg).to(device)# predict_epsilon=False

        #=====================================================================#
        #============================ Up Diffuser ============================#
        #=====================================================================#

        self.model_up = TemporalUnet_Upper(horizon=self.horizon_up, transition_dim=state_dim, attention=use_attention, dim_mults=dim_mults_up).to(device)
        self.ema_model_up = Diffusion_Upper(state_dim, self.model_up, horizon=self.horizon_up, n_timesteps=n_timesteps, predict_epsilon=True, beta_schedule=schedule, w=w, use_cfg=use_cfg).to(device)

        #=====================================================================#
        #=============================== Actor ===============================#
        #=====================================================================#

        self.actor = Actor(state_dim, action_dim, hidden_dim=256).to(device)

        #=====================================================================#
        #=============================== Critic ==============================#
        #=====================================================================#

        self.critic_target = Critic(input_dim=state_dim, d_model=embedding_dim, n_heads=4, d_ff=d_ff, n_layers=2, dropout_rate=0.1).to(device)

        #=====================================================================#
        #================================ Value ==============================#
        #=====================================================================#

        self.value_target = Value(state_dim, hidden_dim=256).to(device)

    # @torch.no_grad()
    def evaluate(self, env_name, normalizer:DatasetNormalizer=None, rtg=None, scale=1.0, image_path=None, show_progress=True, seed=None, render=False, sample_interval=10):
        set_seed(seed)
        env = gym.make(env_name)
        env.seed(seed)
        state, done = env.reset(), False  # directly input the seed into reset to ensure that the initial state is fixed

        if render:
            # frame = env.render(mode="rgb_array") # mujoco maze2d
            frame = env.sim.render(width=640, height=480, camera_name="vil_camera") # adort (pen:fixed) (door:vil_camera) (hammer:vil_camera) (relocate:vil_camera)
            img = Image.fromarray(frame)
            img.save(os.path.join(image_path, f'frame-{0}.png'))
        
        history_state = []
        history_rtg = []
        episode_reward = 0
        self.break_count = 0
        rtg = rtg / scale
        min_rtg = rtg / 3
        total_step = 0
        break_flag = False
        start_time = time.perf_counter()  

        if isinstance(state, dict):
            state = state["observation"] # shape:(state_dim,)
        state = normalizer.normalize(state, "observations")
        history_state.append(state)
        history_rtg.append(rtg)
        
        while not done:
            _state: torch.Tensor = torch.tensor(np.stack(history_state, 0), dtype=torch.float32, device=self.device).unsqueeze(0)
            _rtg: torch.Tensor = torch.tensor(np.stack(history_rtg, 0), dtype=torch.float32, device=self.device).unsqueeze(0)

            action_list, threshold_list = self.select_action(_state, _rtg)
            for j in range(self.horizon_low - 1):
                action = action_list[j]
                state, reward, done, _ = env.step(action)
                rtg -= reward / scale
                if self.use_adapt_art:
                    break_flag = (rtg < self.threshold_scale*threshold_list[j])

                rtg = np.clip(rtg, min_rtg, None)
                episode_reward += reward

                if isinstance(state, dict):
                    state = state["observation"]
                state = normalizer.normalize(state, "observations")
                history_state.append(state)
                history_rtg.append(rtg)

                if show_progress:
                    total_step += 1
                    print(f"                                                              ", end="\r")
                    print(f"steps: {total_step} -------- rewards: {episode_reward}", end="\r")

                # render
                if render:
                    if total_step % sample_interval == 0:
                        # frame = env.render(mode="rgb_array") # mujoco maze2d
                        frame = env.sim.render(width=640, height=480, camera_name="vil_camera") # adort (pen:fixed) (door:vil_camera) (hammer:vil_camera) (relocate:vil_camera)
                        img = Image.fromarray(frame)
                        img.save(os.path.join(image_path, f'frame-{total_step}.png'))

                if done: break

                if self.use_adapt_art and break_flag:
                    self.break_count += 1
                    break

        end_time = time.perf_counter() 
        sps = round((end_time - start_time)/total_step, 2) 
        
        if show_progress:
            print(f"reward: {episode_reward}, normalized_scores: {env.get_normalized_score(episode_reward)}, total_step: {total_step}, sps: {sps}s, break_count: {self.break_count}, seed: {seed}")

        return episode_reward, sps

    def load(self, path=None, path_diffusion=None):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.feasible_generator.load_state_dict(checkpoint['feasible_generator'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.value_target.load_state_dict(checkpoint['value_target'])

        # checkpoint = torch.load(path_diffusion, map_location=self.device, weights_only=True)
        self.ema_model_low.load_state_dict(checkpoint['ema_model_low'])
        self.ema_model_up.load_state_dict(checkpoint['ema_model_up'])

        self.feasible_generator.eval()
        self.model_low.eval()
        self.model_up.eval()
        self.ema_model_low.eval()
        self.ema_model_up.eval()
        self.critic_target.eval()
        self.value_target.eval()
        self.actor.eval()

    @torch.no_grad()
    def select_action(self, state, rtg):
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

                if self.use_diffusion:
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

    def calculate(self, env_name, eval_episodes=10, normalizer:DatasetNormalizer=None, rtg=None, scale=1.0, show_progress=True, seed=None):
        print('Evaluate seed: ', seed)
        set_seed(seed)
        env = gym.make(env_name)

        return_to_go = rtg
        scores = []
        reward_list = [[] for _ in range(eval_episodes)]

        for idx in range(eval_episodes):
            self.break_count = 0
            state, done = env.reset(seed=seed), False  # directly input the seed into reset to ensure that the initial state is fixed
            
            history_state = []
            history_rtg = []
            episode_reward = 0
            rtg = return_to_go
            rtg = rtg / scale
            min_rtg = rtg / 3
            total_step = 0
            break_flag = False

            if isinstance(state, dict):
                state = state["observation"] # shape:(state_dim,)
            state = normalizer.normalize(state, "observations")
            history_state.append(state)
            history_rtg.append(rtg)
            
            while not done:
                _state: torch.Tensor = torch.tensor(np.stack(history_state, 0), dtype=torch.float32, device=self.device).unsqueeze(0)
                _rtg: torch.Tensor = torch.tensor(np.stack(history_rtg, 0), dtype=torch.float32, device=self.device).unsqueeze(0)

                action_list, threshold_list = self.select_action(_state, _rtg)
                for j in range(7):
                    action = action_list[j]
                    state, reward, done, _ = env.step(action)
                    rtg -= reward / scale
                    reward_list[idx].append(reward)
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
            
            if show_progress:
                print(f"reward: {episode_reward}, normalized_scores: {env.get_normalized_score(episode_reward)}, total_step: {total_step}, break_count: {self.break_count}")

            scores.append(episode_reward)

        idx = scores.index(max(scores))
        max_score = max(scores)

        print("max:", np.round(max_score, 4))   
        print("max list:", np.round(np.array(reward_list[idx]), 4).tolist())   