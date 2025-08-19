from typing import Dict, List, Tuple, Union
import os
import shutil
import argparse
import tqdm
import gym
import numpy as np
import torch
import d4rl
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from agents.policy import Policy
from datasets.dataset import SequenceDatasetV2
from datasets.normalization import DatasetNormalizer
from utils.logger import setup_logger
from utils.helpers import cycle
from utils.utils import EarlyStopping, set_seed, record, seed_worker, generate_horizon_sequence

hyperparameters = {
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'horizon_seq': 10, 'horizon_up': 4, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.1, 'threshold_scale': 1.0, 'rtg': 12000.0},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'horizon_seq': 10, 'horizon_up': 4, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.1, 'threshold_scale': 1.0, 'rtg': 5300.0},
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'horizon_seq': 10, 'horizon_up': 4, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.1, 'threshold_scale': 1.0, 'rtg': 5300.0},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 4, 'horizon_low': 8, 'interval': 7, 'n_timesteps': 5, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 3600.0},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 4, 'horizon_low': 8, 'interval': 7, 'n_timesteps': 5, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 3100.0},
    'hopper-medium-v2':              {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 4, 'horizon_low': 8, 'interval': 7, 'n_timesteps': 5, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 3100.0},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.1, 'threshold_scale': 1.0, 'rtg': 5100.0},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.1, 'threshold_scale': 1.0, 'rtg': 4200.0},
    'walker2d-medium-v2':            {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.1, 'threshold_scale': 1.0, 'rtg': 4200.0},
    'pen-human-v1':                  {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 2, 'scalar': 1.3, 'threshold_scale': 1.0, 'rtg': 6000.0},
    'pen-cloned-v1':                 {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 2, 'scalar': 1.3, 'threshold_scale': 1.0, 'rtg': 6000.0},
    'pen-expert-v1':                 {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 2, 'scalar': 1.3, 'threshold_scale': 1.0, 'rtg': 6000.0},
    'kitchen-partial-v0':            {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 500.0},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 400.0},
    'door-human-v0':                 {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 2, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 1000.0},
    'door-cloned-v0':                {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 2, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 1000.0},
    'door-expert-v0':                {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 3000.0},
    'hammer-human-v0':               {'lr': 3e-4, 'horizon_seq': 10, 'horizon_up': 4, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 10000.0},
    'hammer-cloned-v0':              {'lr': 3e-4, 'horizon_seq': 10, 'horizon_up': 4, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 3000.0},
    'hammer-expert-v0':              {'lr': 3e-4, 'horizon_seq': 10, 'horizon_up': 4, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 17000.0},
    'relocate-human-v0':             {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 1000.0},
    'relocate-cloned-v0':            {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 3000.0},
    'relocate-expert-v0':            {'lr': 3e-4, 'horizon_seq': 22, 'horizon_up': 8, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.2, 'threshold_scale': 1.0, 'rtg': 4000.0},
    'maze2d-umaze-v1':               {'lr': 3e-4, 'horizon_seq': 10, 'horizon_up': 4, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.1, 'threshold_scale': 1.0, 'rtg': 300.0},
    'maze2d-medium-v1':              {'lr': 3e-4, 'horizon_seq': 10, 'horizon_up': 4, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.1, 'threshold_scale': 1.0, 'rtg': 300.0},
    'maze2d-large-v1':               {'lr': 3e-4, 'horizon_seq': 10, 'horizon_up': 4, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.1, 'threshold_scale': 1.0, 'rtg': 400.0},
    'antmaze-umaze-diverse-v2':      {'lr': 3e-4, 'horizon_seq': 10, 'horizon_up': 4, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.1, 'threshold_scale': 1.0, 'rtg': 1.0},
    'antmaze-medium-diverse-v2':     {'lr': 3e-4, 'horizon_seq': 10, 'horizon_up': 4, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.1, 'threshold_scale': 1.0, 'rtg': 1.0},
    'antmaze-large-diverse-v2':      {'lr': 3e-4, 'horizon_seq': 10, 'horizon_up': 4, 'horizon_low': 4, 'interval': 3, 'n_timesteps': 5, 'scalar': 1.1, 'threshold_scale': 1.0, 'rtg': 1.0},
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='hopper-medium-expert-v2')
    parser.add_argument('--seed', type=int, default=0) 
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=5e4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--schedule', type=str, default='linear') # [linear, vp, cosine]
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--mamba_layers', type=int, default=2)
    parser.add_argument('--outter_layers', type=int, default=2)
    parser.add_argument('--attn_layers', type=int, default=2)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--random_len', type=bool, default=False)
    parser.add_argument('--early_stop', type=bool, default=False)
    parser.add_argument('--use_cfg', type=bool, default=True)
    parser.add_argument('--use_adapt_art', type=bool, default=True) # False
    parser.add_argument('--use_diffusion', type=bool, default=True) # False
    parser.add_argument('--step_start', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=2)
    parser.add_argument('--model', type=str, default='MTPN') # [Transformer, LSTM, RNN, GRU, Mamba, MTPN]
    return parser.parse_args()

def get_dir():
    current_file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(current_file_path)
    return dir_path

def main():

    #=================================================#
    #==============Hyperparameter setting=============#
    #=================================================#

    args = parse_args()
    env_name = args.env_name
    seed = args.seed
    batch_size = args.batch_size
    epochs = args.epochs
    eval_freq = args.eval_freq
    gamma = args.gamma
    schedule = args.schedule
    device = args.device
    tau = args.tau
    mamba_layers = args.mamba_layers
    outter_layers = args.outter_layers
    attn_layers = args.attn_layers
    eval_episodes = args.eval_episodes
    random_len = args.random_len
    early_stop = args.early_stop
    use_cfg = args.use_cfg
    use_adapt_art = args.use_adapt_art
    use_diffusion = args.use_diffusion
    step_start = args.step_start
    update_every = args.update_every
    model = args.model

    horizon_seq = hyperparameters[env_name]['horizon_seq']
    horizon_up = hyperparameters[env_name]['horizon_up']        # horizon_up = horizon_seq // interval + 1
    horizon_low = hyperparameters[env_name]['horizon_low']      # horizon_low = interval + 1
    interval = hyperparameters[env_name]['interval']
    n_timesteps = hyperparameters[env_name]['n_timesteps']
    lr = hyperparameters[env_name]['lr']
    w = hyperparameters[env_name]['scalar']
    threshold_scale = hyperparameters[env_name]['threshold_scale']
    rtg = hyperparameters[env_name]['rtg']

    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    planner_loss_t, model_loss_t = 100., 100.
    max_reward, avg_reward = 0, 0

    if 'maze2d' in args.env_name:
        scale = 500
    elif ('halfcheetah' in args.env_name) or ('hammer' in args.env_name):
        scale = 10000
    elif 'antmaze' in args.env_name:
        scale = 10
    else:
        scale = 1000

    cnt = 0

    dim_mults_up = generate_horizon_sequence(horizon_up)
    dim_mults_low = generate_horizon_sequence(horizon_low)
    print("\n=================== Successfully set hyperparameters ===================\n")

    #=================================================#
    #===================Create wandb==================#
    #=================================================#

    wandb.login()
    wandb_logger = wandb.init(
        project="APHD " + f"({env_name})",
        name=f"{model}/{seed}-Horzion-{horizon_seq}-{horizon_up}-{horizon_low}-{interval}",
        config=vars(args),
    )
    print("\n=================== Successfully create wandb ===================\n")

    #=================================================#
    #====================Env setting==================#
    #=================================================#

    set_seed(seed)
    env = gym.make(env_name)
    
    if 'Fetch' in env_name:
        observation_dim = env.observation_space['observation'].shape[0] 
    else:
        observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = env.action_space
    
    dataset = SequenceDatasetV2(env_name, horizon=horizon_seq, returns_scale=scale, 
                                termination_penalty=None, load_path=None)
    normalizer:DatasetNormalizer = dataset.normalizer

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True, worker_init_fn=seed_worker)
    print("\n=================== Successfully set environment ===================\n")

    #=================================================#
    #======================Save dir===================#
    #=================================================#

    current_file = get_dir()

    results_dir = os.path.join(current_file, f"results/{env_name}-{model}/{seed}-Horzion-{horizon_seq}-{horizon_up}-{horizon_low}-{interval}") 
    os.makedirs(results_dir, exist_ok=True)  
    print("\n=================== Successfully create dir ===================\n")

    #=================================================#
    #=======================Logger====================#
    #=================================================#
        
    variant = vars(args)
    variant.update(version=f"Adaptive-Planning-Hierarchical-Diffuser-Policies-RL")
    variant.update(observation_dim=observation_dim)
    variant.update(action_dim=action_dim)
    variant.update(horizon_seq=horizon_seq)
    variant.update(horizon_up=horizon_up)
    variant.update(horizon_low=horizon_low)
    variant.update(interval=interval)
    variant.update(n_timesteps=n_timesteps)
    variant.update(lr=lr)
    variant.update(w=w)
    variant.update(threshold_scale=threshold_scale)
    variant.update(rtg=rtg)
    variant.update(dim_mults_up=dim_mults_up)
    variant.update(dim_mults_low=dim_mults_low)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)

    print("\n=================== Successfully log information ===================\n")

    #=================================================#
    #======================Training===================#
    #=================================================#
    
    policy = Policy(env_name,
                observation_dim,
                action_dim,
                action_scale,
                horizon_seq,
                horizon_up,
                horizon_low,
                interval,
                device,
                w=w,
                threshold_scale=threshold_scale,
                discount=gamma,
                tau=tau,
                n_timesteps=n_timesteps,
                mamba_layers=mamba_layers,
                outter_layers=outter_layers,
                attn_layers=attn_layers,
                eval_episodes=eval_episodes,
                random_len=random_len,
                lr=lr,
                schedule=schedule,
                model=model,
                dim_mults_up=dim_mults_up,
                dim_mults_low=dim_mults_low,
                args=args,
                update_every=update_every,
                use_cfg=use_cfg,
                use_adapt_art=use_adapt_art,
                use_diffusion=use_diffusion)
    
    
    stop_check = EarlyStopping(tolerance=1, min_delta=0.)

    print("=================TRAINING START=================")
    print(f"[ Training dataset ] {env_name}")
    print(f"[ Normalized scale ] {scale}")

    for epoch in range(epochs):
        epoch += 1
        metric = {'loss/planner_low': [], 'loss/planner_up': [], 
                  'loss/model': [], 'loss/critic': [], 
                  'loss/value': [], 'loss/actor': []}
        
        print(f"[ Current epoch ] {epoch}/{epochs}")

        # train
        for _, batch in enumerate(cycle(dataloader)):
            cnt += 1
            loss: Dict = policy.train(batch, cnt)
            policy.step_ema(cnt, step_start)
            policy.step_critic(cnt, step_start)
            policy.step_value(cnt, step_start)

            metric['loss/planner_low'].append(loss['loss/planner_low'])
            metric['loss/planner_up'].append(loss['loss/planner_up'])
            metric['loss/model'].append(loss['loss/model'])
            metric['loss/critic'].append(loss['loss/critic'])
            metric['loss/value'].append(loss['loss/value'])
            metric['loss/actor'].append(loss['loss/actor'])  
            wandb_logger.log(data=loss, step=cnt)

            if cnt % eval_freq == 0: break

        # eval
        reward_td, sps_td = policy.evaluate(env_name, eval_episodes, normalizer, rtg, scale, use_diffusion=True, show_progress=True, seed=seed)
        
        # record
        loss = {
                'loss/planner_low': np.mean(metric['loss/planner_low']), 
                'loss/planner_up': np.mean(metric['loss/planner_up']), 
                'loss/model': np.mean(metric['loss/model']), 
                'loss/critic': np.mean(metric['loss/critic']), 
                'loss/value': np.mean(metric['loss/value']), 
                'loss/actor': np.mean(metric['loss/actor'])}

        record(epoch, loss, reward_td, sps_td)
        wandb_logger.log(data=reward_td, step=cnt)
        wandb_logger.log(data=sps_td, step=cnt)

        # early_stop
        if early_stop:
            stop_model = stop_check(model_loss_t, loss['loss/model'])
            stop_planner = stop_check(planner_loss_t, loss['loss/planner_low'])
            model_loss_t = loss['loss/model']
            planner_loss_t = loss['loss/planner_low']

            if stop_model and stop_planner: break

        # save
        if reward_td["reward/max"] > max_reward:
            max_reward = reward_td["reward/max"]
            policy.save(results_dir + "/best_max.pth")

        if reward_td["reward/avg"] > avg_reward:
            avg_reward = reward_td["reward/avg"]
            policy.save(results_dir + "/best_avg.pth")

        # policy.save(results_dir + f"/{epoch}.pth")

    policy.save(results_dir + "/last.pth")

    print("\n=================== Algorithm Training complete ===================\n")

if __name__ == "__main__":
    main()