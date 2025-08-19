import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from agents.uncond import Uncond_Transformer, Uncond_MambaPolicy, Uncond_MTPN_Policy


def get_dir():
    current_file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(current_file_path)
    return dir_path

def generate_data(num, device='cpu'):
    high_freq_prob = 0.8
    low_freq_prob = 0.2

    high_freq_num, low_freq_num = int(num * high_freq_prob), int(num * low_freq_prob)

    high_freq_alpha = 5
    low_freq_alpha = 3

    trajectory_lenght = 64
    # f(x) = cos(\alpha * 2 * pi x)
    # generate_trajectory
    high_freq_x = np.linspace(0, 1, trajectory_lenght)
    low_freq_x = np.linspace(0, 1, trajectory_lenght)

    # expand to high_freq_num, low_freq_num
    high_freq_y = np.cos(high_freq_alpha * 2 * np.pi * high_freq_x) + np.random.normal(0, 0.1, (high_freq_num, trajectory_lenght))
    
    low_freq_y = np.cos(low_freq_alpha * 2 * np.pi * low_freq_x) + np.random.normal(0, 0.1, (low_freq_num, trajectory_lenght))

    return torch.tensor(np.concatenate([high_freq_y, low_freq_y], axis=0), dtype=torch.float32).to(device)

def fourier_analysis(samples):
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    samples = samples.squeeze()
    freq = fft(samples)
    abs_y = np.abs(freq)

    normalization_y=abs_y/32
    normalization_y = normalization_y[:32]

    return normalization_y

class Replaybuffer:
    def __init__(self, traj) -> None:
        self.traj = traj
    def sample(self, batch_size):
        indices = np.random.choice(self.traj.shape[0], batch_size)
        return self.traj[indices]
    
#=====================================================================#
#=============================== Prepare =============================#
#=====================================================================#
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

N = 256
state_dim = 1
device = 'cuda:0'
trajectories = generate_data(500, device)
replaybuffer = Replaybuffer(trajectories)

fs = 64 / 2
half_x = np.linspace(0, fs, 32)
fig1, axs1 = plt.subplots(2, 3, figsize=(5 * 4, 6))
fig2, axs2 = plt.subplots(2, 3, figsize=(5 * 4, 6))

transfomer = Uncond_Transformer(state_dim, 32, 2, 64, 2, 0.1).to(device)
mtpn = Uncond_MTPN_Policy(input_dim=state_dim, d_model=64, d_ff=128, mamba_dim=512, n_heads=2, mamba_layers=3, outter_layers=1, attn_layers=1,
                          dropout_rate=0.1, ssm_cfg={"d_state":64, "expand":2}, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, device=device).to(device)

optimizer_transfomer = torch.optim.Adam(transfomer.parameters(), lr=1e-3)
optimizer_mtpn = torch.optim.Adam(mtpn.parameters(), lr=1e-3)

print("\n=================== Successfully prepare data ===================\n")

#=====================================================================#
#=========================== Plot Train Data =========================#
#=====================================================================#

x = np.linspace(0, 2, 64)
for i in range(trajectories.shape[0]):
    axs1[0][0].plot(x, trajectories[i].cpu().numpy(), alpha=0.01, c='blue')
    axs1[0][0].set_xlabel('x', fontsize=18)
    axs1[0][0].set_ylabel('y', fontsize=18)
    axs1[0][0].set_title('Train data', fontsize=18)
    freq = fourier_analysis(trajectories[i])
    axs1[1][0].plot(half_x, freq, alpha=0.01, c='red')
    axs1[1][0].set_xlabel('Frequency $f$/Hz', fontsize=18)
    axs1[1][0].set_ylabel('Magnitude', fontsize=18)

x = np.linspace(0, 2, 64)
for i in range(trajectories.shape[0]):
    axs2[0][0].plot(x, trajectories[i].cpu().numpy(), alpha=0.01, c='blue')
    axs2[0][0].set_xlabel('x', fontsize=18)
    axs2[0][0].set_ylabel('y', fontsize=18)
    axs2[0][0].set_title('Train data', fontsize=18)
    freq = fourier_analysis(trajectories[i])
    axs2[1][0].plot(half_x, freq, alpha=0.01, c='red')
    axs2[1][0].set_xlabel('Frequency $f$/Hz', fontsize=18)
    axs2[1][0].set_ylabel('Magnitude', fontsize=18)

print("\n=================== Successfully plot train Data ===================\n")

#=====================================================================#
#====================== Train Autoregressive Model ===================#
#=====================================================================#


for i in range(2000):
    batch = replaybuffer.sample(32)
    batch = batch.unsqueeze(-1)

    optimizer_transfomer.zero_grad()
    optimizer_mtpn.zero_grad()

    input = batch[:, :-1]
    target = batch[:, 1:]

    output_transfomer = transfomer(input)
    output_mtpn = mtpn(input)

    loss_transformer = F.mse_loss(output_transfomer, target)
    loss_mtpn = F.mse_loss(output_mtpn, target)

    loss_transformer.backward()
    optimizer_transfomer.step()

    loss_mtpn.backward()
    optimizer_mtpn.step()
    if i % 100 == 0:
        print(f'transformer loss: {loss_transformer.item()}')
        print(f'mtpn loss: {loss_mtpn.item()}')
        print("\n")

print("\n=================== Training autoregression completed ===================\n")

state = torch.tensor([[1.0]]).repeat_interleave(N, dim=0).unsqueeze(-1).to(device) #+ torch.randn(N, 1, 1).to(device) * 0.
transfomer.train()
with torch.no_grad():
    samples_transformer = transfomer.sample(state, 64)
x = np.linspace(0, 2, 64)
for i in range(N):
    axs1[0][1].plot(x, samples_transformer[i].cpu().numpy(), alpha=0.01, c='blue')
    axs1[0][1].set_xlabel('x', fontsize=18)
    axs1[0][1].set_ylabel('y', fontsize=18)
    axs1[0][1].set_title('Transformer planning', fontsize=18)
    freq = fourier_analysis(samples_transformer[i])
    axs1[1][1].plot(half_x, freq, alpha=0.01, c='red')
    axs1[1][1].set_xlabel('Frequency $f$/Hz', fontsize=18)
    axs1[1][1].set_ylabel('Magnitude', fontsize=18)

state = torch.tensor([[1.0]]).repeat_interleave(N, dim=0).unsqueeze(-1).to(device) #+ torch.randn(N, 1, 1).to(device) * 0.2
mtpn.train()
with torch.no_grad():
    samples_mtpn = mtpn.sample(state, 64)
x = np.linspace(0, 2, 64)
for i in range(N):
    axs2[0][1].plot(x, samples_mtpn[i].cpu().numpy(), alpha=0.01, c='blue')
    axs2[0][1].set_xlabel('x', fontsize=18)
    axs2[0][1].set_ylabel('y', fontsize=18)
    axs2[0][1].set_title('MTPN planning', fontsize=18)
    freq = fourier_analysis(samples_mtpn[i])
    axs2[1][1].plot(half_x, freq, alpha=0.01, c='red')
    axs2[1][1].set_xlabel('Frequency $f$/Hz', fontsize=18)
    axs2[1][1].set_ylabel('Magnitude', fontsize=18)

print("\n=================== Ploting autoregression completed ===================\n")

# #=====================================================================#
# #=========================== Train Diffusion =========================#
# #=====================================================================#

from agents.diffusion import Uncond_Diffusion_Lower as Diffusion_Lower
from agents.diffusion import Uncond_Diffusion_Upper as Diffusion_Upper
from agents.temporal import Uncond_TemporalUnet_Lower as TemporalUnet_Lower
from agents.temporal import Uncond_TemporalUnet_Upper as TemporalUnet_Upper

horizon_up = 2
horizon_low = 64
interval = 63
up_index = [interval * i for i in range(horizon_up)]

model_l = TemporalUnet_Lower(horizon_low, state_dim, dim=32, dim_mults=(1,2,4,8)).to(device)
model_u = TemporalUnet_Upper(horizon_up, state_dim, dim=32, dim_mults=(1,2)).to(device)
planner_l = Diffusion_Lower(state_dim, model_l, horizon=horizon_low, n_timesteps=10, predict_epsilon=True, beta_schedule='vp').to(device)# predict_epsilon=False
planner_u = Diffusion_Upper(state_dim, model_u, horizon=horizon_up, n_timesteps=10, predict_epsilon=True, beta_schedule='vp').to(device)# predict_epsilon=False

# train the model
optimizer_l = torch.optim.Adam(planner_l.parameters(), lr=2e-3)
optimizer_u = torch.optim.Adam(planner_u.parameters(), lr=2e-3)

for i in range(2000):
    batch = replaybuffer.sample(32)
    batch = batch.unsqueeze(-1)
    batch_up = batch[:, up_index,:]
    batch_low = batch[:, :horizon_low,:]

    optimizer_u.zero_grad()
    loss_u = planner_u.loss(batch_up)
    loss_u.backward()
    optimizer_u.step()

    optimizer_l.zero_grad()
    loss_l = planner_l.loss(batch_low, batch_up[:, 1])
    loss_l.backward()
    optimizer_l.step()

    if i % 100 == 0:
        print(f'low level loss: {loss_l.item()}')
        print(f'high level loss: {loss_u.item()}')
        print("\n")

print("\n=================== Training diffusion completed ===================\n")

with torch.no_grad():
    samples_transformer = transfomer.sample(state, 64)
    sub_goals = planner_u(samples_transformer[:, up_index, :])
    for i in range(horizon_up-1):
        
        sub_goal = sub_goals[:, i+1]
        low_samples = planner_l(samples_transformer[:,i*interval: i*interval+horizon_low], sub_goal)
        samples_transformer[:,i*interval: i*interval+horizon_low] = low_samples

with torch.no_grad():
    samples_mtpn = mtpn.sample(state, 64)
    sub_goals = planner_u(samples_mtpn[:, up_index, :])
    for i in range(horizon_up-1):
        
        sub_goal = sub_goals[:, i+1]
        low_samples = planner_l(samples_mtpn[:,i*interval: i*interval+horizon_low], sub_goal)
        samples_mtpn[:,i*interval: i*interval+horizon_low] = low_samples

x = np.linspace(0, 2, 64)
for i in range(N):
    axs1[0][2].plot(x, samples_transformer[i].cpu().numpy(), alpha=0.01, c='blue')
    axs1[0][2].set_xlabel('x', fontsize=18)
    axs1[0][2].set_ylabel('y', fontsize=18)
    axs1[0][2].set_title('Hierarchical diffusion optimization', fontsize=18)
    freq = fourier_analysis(samples_transformer[i])
    axs1[1][2].plot(half_x, freq, alpha=0.01, c='red')
    axs1[1][2].set_xlabel('Frequency $f$/Hz', fontsize=18)
    axs1[1][2].set_ylabel('Magnitude', fontsize=18)

x = np.linspace(0, 2, 64)
for i in range(N):
    axs2[0][2].plot(x, samples_mtpn[i].cpu().numpy(), alpha=0.01, c='blue')
    axs2[0][2].set_xlabel('x', fontsize=18)
    axs2[0][2].set_ylabel('y', fontsize=18)
    axs2[0][2].set_title('Hierarchical diffusion optimization', fontsize=18)
    freq = fourier_analysis(samples_mtpn[i])
    axs2[1][2].plot(half_x, freq, alpha=0.01, c='red')
    axs2[1][2].set_xlabel('Frequency $f$/Hz', fontsize=18)
    axs2[1][2].set_ylabel('Magnitude', fontsize=18)

print("\n=================== Ploting hierarchical diffusion completed ===================\n")

# save fig

current_file = get_dir()
fig1.tight_layout(rect=[0, 0.03, 1, 0.97])
results_dir = os.path.join(current_file, "toy_exp(1).pdf") 
fig1.savefig(results_dir, bbox_inches='tight')

fig2.tight_layout(rect=[0, 0.03, 1, 0.97])
results_dir = os.path.join(current_file, "toy_exp(2).pdf") 
fig2.savefig(results_dir, bbox_inches='tight')
