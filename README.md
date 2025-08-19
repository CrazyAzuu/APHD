# Adaptive Planning Hierarchical Diffuser for Multi-Step Action Execution in Offline RL

## Motivation

## Architecture
![](./figures/APHD.png)
Overview of the proposed APHD method. Given an input trajectory sequence, the MTPN first autoregressively generates a feasible trajectory. This trajectory is then refined by a hierarchical diffusion model that constructs multi-horizon temporal abstractions. The final action sequence is derived by the inverse dynamics and executed via the adaptive replanning trigger. "Skip" denotes equidistant sampling of states, and $H-1$ represents autoregressive planning steps.

## Autoregressive Model
![](./figures/MTPN.png)
The architecture of MTPN. After embedding, the trajectory is processed in parallel by mamba (for local features) and an attention module (for global features). The fused representations are then decoded to predict state and return-to-go.

## Visualization Results
