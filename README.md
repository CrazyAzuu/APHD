# Adaptive Planning Hierarchical Diffuser for Multi-Step Action Execution in Offline RL

## Motivation

## Autoregressive Model
<div align="center">
  <img src="./figures/MTPN.png" width="600">
</div>
The architecture of MTPN. After embedding, the trajectory is processed in parallel by mamba (for local features) and an attention module (for global features). The fused representations are then decoded to predict state and return-to-go.

## Architecture
<div align="center">
  <img src="./figures/APHD.png">
</div>
Overview of the proposed APHD method. Given an input trajectory sequence, the MTPN first autoregressively generates a feasible trajectory. This trajectory is then refined by a hierarchical diffusion model that constructs multi-horizon temporal abstractions. The final action sequence is derived by the inverse dynamics and executed via the adaptive replanning trigger. "Skip" denotes equidistant sampling of states, and $H-1$ represents autoregressive planning steps.

## Visualization Results
### Hopper
hopper-medium-expert-v2
<img src="./figures/hopper-medium-expert-v2.png"> 
hopper-medium-replay-v2
<img src="./figures/hopper-medium-replay-v2.png">
hopper-medium-v2
<img src="./figures/hopper-medium-v2.png">

### Walker2d
walker2d-medium-expert-v2
<img src="./figures/walker2d-medium-expert-v2.png"> 
walker2d-medium-replay-v2
<img src="./figures/walker2d-medium-replay-v2.png">
walker2d-medium-v2
<img src="./figures/walker2d-medium-v2.png">

### Halfcheetah
halfcheetah-medium-expert-v2
<img src="./figures/halfcheetah-medium-expert-v2.png"> 
halfcheetah-medium-replay-v2
<img src="./figures/halfcheetah-medium-replay-v2.png">
halfcheetah-medium-v2
<img src="./figures/halfcheetah-medium-v2.png">

### Door
door-expert-v0
<img src="./figures/door-expert-v0.png"> 
door-cloned-v0
<img src="./figures/door-cloned-v0.png">
door-human-v0
<img src="./figures/door-human-v0.png">

### Pen
pen-expert-v0
<img src="./figures/pen-expert-v1.png"> 
pen-cloned-v0
<img src="./figures/pen-cloned-v1.png">
pen-human-v0
<img src="./figures/pen-human-v1.png">

### Hammer
hammer-expert-v0
<img src="./figures/hammer-expert-v0.png"> 
hammer-cloned-v0
<img src="./figures/hammer-cloned-v0.png">
hammer-human-v0
<img src="./figures/hammer-human-v0.png">
