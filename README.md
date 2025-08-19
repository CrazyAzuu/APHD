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
Overview of the proposed APHD method. Given an input trajectory sequence, the MTPN first autoregressively generates a feasible trajectory. This trajectory is then refined by a hierarchical diffusion model that constructs multi-horizon temporal abstractions. The final action sequence is derived by the inverse dynamics and executed via the adaptive replanning trigger. "Skip" denotes equidistant sampling of states, and "H-1" represents autoregressive planning steps.

## Visualization Results
### Hopper-medium-expert-v2
<img src="./figures/hopper-medium-expert-v2.png"> 

### Hopper-medium-replay-v2
<img src="./figures/hopper-medium-replay-v2.png">

### Hopper-medium-v2
<img src="./figures/hopper-medium-v2.png">

### Walker2d-medium-expert-v2
<img src="./figures/walker2d-medium-expert-v2.png"> 

### Walker2d-medium-replay-v2
<img src="./figures/walker2d-medium-replay-v2.png">

### Walker2d-medium-v2
<img src="./figures/walker2d-medium-v2.png">

### Halfcheetah-medium-expert-v2
<img src="./figures/halfcheetah-medium-expert-v2.png"> 

### Halfcheetah-medium-replay-v2
<img src="./figures/halfcheetah-medium-replay-v2.png">

### Halfcheetah-medium-v2
<img src="./figures/halfcheetah-medium-v2.png">

### Door-expert-v0
<img src="./figures/door-expert-v0.png"> 

### Door-cloned-v0
<img src="./figures/door-cloned-v0.png">

### Door-human-v0
<img src="./figures/door-human-v0.png">

### Pen-expert-v0
<img src="./figures/pen-expert-v1.png"> 

### Pen-cloned-v0
<img src="./figures/pen-cloned-v1.png">

### Pen-human-v0
<img src="./figures/pen-human-v1.png">

### Hammer-expert-v0
<img src="./figures/hammer-expert-v0.png"> 

### Hammer-cloned-v0
<img src="./figures/hammer-cloned-v0.png">

### Hammer-human-v0
<img src="./figures/hammer-human-v0.png">

### Relocate-expert-v0
<img src="./figures/relocate-expert-v0.png"> 

### Relocate-cloned-v0
<img src="./figures/relocate-cloned-v0.png">

### Relocate-human-v0
<img src="./figures/relocate-human-v0.png">

### Maze2d-umaze-v1
<img src="./figures/maze2d-umaze-v1.png"> 

### Maze2d-medium-v1
<img src="./figures/maze2d-medium-v1.png">

### Maze2d-large-v1
<img src="./figures/maze2d-large-v1.png">
