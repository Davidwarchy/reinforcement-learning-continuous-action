
# Solve Continuous Action Problem with Reinforcement Learning 

<video controls width="640" height="360">
  <source src="combined_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Environment Description 
A rod suspending on one end from a pivot point. 

The upward position is angle 0. Angle from upward position is $\theta$. 

The downward position is angle pi (-pi), $\theta=\pi$. 

Clockwise motion from upward position decreases $\theta$. Anticlockwise increases $\theta$. 

## Inputs and Outputs
Official description of pendulum can be found [here](https://github.com/openai/gym/wiki/Pendulum-v1)

### Observation (inputs) 
We only need angle $\theta$ and $\dot{\theta}$. 

But in order to have a smooth description of the observation space, we need to have $sin(\theta)$ and $cos(\theta)$. So in total we have 3 inputs. 

| Num | Observation  | Min   | Max   |
|-----|--------------|-------|-------|
| 0   | cos(θ)       | -1.0  | 1.0   |
| 1   | sin(θ)       | -1.0  | 1.0   |
| 2   | $\dot{\theta}$        | -8.0  | 8.0   |


### Actions (outputs)
We have one action - a torque applied to the pendulum. Torque is directly proportion to acceleration 

The torque causes an angular acceleration 𝛼 that is applied continuously throughout the time step. This angular acceleration changes the angular velocity 𝜔 over that time. The angular velocity is updated, which in turn affects the angle 𝜃 of the pendulum. 

| Num | Action        | Min   | Max   |
|-----|---------------|-------|-------|
| 0   | Joint effort  | -2.0  | 2.0   |

## Reward Description
Reward = $-(𝜃^2 + 0.1*\dot{\theta}^2 + 0.001*action^2)$

## Training
This code implements a reinforcement learning agent using the DDPG (Deep Deterministic Policy Gradient) algorithm. 

The agent learns by interacting with the environment and improving its policy based on the rewards it receives.

The agent is trained for up to 200 episodes, and the environment is recorded as a video. The training is considered successful when the average reward over the last 10 episodes exceeds -200.