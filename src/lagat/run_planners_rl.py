import torch
import torch.nn.functional as F
from pogema import pogema_v0, AnimationMonitor
from typing import Any

from .features import (
    get_cost_to_go_matrices,
    get_pyg_data_from_observations,
)
from .utils import suppress_stdout

# Like run_neural_policy, but keeps the log-probabilities of the chosen moves for RL training
# No @torch.no_grad(), We want to keep the gradients for the training
def run_neural_policy_rl(
    model,
    grid_config,
    animation: bool = False,
    max_episodes: int = 128,
    device: str = "cpu",
    sampling_temperature: float = 1.0,
    **_,
):
    model.eval()
    env = pogema_v0(grid_config=grid_config) # Create the environment
    if animation:
        env = AnimationMonitor(env)

    obs, infos = env.reset() # Place the agents. obs is a list with one dictionary per agent (position, goal, what it sees)
    cost_to_go_matrices = get_cost_to_go_matrices(obs) # Distance to each agent's goal
    params = dict(
        obs_radius=model.obs_radius,
        cost_to_go_matrices=cost_to_go_matrices,
    )

    all_actions = []
    all_observations = []
    all_terminated = []
    all_log_probs = [] # List of log-probabilities

    for _ in range(max_episodes):
        # sample action
        data = get_pyg_data_from_observations(obs, **params).to(device)

        # Turn the scores into probabilities
        # No detach(). probs must stay linked to the model so loss.backward() can learn from them
        probs = F.softmax(model(data) / sampling_temperature, dim=-1).cpu()

        sampled = torch.multinomial(probs, num_samples = 1) # Tensor with shape [num_agents, 1]
        all_log_probs.append(torch.log(probs).gather(1, sampled).squeeze(-1)) # Shape [num_agents]

        actions = sampled.squeeze(-1).numpy()
        all_observations.append(obs)
        all_actions.append(actions)
        obs, rewards, terminated, truncated, infos = env.step(actions)
        all_terminated.append(terminated)

        if all(terminated) or all(truncated):
            break

    return env, all_actions, all_observations, all_terminated, all_log_probs
