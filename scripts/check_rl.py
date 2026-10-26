import torch
from pogema import GridConfig
from lagat.model import GNNPlanner
from lagat.run_planners_rl import run_neural_policy_rl
from lagat.run_planners import get_solution_quality

model = GNNPlanner.reconstruct("assets/pretrained/success_best.pt")
grid_config = GridConfig(
    num_agents=4, size=10, density=0.2, seed=0,
    max_episode_steps=128, obs_radius=5,
    observation_type="MAPF", on_target="nothing", collision_system="soft",
)

env, actions, observations, terminated, log_probs = run_neural_policy_rl(model, grid_config, greedy = True)

# Calculate cost
quality = get_solution_quality(actions, observations, terminated)
C = quality["sum_of_costs"]
# Cost-based RL loss
loss = C * torch.stack(log_probs).sum()
loss.backward()

print("steps:", len(log_probs))
print("shape of one step:", log_probs[0].shape)
print("has gradient link:", log_probs[0].requires_grad)
print(quality)
print("solved:", quality["solved"], "cost C:", C)
print("loss:", loss.item())
print("gradient on first weight exists:", next(model.parameters()).grad is not None)