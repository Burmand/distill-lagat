import random
import torch
from pogema import GridConfig
from lagat.model import GNNPlanner
from lagat.run_planners_rl import run_neural_policy_rl
from lagat.run_planners import get_solution_quality
random.seed(0)
torch.manual_seed(0)

NUM_BATCHES = 20
K = 8 # Episodes per batch
MAX_STEPS = 128

# load the model
model = GNNPlanner.reconstruct("assets/pretrained/success_best.pt")
# Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in range(NUM_BATCHES):
    losses, costs, successes = [], [], []

    for k in range(K):
        num_agents = random.choice([2, 3, 4])
        grid_config = GridConfig(
            num_agents=num_agents, size=10, density=0.2, seed=batch * K +k,
            max_episode_steps=MAX_STEPS, obs_radius=5,
            observation_type="MAPF", on_target="nothing",
            collision_system="soft",)

        # Play one episode with run_neural_policy_rl
        env, actions, observations, terminated, log_probs = run_neural_policy_rl(model, grid_config)
        # Calculate cost
        quality = get_solution_quality(actions, observations, terminated)
        C = quality["sum_of_costs"]

        if not quality["solved"]:
            C = MAX_STEPS * num_agents   # Failure penalty instead of 0

        # Compute this episode's loss and append it to losses
        loss = C * torch.stack(log_probs).sum()
        losses.append(loss)

        costs.append(C)
        successes.append(quality["solved"])

    loss = torch.stack(losses).mean()
    optimizer.zero_grad() # Clear gradients from the previous batch
    loss.backward() # Backpropagation. Compute how each weight should change
    optimizer.step() # Gradient descent step. Update the weights

    print(f"batch {batch}: mean cost {sum(costs)/K:.1f}, success {sum(successes)/K:.2f}, loss {loss.item():.2f}")