import random
import torch
from pogema import GridConfig
from lagat.model import GNNPlanner
from lagat.run_planners_rl import run_neural_policy_rl
from lagat.run_planners import get_solution_quality
from lagat.features import get_cost_to_go_matrices

random.seed(0)
torch.manual_seed(0)    

FROM_SCRATCH = False  # False = Pretrained model
LOSS = "delay" # "cost" = C * log P, "delay" = C - lower bound
BASELINE = True # True = substract the batch average B, False = plain loss 

NUM_BATCHES = 50
K = 8 # Episodes per batch
MAX_STEPS = 128
AGENT_COUNTS = [8,9,10]
MAP_SIZE = 20


# load the model
if FROM_SCRATCH:
    model = GNNPlanner()
else:
    model = GNNPlanner.reconstruct("assets/pretrained/success_best.pt")
# Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 if FROM_SCRATCH else 1e-5)


TEST_SEEDS = range(10000, 10020) # 20 Fixed scenarios, different from seeds used in training

# Play the fixed test scenarios with greedy moves. Returns success rate, mean cost ratio
@torch.no_grad() # Testing only, no gradients needed
def evaluate(model):
    ratios, successes = [], []
    for i, seed in enumerate(TEST_SEEDS):
        num_agents = [2,3,4][i % 3]

        grid_config = GridConfig(
            num_agents=AGENT_COUNTS[i % len(AGENT_COUNTS)], size=MAP_SIZE, density=0.2, seed=seed,
            max_episode_steps=MAX_STEPS, obs_radius=5,
            observation_type="MAPF", on_target="nothing",
            collision_system="soft",
        )

        # Play one episode with run_neural_policy_rl
        env, actions, observations, terminated, log_probs = run_neural_policy_rl(model, grid_config, greedy = True)

        # Calculate cost
        quality = get_solution_quality(actions, observations, terminated)
        C = quality["sum_of_costs"]

        successes.append(quality["solved"])
        if quality["solved"]:
            # Cost ratio: total cost / lower bound where 1.0 is perfect
            ratios.append(quality["sum_of_costs_subopt"])

    success_rate = sum(successes) / len(successes)
    mean_ratio = sum(ratios) / len(ratios) if ratios else float("nan") # Return nan if nothing was solved

    return success_rate, mean_ratio

# How good is the starting model?
print("test before training:", evaluate(model))

for batch in range(NUM_BATCHES):
    log_prob_sums, costs, successes = [], [], []

    for k in range(K):
        num_agents = random.choice(AGENT_COUNTS)
        grid_config = GridConfig(
            num_agents=num_agents, size=MAP_SIZE, density=0.2, seed=batch * K +k,
            max_episode_steps=MAX_STEPS, obs_radius=5,
            observation_type="MAPF", on_target="nothing",
            collision_system="soft",
        )

        # Play one episode with run_neural_policy_rl
        env, actions, observations, terminated, log_probs = run_neural_policy_rl(model, grid_config)
        # Calculate cost
        quality = get_solution_quality(actions, observations, terminated)
        C = quality["sum_of_costs"]

        if not quality["solved"]:
            D = get_cost_to_go_matrices(observations[0]) # Distance maps, one per agent
            remaining = sum([D[i][observations[-1][i]["global_xy"]] for i in range (num_agents)]) # -1 = last step
            C = MAX_STEPS * num_agents + remaining # Failure cost is time limit + remaining distance
        
        if LOSS == "delay":
            C = C - quality["sum_of_costs_lb"]

        C = C / num_agents # cost per agent, so that episodes with different agent counts are comparable
        
        # Sum of log P of all moves in this episode
        log_prob_sums.append(torch.stack(log_probs).sum())

        costs.append(C)
        successes.append(quality["solved"])

    B = sum(costs) / K if BASELINE else 0 # 0 for the plain cost loss, batch average cost for the baseline loss
    loss = torch.stack([(c - B) * lp for c, lp in zip(costs, log_prob_sums)]).mean()

    optimizer.zero_grad() # Clear gradients from the previous batch
    loss.backward() # Backpropagation. Compute how each weight should change
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # cap the update size
    optimizer.step() # Gradient descent step. Update the weights

    print(f"batch {batch}: mean cost {sum(costs)/K:.1f}, success {sum(successes)/K:.2f}, loss {loss.item():.2f}")
    # Every 5 batches: test on the fixed scenarios
    if (batch + 1) % 5 == 0:
        print("test:", evaluate(model))

# Save the trained model (weights + settings) so eval_model.py can test it
model.save(f"outputs/rl_{'scratch' if FROM_SCRATCH else 'pretrained'}_{LOSS}_{'base' if BASELINE else 'nobase'}.pt")