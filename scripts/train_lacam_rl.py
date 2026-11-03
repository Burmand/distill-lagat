import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from pogema import GridConfig
from lagat.interfaces.lagat.inference import LagatInference, LagatInferenceConfig
from lagat.run_planners import run_full_horizon_planner, get_solution_quality
from lagat.model import GNNPlanner
import tempfile
import os
import random

random.seed(0)
torch.manual_seed(0)   

FROM_SCRATCH = False # False = Pretrained model
LOSS = "delay" # "cost" = C * log P, "delay" = C - lower bound
BASELINE = True # True = substract the batch average B, False = plain loss 

NUM_BATCHES = 5
K = 8 # Episodes per batch
MAX_STEPS = 128
AGENT_COUNTS = [4]
MAP_SIZE = 20
ENTROPY_ALPHA = 1e-3

# LaCAM starts with the pretrained model; after each update it uses our updated copy
current_jit = "assets/pretrained/success_best_jit.pt"
JIT_PATH = "outputs/lacam_rl_jit.pt"
RECORD_PATH = os.path.join(tempfile.gettempdir(), "recorded_x.pt")

# Python copy of the model with the same weights LaCAM used
model = GNNPlanner.reconstruct("assets/pretrained/success_best.pt")

# Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 if FROM_SCRATCH else 1e-5)

TEST_SEEDS = range(10000, 10020) # 20 Fixed scenarios, different from seeds used in training

# Play the fixed test scenarios with greedy moves. Returns success rate, mean cost ratio
@torch.no_grad() # Testing only, no gradients needed
def evaluate(model):
    ratios, successes = [], []
    for i, seed in enumerate(TEST_SEEDS):

        grid_config = GridConfig(
            num_agents=AGENT_COUNTS[i % len(AGENT_COUNTS)], size=MAP_SIZE, density=0.2, seed=seed,
            max_episode_steps=MAX_STEPS, obs_radius=5,
            observation_type="MAPF", on_target="nothing",
            collision_system="soft",
        )

        # LaCAM plays the test scenario with the current model (argmax, no learning)
        planner = LagatInference(cfg=LagatInferenceConfig(model_path=current_jit, lns=False))
        env, actions, observations, terminated = run_full_horizon_planner(planner, grid_config)

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
    # LaCAM uses the newest model in this batch
    planner = LagatInference(cfg=LagatInferenceConfig(model_path=current_jit, lns=False, sampling=True))
    log_prob_sums, costs, successes, entropies = [], [], [], []
    entropies_per_decision = [] # For printing


    for k in range(K):
        planner.cfg.seed = batch * K + k # LaCAM seed per run
        num_agents = random.choice(AGENT_COUNTS)
        grid_config = GridConfig(
            num_agents=num_agents, size=MAP_SIZE, density=0.2, seed=batch * K +k,
            max_episode_steps=MAX_STEPS, obs_radius=5,
            observation_type="MAPF", on_target="nothing",
            collision_system="soft",
        )

        # Remove the previous episode's file so we never train on old x's
        if os.path.exists(RECORD_PATH):
            os.remove(RECORD_PATH)

        # LaCAM solves the scenario; C++ records every x the model saw
        env, actions, observations, terminated = run_full_horizon_planner(planner, grid_config)

        # Calculate cost
        quality = get_solution_quality(actions, observations, terminated)
        # Failed search: no recorded x, nothing to learn from
        if not quality["solved"]:
            continue

        # Load the x's LaCAM just recorded for this scenario
        node_features, edge_index, edge_attr, recorded_L = torch.load(RECORD_PATH, weights_only=False)

        total_log_pi = 0
        total_entropy = 0
        for i in range(len(node_features)):
            # One recorded situation x, packed the way the model expects
            data = Data(x = node_features[i], edge_index=edge_index[i], edge_attr=edge_attr[i])
            # Model scores for every agent's 5 moves
            logits = model(data)
            # log probabilities of the 5 moves for each agent
            log_probs = F.log_softmax(logits, dim = -1)
            # Turn log probabilities to probabilities
            probs = log_probs.exp()
            # Entropy of this x
            x_entropy = -(probs * log_probs).sum()
            # Entropy of the episode
            total_entropy = total_entropy + x_entropy
            # L - each agent's top move
            #top_move = logits.argmax(dim = -1, keepdim = True)
            top_move = recorded_L[i].unsqueeze(1) # Log pi of the move LaCAM actually sampled at this x
            # log pi (L|x) of this situation
            log_pi = log_probs.gather(1, top_move).sum()
            total_log_pi = total_log_pi + log_pi
        print(f"log pi(L|x) of the episode: {total_log_pi}")
        # Average entropy per agent decision, comparable between episodes of different length
        entropies_per_decision.append(total_entropy.item() / (len(node_features) * num_agents))
        

        C = quality["sum_of_costs"]
        if LOSS == "delay":
            C = C - quality["sum_of_costs_lb"]
        C = C / num_agents
        costs.append(C)
        log_prob_sums.append(total_log_pi)
        entropies.append(total_entropy)

    # Baseline: average cost of this batch
    B = sum(costs) / len(costs) if BASELINE else 0
    loss = torch.stack([(c - B) * lp for c, lp in zip(costs, log_prob_sums)]).mean()
    loss = loss - ENTROPY_ALPHA * torch.stack(entropies).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss:", loss.item())
    print("gradient on first weight exists:", next(model.parameters()).grad is not None)
    print(f"Entropy: {sum(entropies_per_decision) / len(entropies_per_decision):.3f}")
    # Save the updated model so LaCAM can load it
    model.save("outputs/lacam_rl.pt")
    model_copy = GNNPlanner.reconstruct("outputs/lacam_rl.pt")
    model_copy.eval()

    # One example input so PyTorch can record the model's steps for C++
    jit_model = torch.jit.trace(model_copy, {"x": node_features[0], "edge_index": edge_index[0], "edge_attr": edge_attr[0]})
    jit_model.save(JIT_PATH)
    # From now on LaCAM uses the updated model
    current_jit = JIT_PATH

    # Check progress on the fixed test scenarios (cost, not loss)
    if (batch + 1) % 5 == 0:
        print("test:", evaluate(model))
