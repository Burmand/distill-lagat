"""
Checks that Python can call the C++ LaGAT and get a solution back
"""
from pogema import GridConfig
from lagat.interfaces.lagat.inference import LagatInference, LagatInferenceConfig
from lagat.run_planners import run_full_horizon_planner, get_solution_quality
import torch
import tempfile
import os

grid_config = GridConfig(
    num_agents=4, size=10, density=0.2, seed=0,
    max_episode_steps=128, obs_radius=5,
    observation_type="MAPF", on_target="nothing", collision_system="soft",
)



# LaCAM guided by the pretrained model. LNS is off so the solution is not improved afterwards
lagat_config = LagatInferenceConfig(model_path="assets/pretrained/success_best_jit.pt", lns = False, sampling = True, seed = 1, tau = 1.0)
planner = LagatInference(cfg=lagat_config)

# LaCAM plans the solution and then pogema plays it step by step
env, actions, observations, terminated = run_full_horizon_planner(planner=planner, grid_config=grid_config)

path = os.path.join(tempfile.gettempdir(), "recorded_x.pt")
node_features, edge_index, edge_attr, recorded_L, valid = torch.load(path, weights_only=False)
print(f"How many x's: {len(node_features)}")
print(f"How many L's: {len(recorded_L)}")
print(f"Shape node_feautres: {node_features[0].shape}")
print(f"Shape edge_index: {edge_index[0].shape}")
print(f"Shape edge_attr: {edge_attr[0].shape}")
print(f"Shape recorded_L: {recorded_L[0].shape}")
print(f"Shape valid: {valid[0].shape}")

print(get_solution_quality(actions, observations, terminated))

# Check: without sampling, the recorded L must be the model's own top move
from torch_geometric.data import Data
from lagat.model import GNNPlanner

model = GNNPlanner.reconstruct("assets/pretrained/success_best.pt")

matches = 0
for k in range(len(node_features)):  # one recorded x at a time
    data = Data(x=node_features[k], edge_index=edge_index[k], edge_attr=edge_attr[k])
    logits = model(data)  # [num_agents, 5], one score per move
    top_move = logits.argmax(-1)  # the model's own top move per agent
    matches += (top_move == recorded_L[k]).sum().item()  # how many agents agree with LaCAM's L

print(f"L matches model's top move: {matches} / {len(node_features) * 4}")
