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

path = os.path.join(tempfile.gettempdir(), "recorded_x.pt")
node_features, edge_index, edge_attr = torch.load(path, weights_only=False)
print(f"How many x's: {len(node_features)}")
print(f"Shape node_feautres: {node_features[0].shape}")
print(f"Shape edge_index: {edge_index[0].shape}")
print(f"Shape edge_attr: {edge_attr[0].shape}")

# LaCAM guided by the pretrained model. LNS is off so the solution is not improved afterwards
lagat_config = LagatInferenceConfig(model_path="assets/pretrained/success_best_jit.pt", lns = False)
planner = LagatInference(cfg=lagat_config)

# LaCAM plans the solution and then pogema plays it step by step
env, actions, observations, terminated = run_full_horizon_planner(planner=planner, grid_config=grid_config)
print(get_solution_quality(actions, observations, terminated))
