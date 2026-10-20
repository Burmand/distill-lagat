"""
Logic of the GNN that decides how each agent should move.
For every agent it looks at:
- An area around the agent
- The nearby agents

And then gives 5 scores per agent, one for each possible move
"""
import torch
from pathlib import Path
import yaml
import torch.nn as nn
import torch_geometric


from dataclasses import asdict, dataclass


@dataclass(eq=False, repr=False)
class GNNPlanner(nn.Module):
    """
    Planner works in 3 steps:
    1. encoder: each agent summarizes its map
    2. GNN: agents share their summaries with neighbors
    3. decoder: turns the result into 5 scores
    """
    obs_radius: int = 5
    dim_hidden: int = 128
    num_encoder_layers: int = 2
    num_gnn_layers: int = 3
    num_decoder_layers: int = 2
    heads: int = 1

    # should be aligned with observations
    # The two channels are "obs_cost_to_go" - distance to goal and "obs_agent" - other agents in the window
    num_channels: int = 2
    edge_dim: int = 3
    num_classes: int = 5

    def __post_init__(self) -> None:
        super().__init__()

        # encoder
        # How many numbers one agent's map contains. For example: 2 channels * 11 * 11 = 242
        dim_first_layer = self.num_channels * (self.obs_radius * 2 + 1) ** 2
        # Flatten the map into one row of numbers
        layers = [nn.Flatten(1)]
        # num_encoder_layers times: add a Linear layer + ReLU
        for i in range(self.num_encoder_layers):
            layers.extend(
                [
                    nn.Linear(
                        self.dim_hidden if i > 0 else dim_first_layer, self.dim_hidden
                    ),
                    nn.ReLU(),
                ]
            )
        # Connect these steps so they run in order (Flatten, Linear, ReLU, Linear, ReLU)
        # Sizes of the numbers flowing through: 242 -> 128 -> 128
        self.encoder = nn.Sequential(*layers)

        # GNN
        self.gnn = torch_geometric.nn.models.GAT(
            in_channels=self.dim_hidden, # Each agent comes in as 128 numbers (encoder's output)
            hidden_channels=self.dim_hidden // self.heads, # Size of vectors between rounds (128)
            num_layers=self.num_gnn_layers, # 3 rounds of sharing information with neighbors. 1 round = 1 GNN layer
            out_channels=self.dim_hidden, # Each agent leaves as 128 numbers
            v2=True,
            edge_dim=self.edge_dim, # The links between agents have 3 numbers (dx, dy, distance)
            add_self_loops=False, # Agents already link to themselves in features.py
            heads=self.heads,
            concat=True,
        )

        # decoder
        # Start an empty list, input will be a row of 128 numbers per agent.
        layers = []
        # num_decoder_layers times: Add a Linear layer + ReLU
        for _ in range(self.num_decoder_layers):
            layers.extend(
                [
                    nn.Linear(self.dim_hidden, self.dim_hidden),
                    nn.ReLU(),
                ]
            )

        # Add one Linear layer 128 -> 5 that outputs the final scores. No ReLU because the scores can be negative
        layers.append(nn.Linear(self.dim_hidden, self.num_classes))

        # Connect the steps in order (Linear, ReLU, Linear, ReLU, Linear)
        self.decoder = nn.Sequential(*layers)

    # Runs the neural network. Output: raw scores (no softmax), shape [num_agents, 5]
    def forward(self, data, **_) -> torch.Tensor:
        x_enc = self.encoder(data["x"])
        x_comm = self.gnn(
            x_enc, edge_index=data["edge_index"], edge_attr=data["edge_attr"]
        )
        x = self.decoder(x_comm + x_enc)
        return x

    # Stores the trained model
    def save(self, filename: Path | str) -> None:
        torch.save(self.state_dict(), filename)
        with open(f"{filename}_hypra.yaml", "w") as f:
            yaml.dump(asdict(self), f)

    # Loads a saved model
    @staticmethod
    def reconstruct(fpath: Path | str, device: str = "cpu", **kwargs):
        with open(f"{fpath}_hypra.yaml", "r") as f:
            model = GNNPlanner(**{**yaml.safe_load(f), **kwargs})
        model.load_state_dict(torch.load(open(fpath, "rb"), map_location=device))
        return model
