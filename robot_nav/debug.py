from pathlib import Path

import torch
import numpy as np
import logging
import time
import random
from shapely.geometry import Point

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
if __name__ == "__main__":
    # ---- DEBUG: inspect checkpoint vs. current model keys ----
    # _ckpt_path = Path("robot_nav/models/MARL/marlTD3/checkpoint/Mar.15_obstacle_14robot_reward8/TD3-MARL-obstacle-14robots_actor.pth")
    _ckpt_path = Path("robot_nav/models/MARL/marlTD3/checkpoint/obstacle_6robots_v4/TD3-MARL-obstacle-6robots-reward6_actor.pth")
    _ckpt = torch.load(_ckpt_path, map_location=torch.device("cpu"))
    print("\n=== Checkpoint actor keys ===")
    for k in _ckpt.keys():
        print(f"  {k}")
    # Build a temporary actor to compare expected keys
    from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import ActorObstacle
    _tmp_actor = ActorObstacle(
        action_dim=2,
        embedding_dim=256,  # matches TD3Obstacle
    )
    _model_keys = set(_tmp_actor.state_dict().keys())
    _ckpt_keys  = set(_ckpt.keys())
    print("\n=== Keys in model but MISSING from checkpoint ===")
    for k in sorted(_model_keys - _ckpt_keys):
        print(f"  {k}")
    print("\n=== Keys in checkpoint but UNEXPECTED by model ===")
    for k in sorted(_ckpt_keys - _model_keys):
        print(f"  {k}")
    del _tmp_actor, _ckpt

#   attention.message_graph.q.weight
#   attention.message_graph.k.weight
#   attention.message_graph.v.weight
#   attention.message_graph.v.bias
#   attention.message_graph.attn_score_layer.0.weight
#   attention.message_graph.attn_score_layer.0.bias
#   attention.message_graph.attn_score_layer.2.weight
#   attention.message_graph.attn_score_layer.2.bias
#   attention.embedding1.weight
#   attention.embedding1.bias
#   attention.embedding2.weight
#   attention.embedding2.bias
#   attention.hard_mlp.0.weight
#   attention.hard_mlp.0.bias
#   attention.hard_mlp.2.weight
#   attention.hard_mlp.2.bias
#   attention.hard_encoding.weight
#   attention.hard_encoding.bias
#   attention.hard_mlp_obs.0.weight
#   attention.hard_mlp_obs.0.bias
#   attention.hard_mlp_obs.2.weight
#   attention.hard_mlp_obs.2.bias
#   attention.hard_encoding_obs.weight
#   attention.hard_encoding_obs.bias
#   attention.decode_1.weight
#   attention.decode_1.bias
#   attention.decode_2.weight
#   attention.decode_2.bias
#   policy_head.0.weight
#   policy_head.0.bias
#   policy_head.2.weight
#   policy_head.2.bias
#   policy_head.4.weight
#   policy_head.4.bias