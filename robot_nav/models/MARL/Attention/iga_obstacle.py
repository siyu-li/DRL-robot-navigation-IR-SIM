"""
Graph Attention Network with Obstacle Nodes (IGA-Obstacle).

This module extends the IGA attention mechanism to include static obstacle nodes
in the graph. Obstacles act as message senders only - they do not have queries
and do not produce actions. Robot nodes receive messages from both other robots
and obstacles.

Key differences from iga.py:
- Obstacle nodes are included in the graph as static senders
- Only robot nodes compute queries and produce output embeddings
- Edge features between robot-obstacle pairs use obstacle clearance
- Obstacles have zero velocity and no goal-related features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


def entropy_from_attention(attn_weights, target_nodes, num_nodes, eps=1e-10):
    """
    Compute mean per-node entropy of incoming softmax attention distributions.

    Args:
        attn_weights (Tensor): Per-edge attention probabilities of shape (num_edges,).
        target_nodes (Tensor): Destination node indices for each edge of shape (num_edges,).
        num_nodes (int): Total number of nodes (robots only for output).
        eps (float, optional): Small constant for numerical stability in log. Defaults to 1e-10.

    Returns:
        Tensor: Scalar tensor containing the mean entropy across nodes that have at least
            one incoming edge. Returns 0.0 if no nodes have incoming edges.
    """
    attn_log = (attn_weights + eps).log()
    contrib = -(attn_weights * attn_log)

    entropies = torch.zeros(num_nodes, device=attn_weights.device).index_add_(
        0, target_nodes, contrib
    )
    counts = torch.zeros(num_nodes, device=attn_weights.device).index_add_(
        0, target_nodes, torch.ones_like(attn_weights)
    )

    mask = counts > 0
    entropies = entropies[mask] / counts[mask]
    return (
        entropies.mean()
        if mask.any()
        else torch.tensor(0.0, device=attn_weights.device)
    )


class GoalAttentionLayerObstacle(MessagePassing):
    """
    Message-passing layer with learned attention over goal/edge features.

    Extends GoalAttentionLayer to handle heterogeneous graphs with robot and
    obstacle nodes. Only robot nodes (indices 0 to n_robots-1) are targets.

    Args:
        node_dim (int): Dimensionality of node features.
        edge_dim (int): Dimensionality of edge attributes.
        out_dim (int): Output dimensionality for projected messages.
    """

    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__(aggr="add")
        self.q = nn.Linear(node_dim, out_dim, bias=False)
        self.k = nn.Linear(edge_dim, out_dim, bias=False)
        self.v = nn.Linear(edge_dim, out_dim)
        self.attn_score_layer = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1),
        )
        self._last_attn_weights = None

    def forward(self, x, edge_index, edge_attr, n_robots):
        """
        Run attention-based message passing.

        Args:
            x (Tensor): Node features of shape (num_nodes, node_dim).
                        First n_robots nodes are robots, rest are obstacles.
            edge_index (LongTensor): Edge indices of shape (2, num_edges)
                with format [source, target]. Targets are only robot indices.
            edge_attr (Tensor): Edge attributes of shape (num_edges, edge_dim).
            n_robots (int): Number of robot nodes (only these are targets).

        Returns:
            tuple:
                out (Tensor): Updated robot node features of shape (n_robots, out_dim).
                attn_weights (Tensor or None): Last per-edge softmax weights.
        """
        # Only compute queries for robot nodes
        q_robots = self.q(x[:n_robots])  # (n_robots, out_dim)

        # Propagate with size hint: (num_all_nodes, n_robots)
        # This tells PyG that sources can be any node but targets are only robots
        out = self.propagate(
            edge_index,
            x=(x, q_robots),  # (source_features, target_queries)
            edge_attr=edge_attr,
            size=(x.size(0), n_robots),
        )
        return out, self._last_attn_weights

    def message(self, x_i, edge_attr, index, ptr, size_i):
        """
        Compute per-edge messages using attention weights.

        Args:
            x_i (Tensor): Target-node queries for each edge, shape (num_edges, out_dim).
            edge_attr (Tensor): Edge attributes for each edge, shape (num_edges, edge_dim).
            index (LongTensor): Target node indices per edge, shape (num_edges,).
            ptr: Unused (PyG internal).
            size_i: Number of target nodes.

        Returns:
            Tensor: Per-edge messages of shape (num_edges, out_dim).
        """
        k = F.leaky_relu(self.k(edge_attr))
        v = F.leaky_relu(self.v(edge_attr))
        attention_input = torch.cat([x_i, k], dim=-1)
        scores = self.attn_score_layer(attention_input).squeeze(-1)
        attn_weights = softmax(scores, index, num_nodes=size_i)
        self._last_attn_weights = attn_weights.detach()
        return v * attn_weights.unsqueeze(-1)


class AttentionObstacle(nn.Module):
    """
    Multi-robot attention mechanism with obstacle nodes.

    Extends the Attention module to include static obstacle nodes in the graph.
    Obstacles act as message senders only - robot nodes receive messages from
    both other robots and nearby obstacles.

    Node layout: [robot_0, robot_1, ..., robot_{n-1}, obs_0, obs_1, ..., obs_{m-1}]

    Args:
        embedding_dim (int): Dimension of the agent embedding vector.
        robot_node_dim (int): Input dimension for robot node features. Default 5.
        obstacle_node_dim (int): Input dimension for obstacle node features. Default 5.

    Robot node features (5-dim, extracted from state indices 4:9):
        [distance_to_goal, cos(heading_error), sin(heading_error), lin_vel, ang_vel]

    Obstacle node features (5-dim):
        [0, 0, 0, 0, 0] for static obstacles (placeholder, learned embedding)
    """

    def __init__(self, embedding_dim, robot_node_dim=5, obstacle_node_dim=5):
        """
        Initialize the attention module with obstacle support.

        Args:
            embedding_dim (int): Output embedding dimension per agent.
            robot_node_dim (int): Input feature dim for robots. Default 5.
            obstacle_node_dim (int): Input feature dim for obstacles. Default 5.
        """
        super(AttentionObstacle, self).__init__()
        self.embedding_dim = embedding_dim

        # Message passing layer (obstacle-aware)
        # Edge dim: 7 (robot-robot) + 3 (goal polar) = 10
        # For robot-obstacle edges, goal polar is zeroed
        self.message_graph = GoalAttentionLayerObstacle(
            node_dim=embedding_dim, edge_dim=10, out_dim=embedding_dim
        )

        # Robot node encoder (same as original)
        self.embedding1 = nn.Linear(robot_node_dim, 128)
        nn.init.kaiming_uniform_(self.embedding1.weight, nonlinearity="leaky_relu")
        self.embedding2 = nn.Linear(128, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding2.weight, nonlinearity="leaky_relu")

        # Obstacle node encoder (separate, same architecture)
        self.obs_embedding1 = nn.Linear(obstacle_node_dim, 128)
        nn.init.kaiming_uniform_(self.obs_embedding1.weight, nonlinearity="leaky_relu")
        self.obs_embedding2 = nn.Linear(128, embedding_dim)
        nn.init.kaiming_uniform_(self.obs_embedding2.weight, nonlinearity="leaky_relu")

        # Hard attention MLP (for robot-robot edges only, or all edges)
        # Input: embedding + edge features (7-dim: dist, rel_angle_cos/sin, heading_j_cos/sin, action)
        self.hard_mlp = nn.Sequential(
            nn.Linear(embedding_dim + 7, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.hard_encoding = nn.Linear(embedding_dim, 2)

        # Hard attention for robot-obstacle edges
        # Input: robot embedding + obstacle edge features (5-dim: dist, rel_angle_cos/sin, obs_heading_cos/sin)
        self.hard_mlp_obs = nn.Sequential(
            nn.Linear(embedding_dim + 5, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.hard_encoding_obs = nn.Linear(embedding_dim, 2)

        # Decoder (same as original)
        self.decode_1 = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        nn.init.kaiming_uniform_(self.decode_1.weight, nonlinearity="leaky_relu")
        self.decode_2 = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        nn.init.kaiming_uniform_(self.decode_2.weight, nonlinearity="leaky_relu")

    def encode_robot_features(self, embed):
        """
        Encode per-robot features with a two-layer MLP.

        Args:
            embed (Tensor): Raw robot features of shape (B*N_robots, 5).

        Returns:
            Tensor: Encoded embeddings of shape (B*N_robots, embedding_dim).
        """
        embed = F.leaky_relu(self.embedding1(embed))
        embed = F.leaky_relu(self.embedding2(embed))
        return embed

    def encode_obstacle_features(self, embed):
        """
        Encode per-obstacle features with a two-layer MLP.

        Args:
            embed (Tensor): Raw obstacle features of shape (B*N_obs, 5).

        Returns:
            Tensor: Encoded embeddings of shape (B*N_obs, embedding_dim).
        """
        embed = F.leaky_relu(self.obs_embedding1(embed))
        embed = F.leaky_relu(self.obs_embedding2(embed))
        return embed

    def forward(self, robot_embedding, obstacle_embedding):
        """
        Compute hard and soft attentions with obstacle nodes and produce attended embeddings.

        Args:
            robot_embedding (Tensor): Robot states of shape (B, N_robots, 11).
                Layout: [px, py, cos_h, sin_h, dist/17, cos_err, sin_err, lin_v, ang_v, gx, gy]
            obstacle_embedding (Tensor): Obstacle info of shape (B, N_obs, 4).
                Layout: [ox, oy, cos_h, sin_h] (static heading can be arbitrary)

        Returns:
            tuple:
                att_embedding (Tensor): Attended embedding for robots only,
                    shape (B*N_robots, 2*embedding_dim).
                hard_logits_rr (Tensor): Robot-robot hard attention logits,
                    shape (B*N_robots, N_robots).
                hard_logits_ro (Tensor): Robot-obstacle hard attention logits,
                    shape (B*N_robots, N_obs).
                unnorm_rel_dist_rr (Tensor): Unnormalized robot-robot distances,
                    shape (B*N_robots, N_robots, 1).
                unnorm_rel_dist_ro (Tensor): Unnormalized robot-obstacle distances,
                    shape (B*N_robots, N_obs, 1).
                mean_entropy (Tensor): Scalar mean entropy of soft attention.
                hard_weights_rr (Tensor): Binary robot-robot hard mask, shape (B, N_robots, N_robots).
                hard_weights_ro (Tensor): Binary robot-obstacle hard mask, shape (B, N_robots, N_obs).
                comb_w (Tensor): Combined soft weights for visualization.
        """
        if robot_embedding.dim() == 2:
            robot_embedding = robot_embedding.unsqueeze(0)
        if obstacle_embedding.dim() == 2:
            obstacle_embedding = obstacle_embedding.unsqueeze(0)

        batch_size, n_robots, _ = robot_embedding.shape
        _, n_obs, _ = obstacle_embedding.shape
        n_total = n_robots + n_obs
        device = robot_embedding.device

        # === Extract robot features ===
        robot_feat = robot_embedding[:, :, 4:9]  # (B, N_robots, 5)
        robot_position = robot_embedding[:, :, :2]  # (B, N_robots, 2)
        robot_heading = robot_embedding[:, :, 2:4]  # (B, N_robots, 2) [cos, sin]
        robot_action = robot_embedding[:, :, 7:9]  # (B, N_robots, 2)
        robot_goal = robot_embedding[:, :, -2:]  # (B, N_robots, 2)

        # === Extract obstacle features ===
        obs_position = obstacle_embedding[:, :, :2]  # (B, N_obs, 2)
        obs_heading = obstacle_embedding[:, :, 2:4]  # (B, N_obs, 2)

        # Obstacle node features: zeros (static, no goal, no velocity)
        obs_feat = torch.zeros(batch_size, n_obs, 5, device=device)

        # === Encode node features ===
        robot_embed = self.encode_robot_features(
            robot_feat.reshape(batch_size * n_robots, -1)
        ).view(batch_size, n_robots, self.embedding_dim)

        obs_embed = self.encode_obstacle_features(
            obs_feat.reshape(batch_size * n_obs, -1)
        ).view(batch_size, n_obs, self.embedding_dim)

        # === Compute robot-robot edge features ===
        pos_i = robot_position.unsqueeze(2)  # (B, N_r, 1, 2)
        pos_j = robot_position.unsqueeze(1)  # (B, 1, N_r, 2)
        heading_i = robot_heading.unsqueeze(2)  # (B, N_r, 1, 2)
        heading_j = robot_heading.unsqueeze(1).expand(-1, n_robots, -1, -1)  # (B, N_r, N_r, 2)
        action_j = robot_action.unsqueeze(1).expand(-1, n_robots, -1, -1)  # (B, N_r, N_r, 2)
        goal_j = robot_goal.unsqueeze(1).expand(-1, n_robots, -1, -1)  # (B, N_r, N_r, 2)

        rel_vec_rr = pos_j - pos_i  # (B, N_r, N_r, 2)
        rel_dist_rr = torch.linalg.vector_norm(rel_vec_rr, dim=-1, keepdim=True) / 12

        dx_rr, dy_rr = rel_vec_rr[..., 0], rel_vec_rr[..., 1]
        angle_rr = torch.atan2(dy_rr, dx_rr) - torch.atan2(heading_i[..., 1], heading_i[..., 0])
        angle_rr = (angle_rr + np.pi) % (2 * np.pi) - np.pi

        edge_features_rr = torch.cat([
            rel_dist_rr,
            torch.cos(angle_rr).unsqueeze(-1),
            torch.sin(angle_rr).unsqueeze(-1),
            heading_j[..., 0:1],  # (B, N_r, N_r, 1)
            heading_j[..., 1:2],  # (B, N_r, N_r, 1)
            action_j,
        ], dim=-1)  # (B, N_r, N_r, 7)

        # === Compute robot-obstacle edge features ===
        obs_pos_j = obs_position.unsqueeze(1)  # (B, 1, N_obs, 2)
        obs_heading_j = obs_heading.unsqueeze(1).expand(-1, n_robots, -1, -1)  # (B, N_r, N_obs, 2)

        rel_vec_ro = obs_pos_j - pos_i  # (B, N_r, N_obs, 2)
        rel_dist_ro = torch.linalg.vector_norm(rel_vec_ro, dim=-1, keepdim=True) / 12

        dx_ro, dy_ro = rel_vec_ro[..., 0], rel_vec_ro[..., 1]
        angle_ro = torch.atan2(dy_ro, dx_ro) - torch.atan2(heading_i[..., 1], heading_i[..., 0])
        angle_ro = (angle_ro + np.pi) % (2 * np.pi) - np.pi

        edge_features_ro = torch.cat([
            rel_dist_ro,
            torch.cos(angle_ro).unsqueeze(-1),
            torch.sin(angle_ro).unsqueeze(-1),
            obs_heading_j[..., 0:1],  # (B, N_r, N_obs, 1)
            obs_heading_j[..., 1:2],  # (B, N_r, N_obs, 1)
        ], dim=-1)  # (B, N_r, N_obs, 5)

        # === Hard attention for robot-robot ===
        h_i_rr = robot_embed.unsqueeze(2).expand(-1, -1, n_robots, -1)
        hard_input_rr = torch.cat([h_i_rr, edge_features_rr], dim=-1)
        hard_input_rr = hard_input_rr.reshape(batch_size * n_robots, n_robots, -1)

        h_hard_rr = self.hard_mlp(hard_input_rr)
        hard_logits_rr = self.hard_encoding(h_hard_rr)
        hard_weights_rr = F.gumbel_softmax(hard_logits_rr, hard=False, tau=0.2, dim=-1)[..., 1]
        hard_weights_rr = hard_weights_rr.view(batch_size, n_robots, n_robots)

        # === Hard attention for robot-obstacle ===
        h_i_ro = robot_embed.unsqueeze(2).expand(-1, -1, n_obs, -1)
        hard_input_ro = torch.cat([h_i_ro, edge_features_ro], dim=-1)
        hard_input_ro = hard_input_ro.reshape(batch_size * n_robots, n_obs, -1)

        h_hard_ro = self.hard_mlp_obs(hard_input_ro)
        hard_logits_ro = self.hard_encoding_obs(h_hard_ro)
        hard_weights_ro = F.gumbel_softmax(hard_logits_ro, hard=False, tau=0.2, dim=-1)[..., 1]
        hard_weights_ro = hard_weights_ro.view(batch_size, n_robots, n_obs)

        # === Unnormalized distances for BCE supervision ===
        unnorm_rel_dist_rr = torch.linalg.vector_norm(rel_vec_rr, dim=-1, keepdim=True)
        unnorm_rel_dist_rr = unnorm_rel_dist_rr.reshape(batch_size * n_robots, n_robots, 1)

        unnorm_rel_dist_ro = torch.linalg.vector_norm(rel_vec_ro, dim=-1, keepdim=True)
        unnorm_rel_dist_ro = unnorm_rel_dist_ro.reshape(batch_size * n_robots, n_obs, 1)

        # === Goal-relative polar features for soft attention (robot-robot only) ===
        goal_rel_vec = goal_j - pos_i  # (B, N_r, N_r, 2)
        goal_rel_dist = torch.linalg.vector_norm(goal_rel_vec, dim=-1, keepdim=True)
        goal_angle_global = torch.atan2(goal_rel_vec[..., 1], goal_rel_vec[..., 0])
        heading_angle = torch.atan2(heading_i[..., 1], heading_i[..., 0])
        goal_rel_angle = goal_angle_global - heading_angle
        goal_rel_angle = (goal_rel_angle + np.pi) % (2 * np.pi) - np.pi
        goal_polar_rr = torch.cat([
            goal_rel_dist,
            torch.cos(goal_rel_angle).unsqueeze(-1),
            torch.sin(goal_rel_angle).unsqueeze(-1),
        ], dim=-1)  # (B, N_r, N_r, 3)

        # For robot-obstacle edges, goal polar is zeros (obstacles have no goals)
        goal_polar_ro = torch.zeros(batch_size, n_robots, n_obs, 3, device=device)

        # === Soft edge features (10-dim) ===
        # Robot-robot: 7 + 3 = 10
        soft_edge_rr = torch.cat([edge_features_rr, goal_polar_rr], dim=-1)

        # Robot-obstacle: 5 + 2 (action zeros) + 3 (goal polar zeros) = 10
        obs_action_zeros = torch.zeros(batch_size, n_robots, n_obs, 2, device=device)
        soft_edge_ro = torch.cat([edge_features_ro, obs_action_zeros, goal_polar_ro], dim=-1)

        # === Message passing per batch ===
        attn_outputs = []
        entropy_list = []
        combined_w = []

        for b in range(batch_size):
            edge_index_list = []
            edge_attr_list = []

            # All node embeddings: [robots, obstacles]
            node_feats = torch.cat([robot_embed[b], obs_embed[b]], dim=0)  # (N_total, emb_dim)
            soft_feats_rr = soft_edge_rr[b]  # (N_r, N_r, 10)
            soft_feats_ro = soft_edge_ro[b]  # (N_r, N_obs, 10)
            hard_mask_rr = hard_weights_rr[b]  # (N_r, N_r)
            hard_mask_ro = hard_weights_ro[b]  # (N_r, N_obs)

            # Robot-robot edges (no self-loops)
            for i in range(n_robots):
                for j in range(n_robots):
                    if i != j and hard_mask_rr[i, j] > 0.5:
                        edge_index_list.append([j, i])  # j -> i
                        edge_attr_list.append(soft_feats_rr[i, j])

            # Robot-obstacle edges (obstacle j+n_robots -> robot i)
            for i in range(n_robots):
                for j in range(n_obs):
                    if hard_mask_ro[i, j] > 0.5:
                        edge_index_list.append([n_robots + j, i])  # obs_j -> robot_i
                        edge_attr_list.append(soft_feats_ro[i, j])

            # Build edge tensors
            if edge_index_list:
                edge_index = torch.tensor(
                    edge_index_list, dtype=torch.long, device=device
                ).t()
                edge_attr = torch.stack(edge_attr_list, dim=0)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                edge_attr = torch.zeros((0, 10), dtype=robot_embedding.dtype, device=device)

            # Message passing (only robot nodes are targets)
            attn_out, attn_weights = self.message_graph(
                node_feats, edge_index, edge_attr, n_robots
            )
            attn_outputs.append(attn_out)

            # Compute entropy (only over robot nodes)
            if attn_weights is not None and edge_index.shape[1] > 0:
                batch_entropy = entropy_from_attention(
                    attn_weights, edge_index[1], num_nodes=n_robots
                )
            else:
                batch_entropy = torch.tensor(0.0, device=device)
            entropy_list.append(batch_entropy)

            # Combined weights for visualization
            combined_weights = torch.zeros((n_robots, n_total), device=device)
            if attn_weights is not None:
                for idx in range(edge_index.shape[1]):
                    j = edge_index[0, idx].item()
                    i = edge_index[1, idx].item()
                    combined_weights[i, j] = attn_weights[idx]
            combined_w.append(combined_weights)

        # Stack outputs
        attn_stack = torch.stack(attn_outputs, dim=0).reshape(batch_size * n_robots, -1)
        self_embed = robot_embed.reshape(batch_size * n_robots, -1)

        # Concat original + attended
        concat_embed = torch.cat([self_embed, attn_stack], dim=-1)

        # Decode
        x = F.leaky_relu(self.decode_1(concat_embed))
        att_embedding = F.leaky_relu(self.decode_2(x))

        mean_entropy = torch.stack(entropy_list).mean()
        comb_w = torch.stack(combined_w, dim=0)  # (B, N_r, N_total)

        return (
            att_embedding,
            hard_logits_rr[..., 1],
            hard_logits_ro[..., 1],
            unnorm_rel_dist_rr,
            unnorm_rel_dist_ro,
            mean_entropy,
            hard_weights_rr,
            hard_weights_ro,
            comb_w,
        )
