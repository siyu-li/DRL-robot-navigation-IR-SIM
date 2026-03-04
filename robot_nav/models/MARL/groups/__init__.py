"""
Group utilities: group generation, action coupling, and learned mixing.
"""

from robot_nav.models.MARL.groups.group_generator import (
    generate_all_groups,
    generate_original_groups,
    generate_subgroups_recursive,
    filter_groups_by_size,
    print_group_statistics,
)
from robot_nav.models.MARL.groups.combinatorial_group_generator import (
    generate_all_combinations,
    robot_frequency,
    print_combination_statistics,
)
from robot_nav.models.MARL.groups.action_coupling import (
    actions_for_group,
    actions_for_group_from_raw,
)
from robot_nav.models.MARL.groups.mixing_network import MixingNetwork
from robot_nav.models.MARL.groups.learned_action_coupling import (
    compute_mixed_actions,
    compute_mixed_actions_tensor,
    get_embeddings_from_frozen_actor,
)

__all__ = [
    "generate_all_groups",
    "generate_original_groups",
    "generate_subgroups_recursive",
    "filter_groups_by_size",
    "print_group_statistics",
    "generate_all_combinations",
    "robot_frequency",
    "print_combination_statistics",
    "actions_for_group",
    "actions_for_group_from_raw",
    "MixingNetwork",
    "compute_mixed_actions",
    "compute_mixed_actions_tensor",
    "get_embeddings_from_frozen_actor",
]
