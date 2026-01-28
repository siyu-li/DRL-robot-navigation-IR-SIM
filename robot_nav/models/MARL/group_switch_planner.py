"""
Rule-Based Group Switching Controller for Multi-Robot Navigation.

This module implements a switching planner that:
- Computes urgency scores for each robot based on collision risk, stuckness, and goal priority
- Selects an active robot group at configurable intervals
- Only robots in the active group execute actions; others receive zero commands

Urgency Score:
    U_i = w_c * C_i + w_s * S_i + w_g * G_i

Where:
    - C_i: Collision risk based on minimum lidar sector distance
    - S_i: Stuckness based on lack of progress toward goal
    - G_i: Goal priority based on distance to goal

Group Selection:
    1. Find robot with maximum urgency
    2. Determine group size based on collision risk
    3. Select best group containing max-urgency robot using:
       Score(g) = max_{i in g} U_i + lambda_mean * mean_{i in g} U_i - mu_crowd * crowding(g)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


@dataclass
class GroupSwitchConfig:
    """Configuration for group switching planner.
    
    Attributes:
        switch_interval: Steps between group switches.
        random_switching: If True, randomly select groups instead of using urgency scores.
        random_group_size: Group size for random switching (1, 2, or 3). If None, randomly choose size.
        w_collision: Weight for collision risk in urgency score.
        w_stuckness: Weight for stuckness in urgency score.
        w_goal: Weight for goal priority in urgency score.
        d_hard: Hard collision distance (meters).
        d_safe: Safe distance threshold (meters).
        progress_window_k: Window size for progress tracking.
        delta_min: Minimum progress threshold (meters).
        delta_max: Maximum progress threshold (meters).
        d_goal_max: Maximum goal distance for normalization.
        lambda_mean: Weight for mean urgency in group score.
        mu_crowd: Crowding penalty coefficient.
        crowding_eps: Epsilon for crowding calculation.
        collision_threshold_single: Above this collision risk -> single robot group.
        collision_threshold_pair: Above this collision risk -> pair group.
        debug: Enable debug output.
        num_robots: Number of robots in the system.
    """
    
    # Switching interval (steps between group selection)
    switch_interval: int = 10
    
    # Random switching mode
    random_switching: bool = False  # If True, randomly select groups
    random_group_size: Optional[int] = None  # Group size for random mode (1, 2, or 3). None = random size
    
    # Urgency weights
    w_collision: float = 0.4  # Weight for collision risk
    w_stuckness: float = 0.3  # Weight for stuckness
    w_goal: float = 0.3       # Weight for goal priority
    
    # Collision risk parameters
    d_hard: float = 0.3       # Hard collision distance (meters)
    d_safe: float = 1.5       # Safe distance threshold (meters)
    
    # Stuckness parameters
    progress_window_k: int = 20       # Window size for progress tracking
    delta_min: float = 0.01           # Minimum progress threshold (meters)
    delta_max: float = 0.5            # Maximum progress threshold (meters)
    
    # Goal priority parameters
    d_goal_max: float = 15.0   # Maximum goal distance for normalization
    
    # Group selection parameters
    lambda_mean: float = 0.3   # Weight for mean urgency in group score
    mu_crowd: float = 0.2      # Crowding penalty coefficient
    crowding_eps: float = 0.1  # Epsilon for crowding calculation
    
    # Group size selection thresholds (based on max collision risk in active group)
    collision_threshold_single: float = 0.7   # Above this -> single robot group
    collision_threshold_pair: float = 0.4     # Above this -> pair group
    # Below collision_threshold_pair -> triple group
    
    # Debug options
    debug: bool = False
    
    # Number of robots (set during initialization)
    num_robots: int = 5


def generate_default_groups(num_robots: int) -> Dict[int, List[List[int]]]:
    """
    Generate default candidate groups of sizes 1, 2, and 3.
    
    Args:
        num_robots: Number of robots in the system.
        
    Returns:
        Dictionary mapping group size to list of groups of that size.
        
    Example:
        >>> generate_default_groups(3)
        {1: [[0], [1], [2]], 
         2: [[0, 1], [0, 2], [1, 2]], 
         3: [[0, 1, 2]]}
    """
    groups = {1: [], 2: [], 3: []}
    
    # Size 1: each robot individually
    for i in range(num_robots):
        groups[1].append([i])
    
    # Size 2: all pairs
    for i in range(num_robots):
        for j in range(i + 1, num_robots):
            groups[2].append([i, j])
    
    # Size 3: all triples
    for i in range(num_robots):
        for j in range(i + 1, num_robots):
            for k in range(j + 1, num_robots):
                groups[3].append([i, j, k])
    
    return groups


class UrgencyCalculator:
    """Calculates per-robot urgency scores based on collision risk, stuckness, and goal priority."""
    
    def __init__(self, config: GroupSwitchConfig):
        """
        Initialize the urgency calculator.
        
        Args:
            config: Configuration parameters.
        """
        self.config = config
        self.num_robots = config.num_robots
        
        # Progress history buffers for stuckness calculation
        # Each robot has a deque storing recent goal distances
        self.goal_distance_history: List[deque] = [
            deque(maxlen=config.progress_window_k) 
            for _ in range(self.num_robots)
        ]
    
    def reset(self):
        """Reset internal buffers for new episode."""
        for i in range(self.num_robots):
            self.goal_distance_history[i].clear()
    
    def update_history(self, goal_distances: List[float]):
        """
        Update goal distance history for all robots.
        
        Args:
            goal_distances: Current distance to goal for each robot.
        """
        for i, dist in enumerate(goal_distances):
            self.goal_distance_history[i].append(dist)
    
    def compute_collision_risk(self, lidar_sectors: np.ndarray) -> np.ndarray:
        """
        Compute collision risk C_i for each robot.
        
        C_i = clip((d_safe - d_min_i) / (d_safe - d_hard), 0, 1)
        
        Where d_min_i is the minimum distance across all lidar sectors for robot i.
        Higher risk when obstacles are closer.
        
        Args:
            lidar_sectors: Array of shape (num_robots, num_sectors) containing
                          min distances per sector.
                          
        Returns:
            Array of collision risks in [0, 1] for each robot.
        """
        # Get minimum distance across all sectors for each robot
        d_min = np.min(lidar_sectors, axis=1)  # (num_robots,)
        
        # Compute collision risk
        # When d_min <= d_hard: risk = 1.0
        # When d_min >= d_safe: risk = 0.0
        # Linear interpolation in between
        denominator = self.config.d_safe - self.config.d_hard
        if denominator <= 0:
            denominator = 1e-8
        
        risk = (self.config.d_safe - d_min) / denominator
        risk = np.clip(risk, 0.0, 1.0)
        
        return risk
    
    def compute_stuckness(self) -> np.ndarray:
        """
        Compute stuckness S_i for each robot based on progress history.
        
        S_i = clip(1 - (delta_i - Δmin) / (Δmax - Δmin), 0, 1)
        
        Where delta_i = d_goal(t-K) - d_goal(t) is the progress over K steps.
        Positive delta means robot got closer to goal.
        
        Returns:
            Array of stuckness scores in [0, 1] for each robot.
        """
        stuckness = np.zeros(self.num_robots)
        
        for i in range(self.num_robots):
            history = self.goal_distance_history[i]
            
            if len(history) < 2:
                # Not enough history, assume not stuck
                stuckness[i] = 0.0
                continue
            
            # Progress = distance at start of window - distance now
            # Positive means robot got closer to goal
            delta = history[0] - history[-1]
            
            # Map to stuckness: more progress = less stuck
            denominator = self.config.delta_max - self.config.delta_min
            if denominator <= 0:
                denominator = 1e-8
                
            s = 1.0 - (delta - self.config.delta_min) / denominator
            stuckness[i] = np.clip(s, 0.0, 1.0)
        
        return stuckness
    
    def compute_goal_priority(self, goal_distances: List[float]) -> np.ndarray:
        """
        Compute goal priority G_i for each robot.
        
        G_i = clip(d_goal_i / d_goal_max, 0, 1)
        
        Higher distance = higher priority (robot is far from goal and needs attention).
        
        Args:
            goal_distances: Current distance to goal for each robot.
            
        Returns:
            Array of goal priorities in [0, 1] for each robot.
        """
        priority = np.array(goal_distances) / self.config.d_goal_max
        priority = np.clip(priority, 0.0, 1.0)
        
        return priority
    
    def compute_urgency_scores(
        self,
        lidar_sectors: np.ndarray,
        goal_distances: List[float],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute overall urgency scores for all robots.
        
        U_i = w_c * C_i + w_s * S_i + w_g * G_i
        
        Args:
            lidar_sectors: Array of shape (num_robots, num_sectors).
            goal_distances: Distance to goal for each robot.
            
        Returns:
            Tuple of (urgency_scores, component_dict) where:
                - urgency_scores: Array of shape (num_robots,)
                - component_dict: Dictionary with 'collision', 'stuckness', 'goal' arrays
        """
        collision_risk = self.compute_collision_risk(lidar_sectors)
        stuckness = self.compute_stuckness()
        goal_priority = self.compute_goal_priority(goal_distances)
        
        # Weighted combination
        urgency = (
            self.config.w_collision * collision_risk +
            self.config.w_stuckness * stuckness +
            self.config.w_goal * goal_priority
        )
        
        components = {
            'collision': collision_risk,
            'stuckness': stuckness,
            'goal': goal_priority,
        }
        
        return urgency, components


class GroupSwitchPlanner:
    """
    Rule-based planner for selecting active robot groups.
    
    At each switch interval, computes urgency scores and selects the best
    group based on max urgency, mean urgency, and crowding penalty.
    
    Usage:
        config = GroupSwitchConfig(num_robots=5, switch_interval=10)
        planner = GroupSwitchPlanner(config)
        
        # In episode loop:
        planner.reset()
        for step in range(max_steps):
            planner.update(goal_distances)
            active_group = planner.select_group(
                step, lidar_sectors, goal_distances, positions
            )
            # Use active_group to mask actions...
    """
    
    def __init__(
        self,
        config: GroupSwitchConfig,
        candidate_groups: Optional[Dict[int, List[List[int]]]] = None,
    ):
        """
        Initialize the group switch planner.
        
        Args:
            config: Configuration parameters.
            candidate_groups: Dictionary mapping group size to list of groups.
                            If None, uses default groups generated from num_robots.
        """
        self.config = config
        self.num_robots = config.num_robots
        
        # Generate candidate groups if not provided
        if candidate_groups is None:
            self.candidate_groups = generate_default_groups(self.num_robots)
        else:
            self.candidate_groups = candidate_groups
        
        # Flatten all groups for reference
        self.all_groups: List[List[int]] = []
        for size in [1, 2, 3]:
            if size in self.candidate_groups:
                self.all_groups.extend(self.candidate_groups[size])
        
        # Initialize urgency calculator
        self.urgency_calculator = UrgencyCalculator(config)
        
        # Current active group (default: all robots)
        self.current_active_group: List[int] = list(range(self.num_robots))
        self.last_switch_step: int = -config.switch_interval  # Force initial selection
        
        # Cache for debug info
        self._last_urgency_scores: Optional[np.ndarray] = None
        self._last_components: Optional[Dict[str, np.ndarray]] = None
        self._last_group_scores: Optional[List[Tuple[List[int], float]]] = None
    
    def reset(self):
        """Reset planner state for new episode."""
        self.urgency_calculator.reset()
        self.current_active_group = list(range(self.num_robots))
        self.last_switch_step = -self.config.switch_interval
        self._last_urgency_scores = None
        self._last_components = None
        self._last_group_scores = None
    
    def _compute_crowding_penalty(
        self,
        group: List[int],
        positions: Optional[np.ndarray],
    ) -> float:
        """
        Compute crowding penalty for a group based on inter-robot distances.
        
        crowding(g) = 1 / (min_pairwise_distance_in_group + eps)
        
        Args:
            group: List of robot indices in the group.
            positions: Array of shape (num_robots, 2) with [x, y] positions.
                      If None, returns 0.
                      
        Returns:
            Crowding penalty value.
        """
        if positions is None or len(group) < 2:
            return 0.0
        
        # Compute pairwise distances within group
        min_dist = float('inf')
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ri, rj = group[i], group[j]
                dist = np.linalg.norm(positions[ri] - positions[rj])
                min_dist = min(min_dist, dist)
        
        if min_dist == float('inf'):
            return 0.0
        
        return 1.0 / (min_dist + self.config.crowding_eps)
    
    def _compute_group_score(
        self,
        group: List[int],
        urgency_scores: np.ndarray,
        positions: Optional[np.ndarray],
    ) -> float:
        """
        Compute selection score for a candidate group.
        
        Score(g) = max_{i in g} U_i + lambda_mean * mean_{i in g} U_i - mu_crowd * crowding(g)
        
        Args:
            group: List of robot indices.
            urgency_scores: Urgency scores for all robots.
            positions: Robot positions for crowding calculation.
            
        Returns:
            Group score.
        """
        group_urgencies = urgency_scores[group]
        
        max_urgency = np.max(group_urgencies)
        mean_urgency = np.mean(group_urgencies)
        crowding = self._compute_crowding_penalty(group, positions)
        
        score = (
            max_urgency +
            self.config.lambda_mean * mean_urgency -
            self.config.mu_crowd * crowding
        )
        
        return score
    
    def _determine_group_size(self, max_collision_risk: float) -> int:
        """
        Determine appropriate group size based on collision risk.
        
        Higher collision risk -> smaller group (more cautious).
        
        Args:
            max_collision_risk: Maximum collision risk among considered robots.
            
        Returns:
            Target group size (1, 2, or 3).
        """
        if max_collision_risk >= self.config.collision_threshold_single:
            return 1
        elif max_collision_risk >= self.config.collision_threshold_pair:
            return 2
        else:
            return 3
    
    def _select_best_group(
        self,
        urgency_scores: np.ndarray,
        collision_risks: np.ndarray,
        positions: Optional[np.ndarray],
    ) -> List[int]:
        """
        Select the best group based on urgency scores and collision risks.
        
        Algorithm:
        1. Find robot with maximum urgency score
        2. Determine target group size based on that robot's collision risk
        3. Get candidate groups of target size containing max-urgency robot
        4. Score each candidate and select the best
        
        Args:
            urgency_scores: Urgency scores for all robots.
            collision_risks: Collision risk for each robot.
            positions: Robot positions for crowding calculation.
            
        Returns:
            List of robot indices for the selected group.
        """
        # Find robot with maximum urgency
        max_urgency_robot = int(np.argmax(urgency_scores))
        max_collision_risk = collision_risks[max_urgency_robot]
        
        # Determine target group size based on collision risk
        target_size = self._determine_group_size(max_collision_risk)
        
        # Get candidate groups of the target size that contain the max urgency robot
        candidates = [
            g for g in self.candidate_groups.get(target_size, [])
            if max_urgency_robot in g
        ]
        
        # If no candidates found, fall back to single robot
        if not candidates:
            candidates = [[max_urgency_robot]]
        
        # Score all candidates and select best
        group_scores = []
        for group in candidates:
            score = self._compute_group_score(group, urgency_scores, positions)
            group_scores.append((group, score))
        
        # Store for debugging
        self._last_group_scores = sorted(group_scores, key=lambda x: x[1], reverse=True)
        
        # Select group with highest score
        best_group = max(group_scores, key=lambda x: x[1])[0]
        
        return best_group
    
    def _select_random_group(self) -> List[int]:
        """
        Randomly select a group from candidate groups.
        
        If random_group_size is specified in config, only selects from groups of that size.
        Otherwise, randomly selects from all available groups.
        
        Returns:
            List of robot indices for the randomly selected group.
        """
        target_size = self.config.random_group_size
        
        if target_size is not None:
            # Select from groups of specified size
            if target_size in self.candidate_groups and len(self.candidate_groups[target_size]) > 0:
                candidates = self.candidate_groups[target_size]
            else:
                # Fall back to all groups if specified size not available
                candidates = self.all_groups
        else:
            # Randomly select size first, then select group of that size
            available_sizes = [s for s in [1, 2, 3] if s in self.candidate_groups and len(self.candidate_groups[s]) > 0]
            if available_sizes:
                target_size = np.random.choice(available_sizes)
                candidates = self.candidate_groups[target_size]
            else:
                candidates = self.all_groups
        
        # Randomly select a group from candidates
        if len(candidates) > 0:
            idx = np.random.randint(len(candidates))
            return list(candidates[idx])
        else:
            # Fallback: return all robots
            return list(range(self.num_robots))
    
    def update(self, goal_distances: List[float]):
        """
        Update internal history buffers.
        
        Call this every step with current goal distances.
        
        Args:
            goal_distances: Distance to goal for each robot.
        """
        self.urgency_calculator.update_history(goal_distances)
    
    def select_group(
        self,
        step: int,
        lidar_sectors: np.ndarray,
        goal_distances: List[float],
        positions: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        Select active group for current step.
        
        Only performs new selection at switch intervals. Between switches,
        returns the previously selected group.
        
        If random_switching is enabled in config, randomly selects groups
        instead of using urgency-based selection.
        
        Args:
            step: Current step number.
            lidar_sectors: Array of shape (num_robots, num_sectors) with min distances.
                          (Not used in random mode, but kept for API consistency)
            goal_distances: Distance to goal for each robot.
                          (Not used in random mode, but kept for API consistency)
            positions: Optional array of shape (num_robots, 2) with [x, y] positions.
                      (Not used in random mode, but kept for API consistency)
            
        Returns:
            List of robot indices for the active group.
        """
        # Check if it's time to switch
        if step - self.last_switch_step < self.config.switch_interval:
            return self.current_active_group
        
        # Use random or urgency-based selection
        if self.config.random_switching:
            # Random group selection
            best_group = self._select_random_group()
            self._last_urgency_scores = None
            self._last_components = None
            self._last_group_scores = None
        else:
            # Urgency-based selection
            # Compute urgency scores
            urgency_scores, components = self.urgency_calculator.compute_urgency_scores(
                lidar_sectors, goal_distances
            )
            
            # Store for debugging
            self._last_urgency_scores = urgency_scores
            self._last_components = components
            
            # Select best group
            best_group = self._select_best_group(
                urgency_scores,
                components['collision'],
                positions,
            )
        
        # Update state
        self.current_active_group = best_group
        self.last_switch_step = step
        
        # Debug output
        if self.config.debug:
            self._print_debug_info(step)
        
        return self.current_active_group
    
    def get_active_group(self) -> List[int]:
        """Get the current active group without updating."""
        return self.current_active_group
    
    def get_last_urgency_scores(self) -> Optional[np.ndarray]:
        """Get the most recent urgency scores (for logging/debugging)."""
        return self._last_urgency_scores
    
    def get_last_components(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the most recent urgency components (for logging/debugging)."""
        return self._last_components
    
    def _print_debug_info(self, step: int):
        """Print debug information about group selection."""
        print(f"\n{'='*50}")
        print(f"Group Switch Debug (Step {step})")
        print(f"{'='*50}")
        print(f"Mode: {'RANDOM' if self.config.random_switching else 'URGENCY-BASED'}")
        print(f"Active Group: {self.current_active_group}")
        
        if self.config.random_switching:
            size_info = f"size={self.config.random_group_size}" if self.config.random_group_size else "random size"
            print(f"Random selection ({size_info})")
        elif self._last_urgency_scores is not None:
            print("\nUrgency Scores (U_i = w_c*C + w_s*S + w_g*G):")
            for i, u in enumerate(self._last_urgency_scores):
                c = self._last_components['collision'][i]
                s = self._last_components['stuckness'][i]
                g = self._last_components['goal'][i]
                marker = " <-- ACTIVE" if i in self.current_active_group else ""
                print(f"  Robot {i}: U={u:.3f} (C={c:.3f}, S={s:.3f}, G={g:.3f}){marker}")
        
        if self._last_group_scores is not None:
            print("\nTop-3 Group Scores:")
            for idx, (group, score) in enumerate(self._last_group_scores[:3]):
                marker = " <-- SELECTED" if group == self.current_active_group else ""
                print(f"  {idx+1}. Group {group}: Score={score:.3f}{marker}")
        
        print(f"{'='*50}\n")


def extract_lidar_sectors_from_state(
    lidar_scans: List[np.ndarray],
    num_sectors: int = 12,
    lidar_range_max: float = 7.0,
) -> np.ndarray:
    """
    Convert raw lidar scans to sector-based minimum distances.
    
    Divides each lidar scan into sectors and computes the minimum
    distance reading within each sector.
    
    Args:
        lidar_scans: List of lidar scans, each of shape (num_beams,).
        num_sectors: Number of sectors to aggregate into.
        lidar_range_max: Maximum lidar range for denormalization.
        
    Returns:
        Array of shape (num_robots, num_sectors) with min distances per sector.
    """
    num_robots = len(lidar_scans)
    sectors = np.zeros((num_robots, num_sectors))
    
    for i, scan in enumerate(lidar_scans):
        scan = np.asarray(scan).flatten()
        num_beams = len(scan)
        beams_per_sector = max(1, num_beams // num_sectors)
        
        for s in range(num_sectors):
            start_idx = s * beams_per_sector
            end_idx = (s + 1) * beams_per_sector if s < num_sectors - 1 else num_beams
            end_idx = min(end_idx, num_beams)
            
            if start_idx >= num_beams:
                sectors[i, s] = lidar_range_max
                continue
                
            sector_readings = scan[start_idx:end_idx]
            
            # Denormalize if needed (assuming scan might be normalized to [0, 1])
            if len(sector_readings) > 0:
                max_val = np.max(sector_readings)
                if max_val <= 1.0 and max_val > 0:
                    sector_readings = sector_readings * lidar_range_max
                
                sectors[i, s] = np.min(sector_readings)
            else:
                sectors[i, s] = lidar_range_max
    
    return sectors


def get_positions_from_poses(poses: List[List[float]]) -> np.ndarray:
    """
    Extract [x, y] positions from pose list.
    
    Args:
        poses: List of [x, y, theta] poses for each robot.
        
    Returns:
        Array of shape (num_robots, 2) with [x, y] positions.
    """
    return np.array([[p[0], p[1]] for p in poses])
