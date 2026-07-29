"""Shape/determinism tests for the feature builder and the learned prior."""

from __future__ import annotations

import numpy as np
import torch

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE, PRECISE
from robot_nav.models.MARL.capswitcher_14.configs import MOVE_GROUPS, N_ROBOTS
from robot_nav.models.MARL.capswitcher_14.rl.forward_model import ModelState
from robot_nav.models.MARL.capswitcher_14.rl.search.common import Branch
from robot_nav.models.MARL.capswitcher_14.rl.search.features import (
    GLOBAL_FEAT_DIM,
    GROUP_FEAT_DIM,
    GroupFeatureBuilder,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.prior_net import (
    LearnedPrior,
    PriorNet,
)


class _FakeGeom:
    def __init__(self):
        self.rho = 0.2
        self.obstacle_xy = np.array([[3.0, 3.0], [7.0, 1.0]])
        self.obstacle_r = np.array([0.5, 0.4])


class _FakeModel:
    def __init__(self):
        self.geom = _FakeGeom()
        rng = np.random.default_rng(0)
        self.goals = rng.uniform(0, 10, size=(N_ROBOTS, 2))
        self.goal_threshold = 0.3


def _state(seed=1) -> ModelState:
    rng = np.random.default_rng(seed)
    poses = np.concatenate(
        [rng.uniform(0, 10, size=(N_ROBOTS, 2)),
         rng.uniform(-np.pi, np.pi, size=(N_ROBOTS, 1))],
        axis=1,
    )
    return ModelState(poses=poses, last_actions=np.zeros((N_ROBOTS, 2)))


def _stub_branches():
    branches = [
        Branch(mode=COARSE, group=g, step_cost=1.0) for g in range(len(MOVE_GROUPS))
    ]
    branches.append(Branch(mode=PRECISE, group=None, step_cost=2.0))
    return branches


def test_feature_shapes_ranges_and_determinism():
    fb = GroupFeatureBuilder(MOVE_GROUPS)
    model, ms = _FakeModel(), _state()
    gf, glf = fb(model, ms)
    gf2, glf2 = fb(model, ms)
    assert gf.shape == (22, GROUP_FEAT_DIM) and glf.shape == (GLOBAL_FEAT_DIM,)
    assert np.array_equal(gf, gf2) and np.array_equal(glf, glf2)
    # Size one-hot is exactly one per row and matches the group's size class.
    onehot = gf[:, 0:3]
    assert np.all(onehot.sum(axis=1) == 1.0)
    sizes = np.array([len(g) for g in MOVE_GROUPS])
    assert np.array_equal(onehot[:, 0] == 1.0, sizes == 3)
    assert np.array_equal(onehot[:, 2] == 1.0, sizes == 7)
    # Angle features normalised to [0, 1]; clearances capped to <= 1.
    assert np.all(gf[:, 6:8] >= 0.0) and np.all(gf[:, 6:8] <= 1.0)
    assert np.all(gf[:, 8:10] <= 1.0)
    assert np.all(np.isfinite(gf)) and np.all(np.isfinite(glf))


def test_prior_net_forward_shapes():
    net = PriorNet()
    B, K = 5, 22
    logits, margin = net(
        torch.zeros(B, K, GROUP_FEAT_DIM), torch.zeros(B, GLOBAL_FEAT_DIM)
    )
    assert logits.shape == (B, K + 1)
    assert margin.shape == (B, K)


def test_prior_net_save_load_roundtrip(tmp_path):
    net = PriorNet()
    path = tmp_path / "prior.pt"
    net.save(path)
    net2 = PriorNet.load(path)
    gf = torch.randn(1, 22, GROUP_FEAT_DIM)
    glf = torch.randn(1, GLOBAL_FEAT_DIM)
    with torch.no_grad():
        a, am = net(gf, glf)
        b, bm = net2(gf, glf)
    assert torch.allclose(a, b) and torch.allclose(am, bm)


def test_learned_prior_contract_and_single_forward_cache():
    calls = {"n": 0}

    class CountingNet(PriorNet):
        def forward(self, gf, glf):
            calls["n"] += 1
            return super().forward(gf, glf)

    prior = LearnedPrior(CountingNet(), GroupFeatureBuilder(MOVE_GROUPS),
                         feas_margin=0.0)
    model, ms = _FakeModel(), _state()
    branches = _stub_branches()

    logits = prior(model, ms, branches)
    feas = prior.feasibility(model, ms, branches)
    assert logits.shape == (23,)
    assert feas.shape == (23,) and feas.dtype == bool
    assert feas[22]                       # precise edge always feasible
    assert calls["n"] == 1                # both lookups share one forward

    prior(model, _state(seed=2), branches)
    assert calls["n"] == 2                # new state → new forward
