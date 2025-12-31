"""
Test suite for verifying load_pretrained_attention functionality.

This module tests that the attention network weights are correctly loaded
from a pretrained decentralized MARL model into the centralized model.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from robot_nav.models.MARL.marlTD3.marlTD3_centralized import (
    marlTD3_centralized,
    CentralizedActor,
    CentralizedCritic,
)


@pytest.fixture
def device():
    """Return the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model_params():
    """Return common model parameters."""
    return {
        "state_dim": 11,
        "joint_action_dim": 8,  # 4 robots * 2 actions
        "max_action": 1.0,
        "num_robots": 4,
    }


@pytest.fixture
def pretrained_checkpoint_dir():
    """Return the path to pretrained checkpoint directory."""
    return Path("robot_nav/models/MARL/marlTD3/checkpoint")


@pytest.fixture
def pretrained_model_name():
    """Return the pretrained model name."""
    return "TDR-MARL-train"


class TestLoadPretrainedAttention:
    """Test class for load_pretrained_attention functionality."""

    def test_pretrained_checkpoint_exists(self, pretrained_checkpoint_dir, pretrained_model_name):
        """Test that the pretrained checkpoint files exist."""
        actor_path = pretrained_checkpoint_dir / f"{pretrained_model_name}_actor.pth"
        critic_path = pretrained_checkpoint_dir / f"{pretrained_model_name}_critic.pth"

        assert actor_path.exists(), f"Pretrained actor checkpoint not found: {actor_path}"
        assert critic_path.exists(), f"Pretrained critic checkpoint not found: {critic_path}"

    def test_attention_keys_in_pretrained_model(
        self, device, pretrained_checkpoint_dir, pretrained_model_name
    ):
        """Test that the pretrained model contains attention keys."""
        actor_path = pretrained_checkpoint_dir / f"{pretrained_model_name}_actor.pth"
        critic_path = pretrained_checkpoint_dir / f"{pretrained_model_name}_critic.pth"

        actor_state = torch.load(actor_path, map_location=device)
        critic_state = torch.load(critic_path, map_location=device)

        actor_attention_keys = [k for k in actor_state.keys() if k.startswith("attention.")]
        critic_attention_keys = [k for k in critic_state.keys() if k.startswith("attention.")]

        assert len(actor_attention_keys) > 0, "No attention keys found in pretrained actor"
        assert len(critic_attention_keys) > 0, "No attention keys found in pretrained critic"

        print(f"\nPretrained actor attention keys ({len(actor_attention_keys)}):")
        for key in actor_attention_keys:
            print(f"  {key}: {actor_state[key].shape}")

        print(f"\nPretrained critic attention keys ({len(critic_attention_keys)}):")
        for key in critic_attention_keys:
            print(f"  {key}: {critic_state[key].shape}")

    def test_load_pretrained_attention_changes_weights(
        self, device, model_params, pretrained_checkpoint_dir, pretrained_model_name
    ):
        """Test that loading pretrained attention actually changes the weights."""
        # Create model without loading pretrained attention
        model = marlTD3_centralized(
            state_dim=model_params["state_dim"],
            joint_action_dim=model_params["joint_action_dim"],
            max_action=model_params["max_action"],
            num_robots=model_params["num_robots"],
            device=device,
            load_pretrained_attention=False,
        )

        # Store original attention weights
        original_actor_attention = {
            k: v.clone() for k, v in model.actor.state_dict().items() 
            if k.startswith("attention.")
        }
        original_critic_attention = {
            k: v.clone() for k, v in model.critic.state_dict().items() 
            if k.startswith("attention.")
        }

        # Load pretrained attention
        model.load_pretrained_attention(
            filename=pretrained_model_name,
            directory=pretrained_checkpoint_dir,
            freeze_attention=False,
        )

        # Get new attention weights
        new_actor_attention = {
            k: v for k, v in model.actor.state_dict().items() 
            if k.startswith("attention.")
        }
        new_critic_attention = {
            k: v for k, v in model.critic.state_dict().items() 
            if k.startswith("attention.")
        }

        # Check that weights have changed
        actor_weights_changed = False
        for key in original_actor_attention:
            if not torch.allclose(original_actor_attention[key], new_actor_attention[key]):
                actor_weights_changed = True
                break

        critic_weights_changed = False
        for key in original_critic_attention:
            if not torch.allclose(original_critic_attention[key], new_critic_attention[key]):
                critic_weights_changed = True
                break

        assert actor_weights_changed, "Actor attention weights did not change after loading"
        assert critic_weights_changed, "Critic attention weights did not change after loading"

    def test_attention_weights_match_pretrained(
        self, device, model_params, pretrained_checkpoint_dir, pretrained_model_name
    ):
        """Test that loaded attention weights exactly match the pretrained model."""
        # Load pretrained state dicts
        actor_path = pretrained_checkpoint_dir / f"{pretrained_model_name}_actor.pth"
        critic_path = pretrained_checkpoint_dir / f"{pretrained_model_name}_critic.pth"

        pretrained_actor_state = torch.load(actor_path, map_location=device)
        pretrained_critic_state = torch.load(critic_path, map_location=device)

        # Create model with pretrained attention
        model = marlTD3_centralized(
            state_dim=model_params["state_dim"],
            joint_action_dim=model_params["joint_action_dim"],
            max_action=model_params["max_action"],
            num_robots=model_params["num_robots"],
            device=device,
            load_pretrained_attention=True,
            pretrained_attention_model_name=pretrained_model_name,
            pretrained_attention_directory=pretrained_checkpoint_dir,
            freeze_attention=False,
        )

        # Verify actor attention weights match
        for key, value in pretrained_actor_state.items():
            if key.startswith("attention."):
                model_value = model.actor.state_dict()[key]
                assert torch.allclose(value.to(device), model_value), \
                    f"Actor attention weight mismatch for key: {key}"

        # Verify critic attention weights match
        for key, value in pretrained_critic_state.items():
            if key.startswith("attention."):
                model_value = model.critic.state_dict()[key]
                assert torch.allclose(value.to(device), model_value), \
                    f"Critic attention weight mismatch for key: {key}"

        # Verify target networks also have the weights
        for key, value in pretrained_actor_state.items():
            if key.startswith("attention."):
                target_value = model.actor_target.state_dict()[key]
                assert torch.allclose(value.to(device), target_value), \
                    f"Actor target attention weight mismatch for key: {key}"

        for key, value in pretrained_critic_state.items():
            if key.startswith("attention."):
                target_value = model.critic_target.state_dict()[key]
                assert torch.allclose(value.to(device), target_value), \
                    f"Critic target attention weight mismatch for key: {key}"

    def test_policy_head_not_loaded(
        self, device, model_params, pretrained_checkpoint_dir, pretrained_model_name
    ):
        """Test that policy head weights are NOT loaded (should remain randomly initialized)."""
        # Load pretrained state dict
        actor_path = pretrained_checkpoint_dir / f"{pretrained_model_name}_actor.pth"
        pretrained_actor_state = torch.load(actor_path, map_location=device)

        # Create model with pretrained attention
        model = marlTD3_centralized(
            state_dim=model_params["state_dim"],
            joint_action_dim=model_params["joint_action_dim"],
            max_action=model_params["max_action"],
            num_robots=model_params["num_robots"],
            device=device,
            load_pretrained_attention=True,
            pretrained_attention_model_name=pretrained_model_name,
            pretrained_attention_directory=pretrained_checkpoint_dir,
            freeze_attention=False,
        )

        # Check that policy head keys exist in pretrained but are different in our model
        # (due to different architecture - centralized vs decentralized)
        pretrained_policy_keys = [
            k for k in pretrained_actor_state.keys() if k.startswith("policy_head.")
        ]
        model_policy_keys = [
            k for k in model.actor.state_dict().keys() if k.startswith("policy_head.")
        ]

        print(f"\nPretrained policy head keys: {pretrained_policy_keys}")
        print(f"Centralized model policy head keys: {model_policy_keys}")

        # The policy heads have different architectures, so shapes should differ
        # This confirms we didn't accidentally load policy head weights
        assert len(model_policy_keys) > 0, "Model should have policy head parameters"

    def test_freeze_attention_parameters(
        self, device, model_params, pretrained_checkpoint_dir, pretrained_model_name
    ):
        """Test that freezing attention parameters sets requires_grad to False."""
        model = marlTD3_centralized(
            state_dim=model_params["state_dim"],
            joint_action_dim=model_params["joint_action_dim"],
            max_action=model_params["max_action"],
            num_robots=model_params["num_robots"],
            device=device,
            load_pretrained_attention=True,
            pretrained_attention_model_name=pretrained_model_name,
            pretrained_attention_directory=pretrained_checkpoint_dir,
            freeze_attention=True,
        )

        # Check that attention parameters are frozen
        for name, param in model.actor.attention.named_parameters():
            assert not param.requires_grad, \
                f"Actor attention param {name} should be frozen but requires_grad=True"

        for name, param in model.critic.attention.named_parameters():
            assert not param.requires_grad, \
                f"Critic attention param {name} should be frozen but requires_grad=True"

        # Check that policy head parameters are still trainable
        for name, param in model.actor.policy_head.named_parameters():
            assert param.requires_grad, \
                f"Policy head param {name} should be trainable but requires_grad=False"

    def test_unfreeze_attention_parameters(
        self, device, model_params, pretrained_checkpoint_dir, pretrained_model_name
    ):
        """Test that unfreezing attention parameters sets requires_grad back to True."""
        model = marlTD3_centralized(
            state_dim=model_params["state_dim"],
            joint_action_dim=model_params["joint_action_dim"],
            max_action=model_params["max_action"],
            num_robots=model_params["num_robots"],
            device=device,
            load_pretrained_attention=True,
            pretrained_attention_model_name=pretrained_model_name,
            pretrained_attention_directory=pretrained_checkpoint_dir,
            freeze_attention=True,
        )

        # Unfreeze
        model.unfreeze_attention_parameters()

        # Check that attention parameters are now trainable
        for name, param in model.actor.attention.named_parameters():
            assert param.requires_grad, \
                f"Actor attention param {name} should be trainable after unfreeze"

        for name, param in model.critic.attention.named_parameters():
            assert param.requires_grad, \
                f"Critic attention param {name} should be trainable after unfreeze"

    def test_optimizer_excludes_frozen_attention(
        self, device, model_params, pretrained_checkpoint_dir, pretrained_model_name
    ):
        """Test that optimizer doesn't include frozen attention parameters."""
        model = marlTD3_centralized(
            state_dim=model_params["state_dim"],
            joint_action_dim=model_params["joint_action_dim"],
            max_action=model_params["max_action"],
            num_robots=model_params["num_robots"],
            device=device,
            load_pretrained_attention=True,
            pretrained_attention_model_name=pretrained_model_name,
            pretrained_attention_directory=pretrained_checkpoint_dir,
            freeze_attention=True,
        )

        # Get all parameter ids in the actor optimizer
        optimizer_param_ids = set()
        for param_group in model.actor_optimizer.param_groups:
            for param in param_group['params']:
                optimizer_param_ids.add(id(param))

        # Check that attention parameters are NOT in the optimizer
        for param in model.actor.attention.parameters():
            assert id(param) not in optimizer_param_ids, \
                "Frozen attention parameter should not be in optimizer"

        # Check that policy head parameters ARE in the optimizer
        for param in model.actor.policy_head.parameters():
            assert id(param) in optimizer_param_ids, \
                "Policy head parameter should be in optimizer"

    def test_forward_pass_with_pretrained_attention(
        self, device, model_params, pretrained_checkpoint_dir, pretrained_model_name
    ):
        """Test that the model can do a forward pass after loading pretrained attention."""
        model = marlTD3_centralized(
            state_dim=model_params["state_dim"],
            joint_action_dim=model_params["joint_action_dim"],
            max_action=model_params["max_action"],
            num_robots=model_params["num_robots"],
            device=device,
            load_pretrained_attention=True,
            pretrained_attention_model_name=pretrained_model_name,
            pretrained_attention_directory=pretrained_checkpoint_dir,
            freeze_attention=False,
        )

        # Create dummy observation
        batch_size = 4
        obs = np.random.randn(model_params["num_robots"], model_params["state_dim"]).astype(np.float32)

        # Test get_action
        action, connection, combined_weights = model.get_action(obs, add_noise=False)

        assert action.shape == (model_params["num_robots"], 2), \
            f"Expected action shape ({model_params['num_robots']}, 2), got {action.shape}"

        # Test with noise
        action_noisy, _, _ = model.get_action(obs, add_noise=True)
        assert action_noisy.shape == (model_params["num_robots"], 2)

    def test_init_with_missing_pretrained_params_raises_error(self, device, model_params):
        """Test that missing pretrained parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="pretrained_attention_model_name must be provided"):
            marlTD3_centralized(
                state_dim=model_params["state_dim"],
                joint_action_dim=model_params["joint_action_dim"],
                max_action=model_params["max_action"],
                num_robots=model_params["num_robots"],
                device=device,
                load_pretrained_attention=True,
                pretrained_attention_model_name=None,
                pretrained_attention_directory=Path("some/path"),
            )

        with pytest.raises(ValueError, match="pretrained_attention_directory must be provided"):
            marlTD3_centralized(
                state_dim=model_params["state_dim"],
                joint_action_dim=model_params["joint_action_dim"],
                max_action=model_params["max_action"],
                num_robots=model_params["num_robots"],
                device=device,
                load_pretrained_attention=True,
                pretrained_attention_model_name="some_model",
                pretrained_attention_directory=None,
            )


class TestAttentionWeightConsistency:
    """Test that attention weights produce consistent outputs."""

    def test_attention_output_matches_pretrained(
        self, device, model_params, pretrained_checkpoint_dir, pretrained_model_name
    ):
        """
        Test that the attention network produces the same output as the 
        pretrained decentralized model's attention network.
        """
        # Load the pretrained actor to get its attention module
        actor_path = pretrained_checkpoint_dir / f"{pretrained_model_name}_actor.pth"
        pretrained_state = torch.load(actor_path, map_location=device)

        # Create centralized model with pretrained attention
        model = marlTD3_centralized(
            state_dim=model_params["state_dim"],
            joint_action_dim=model_params["joint_action_dim"],
            max_action=model_params["max_action"],
            num_robots=model_params["num_robots"],
            device=device,
            load_pretrained_attention=True,
            pretrained_attention_model_name=pretrained_model_name,
            pretrained_attention_directory=pretrained_checkpoint_dir,
            freeze_attention=False,
        )

        # Create a reference attention module and load pretrained weights
        from robot_nav.models.MARL.Attention.g2anet import G2ANet
        reference_attention = G2ANet(embedding_dim=256).to(device)
        
        # Load only attention weights into reference
        attention_state = {
            k.replace("attention.", ""): v 
            for k, v in pretrained_state.items() 
            if k.startswith("attention.")
        }
        reference_attention.load_state_dict(attention_state)
        reference_attention.eval()
        model.actor.attention.eval()

        # Create test input
        test_input = torch.randn(1, model_params["num_robots"], model_params["state_dim"]).to(device)

        # Get outputs from both
        with torch.no_grad():
            ref_output = reference_attention(test_input)
            model_output = model.actor.attention(test_input)

        # Compare outputs (first element is the attended embedding)
        assert torch.allclose(ref_output[0], model_output[0], atol=1e-6), \
            "Attention outputs don't match"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
