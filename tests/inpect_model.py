import torch

# Load the checkpoint
checkpoint = torch.load("robot_nav/models/MARL/marlTD3/checkpoint/correct/TDR-MARL-train_actor.pth", map_location="cpu")

# See all keys (layer names)
print("All keys in checkpoint:")
for key in checkpoint.keys():
    print(f"  {key}: {checkpoint[key].shape}")

# Filter to see only attention-related keys
print("\nAttention-related keys:")
attention_keys = [k for k in checkpoint.keys() if "attention" in k.lower()]
for key in attention_keys:
    print(f"  {key}: {checkpoint[key].shape}")

# Filter to see policy head keys (non-attention)
print("\nPolicy head keys:")
non_attention_keys = [k for k in checkpoint.keys() if "attention" not in k.lower()]
for key in non_attention_keys:
    print(f"  {key}: {checkpoint[key].shape}")