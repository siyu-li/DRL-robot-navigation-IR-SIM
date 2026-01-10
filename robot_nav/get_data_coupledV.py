from robot_nav.models.MARL.marlTD3.supervised_dataset import (
    create_dataloader,
    SupervisedDatasetGenerator,
)
from robot_nav.models.MARL.marlTD3.coupled_action_policy import CoupledActionPolicy, SharedVelocityHead, CoupledActionActor
from tqdm import tqdm

policy = CoupledActionPolicy()

Generator = SupervisedDatasetGenerator(
    file_paths=["robot_nav/assets/marl_data.yml"],
    num_robots=5,
    v_label_mode="mean")

states, v_labels = Generator.generate_dataset()

train_loader, val_loader = create_dataloader(states, v_labels, batch_size=64, shuffle=True)

pbar = tqdm(train_loader, desc="Training CoupledActionPolicy")
for batch_states, batch_v_labels in pbar:
    print("Batch states shape:", batch_states.shape)  # Should be (batch_size, num_robots, state_dim)
    print("Batch v_labels shape:", batch_v_labels.shape)  # Should be (batch_size, 1)
    loss = policy.train_step_supervised(batch_states, batch_v_labels)
    pbar.set_postfix({"loss": f"{loss:.6f}"})
