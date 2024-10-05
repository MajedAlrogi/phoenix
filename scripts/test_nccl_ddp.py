import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

# Function to set up the distributed environment for each rank
def setup(rank, world_size):
    print(f"Rank {rank}: Setting up process group")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    print(f"Rank {rank}: Process group set up")

# Function to clean up the process group after training
def cleanup():
    dist.destroy_process_group()

# Simple model for distributed training
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.fc(x)

# Training function for each process
def train(rank, world_size):
    print(f"Rank {rank}: Starting training")
    
    # Set up the process group for communication
    setup(rank, world_size)
    
    # Initialize the model and move it to the appropriate device
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Create a loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Dummy training loop
    for epoch in range(10):
        print(f"Rank {rank}: Epoch {epoch} started")
        optimizer.zero_grad()
        
        # Generate random inputs and labels
        inputs = torch.randn(20, 10).to(rank)
        labels = torch.randn(20, 10).to(rank)
        
        # Forward pass and backward pass
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Rank {rank}: Epoch {epoch} completed")

    # Clean up the process group
    cleanup()
    print(f"Rank {rank}: Training completed")

# Main function to spawn the training process for each rank
def main():
    world_size = torch.cuda.device_count()  # Automatically determine the number of GPUs
    print(f"Main: Detected {world_size} GPUs")

    # Spawn one process per GPU
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    print("Main: Starting Distributed Data Parallel (DDP) test")
    main()
    print("Main: DDP test completed")

