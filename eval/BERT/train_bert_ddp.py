import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertForSequenceClassification
from torch.optim import AdamW
import argparse

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT')
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    print(f"Rank {rank}: Starting training")
    setup(rank, world_size)

    # Load BERT model and move it to the correct GPU
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Use PyTorch's AdamW optimizer
    optimizer = AdamW(ddp_model.parameters(), lr=5e-5)

    # Reduced batch size to prevent memory overload
    batch_size = 16  # Reduced from 64 to 16
    sequence_length = 512  # Keeping sequence length the same

    # Dummy dataset (random inputs)
    inputs = torch.randint(0, 100, (batch_size, sequence_length)).to(rank)
    labels = torch.randint(0, 2, (batch_size,)).to(rank)
    attention_mask = torch.ones((batch_size, sequence_length)).to(rank)  # Full attention mask

    # Timing for total training and each epoch
    start_time = time.time()

    for epoch in range(3):  # Training for 3 epochs
        epoch_start = time.time()
        optimizer.zero_grad()

        # Forward pass
        outputs = ddp_model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - epoch_start
        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}, Time per epoch: {epoch_time:.2f} seconds")

    total_training_time = time.time() - start_time
    print(f"Rank {rank}: Total training time: {total_training_time:.2f} seconds")

    cleanup()

def main():
    parser = argparse.ArgumentParser(description="Distributed BERT Training")
    parser.add_argument('--rank', type=int, required=True, help='Rank of the node')
    args = parser.parse_args()

    world_size = int(os.getenv('WORLD_SIZE'))
    rank = args.rank  # Use the rank argument passed from the command line

    # Call the train function for this rank
    train(rank, world_size)

if __name__ == "__main__":
    main()

