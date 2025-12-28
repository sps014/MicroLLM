import os
import torch
import tiktoken
from torch.nn import functional as F
from datasets import load_dataset
from core.micro_llm import ModelConfig, MicroLLM

# --- TRAINING ENGINE ---

def train():
    cfg = ModelConfig()
    model = MicroLLM(cfg).to(cfg.device)
    
    # AdamW is the industry standard for Transformers; it handles weight decay correctly.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # GradScaler enables Mixed Precision, allowing faster math without loss of precision.
    scaler = torch.amp.GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu') 
    
    # Load-and-Resume logic allows training to be interrupted and restarted.
    weights_path = "model_weights.pth"
    if os.path.exists(weights_path):
        print(f"Loading checkpoint: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=cfg.device, weights_only=True))

    # We use 'streaming' to avoid downloading the entire dataset to disk/RAM at once.
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    dataset_iter = iter(dataset)
    enc = tiktoken.get_encoding("gpt2")

    print(f"Initializing training on {cfg.device}...")
    model.train()
    
    for step in range(1, 5001): 
        xs, ys = [], []
        # We loop until the batch is full, handling potential stream timeouts.
        while len(xs) < 8:
            try:
                story = next(dataset_iter)['text']
                ids = enc.encode(story)
                if len(ids) > cfg.block_size:
                    # Input (x) is the sequence; Target (y) is the sequence shifted by 1.
                    xs.append(torch.tensor(ids[:cfg.block_size]))
                    ys.append(torch.tensor(ids[1:cfg.block_size+1]))
            except (StopIteration, Exception):
                dataset_iter = iter(dataset) # Restart stream if it ends or hangs

        x = torch.stack(xs).to(cfg.device)
        y = torch.stack(ys).to(cfg.device)

        # Autocast performs operations in Float16 for speed while keeping master weights in Float32.
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(device_type):
            logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        # Scaler prevents small gradients from flushing to zero in Float16.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            print(f"Step {step} | Training Loss: {loss.item():.4f}")
            # Periodic checkpointing to safeguard progress.
            torch.save(model.state_dict(), weights_path)
    
    return model, enc

# --- INFERENCE ---

def generate(model, enc, prompt: str, max_len: int = 150):
    """
    Predicts tokens sequentially. Each new word becomes context for the next word.
    """
    model.eval()
    idx = torch.tensor(enc.encode(prompt), device=model.config.device).unsqueeze(0)
    
    print(f"\n--- Output for: '{prompt}' ---")
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    for _ in range(max_len):
        # Crop context to the maximum block size the model can handle.
        idx_cond = idx[:, -model.config.block_size:]
        with torch.no_grad(), torch.amp.autocast(device_type):
            logits, _ = model(idx_cond)
            # Temperature scales the logits before Softmax to control creativity.
            logits = logits[:, -1, :] / 0.7
            probs = F.softmax(logits, dim=-1)
            # Multinomial sampling picks the next word based on probability distribution.
            next_id = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, next_id), dim=1)
            # Stream the decoded token to the console immediately.
            print(enc.decode([next_id.item()]), end='', flush=True)
            
    return enc.decode(idx[0].tolist())

if __name__ == "__main__":
    trained_model, tokenizer = train()
    print("\nGeneration complete:")
    generate(trained_model, tokenizer, "Once there was a little robot who")