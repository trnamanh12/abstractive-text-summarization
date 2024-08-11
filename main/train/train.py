import time
import torch
import wandb
from tqdm import tqdm

def train(model, train_loader, optimizer, scheduler, epoch, device):
    model.train()
    start = time.time()
    if epoch == 0:
        model.zero_grad(set_to_none=True)
    running_loss = 0.0
    total_steps = len(train_loader)
    
    # Create a wandb Table
    table = wandb.Table(columns=["Step", "Loss", "Learning Rate"])
    
    # Create a progress bar for this epoch
    progress_bar = tqdm(train_loader, total=total_steps, desc=f"Train Epoch {epoch+1}")
    
    for i, data in enumerate(progress_bar):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        target_ids = data['target_ids'].to(device)
        target_attention_mask = data['target_attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                        decoder_input_ids=target_ids[:, :-1].contiguous(), decoder_attention_mask=target_attention_mask[:, :-1].contiguous(), 
                        labels=target_ids[:,1:].contiguous())
        
        loss = outputs.loss
        running_loss += loss.item()
        loss = loss / 60.0
        loss.backward()
        
        if (((i+1) % 60 == 0 & i != 0) | i ==(len(train_loader) -1) ):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()
            scheduler.step()
            model.zero_grad(set_to_none=True)
        
        if i % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = running_loss / (i + 1)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
            # Log metrics
            wandb.log({
                "train/loss": avg_loss,
                "train/learning_rate": current_lr,
                "train/epoch": epoch + 1,
                "train/step": i + 1,
                "train/progress": (i + 1) / total_steps,
            })
            
            # Add data to the table
            table.add_data(i + 1, avg_loss, current_lr)
    
    end = time.time()
    epoch_loss = running_loss / len(train_loader)
    epoch_time = end - start
    
    print(f"Train Epoch: {epoch+1} completed, Avg Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
    
    # Log epoch-level metrics
    wandb.log({
        "train/epoch_loss": epoch_loss,
        "train/epoch": epoch + 1,
        "train/epoch_time": epoch_time,
        "train/progress_table": table
    })
    
    return epoch_loss



