import time
import torch
import wandb
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, valid_loader, epoch, device):
    model.eval()
    start = time.time()
    running_loss = 0.0
    total_steps = len(valid_loader)
    
    # Create a progress bar
    progress_bar = tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}", total=total_steps)
    
    # Create a W&B Table for detailed logging
    table = wandb.Table(columns=["Step", "Loss"])

    for i, data in enumerate(progress_bar):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        target_ids = data['target_ids'].to(device)
        target_attention_mask = data['target_attention_mask'].to(device)
#         decoder_input_ids= target_ids[:,:-1].contiguous()
#         ecoder_attention_mask=target_attention_mask[:, :-1].contiguous()
#         labels=target_ids[:, 1:].contiguous()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                        decoder_input_ids=target_ids[:,:-1].contiguous(), decoder_attention_mask=target_attention_mask[:, :-1].contiguous(), 
                        labels=target_ids[:, 1:].contiguous())
        
        loss = outputs.loss.item()
        running_loss += loss
        avg_loss = running_loss / (i + 1)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Log step-wise metrics
        wandb.log({
            "val/step_loss": loss,
            "val/avg_loss": avg_loss,
            "val/step": i + 1,
            "val/epoch": epoch + 1,
            "val/progress": (i + 1) / total_steps
        })
        
        # Add data to the table
        table.add_data(i + 1, loss)

    end = time.time()
    final_avg_loss = running_loss / total_steps
    total_time = end - start

    print(f"Validation Epoch: {epoch+1} completed, Avg Loss: {final_avg_loss:.4f}, Time: {total_time:.2f}s")

    # Log epoch-level metrics
    wandb.log({
        "val/epoch_loss": final_avg_loss,
        "val/epoch": epoch + 1,
        "val/epoch_time": total_time
    })

    # Log the table
    wandb.log({"val/loss_table": table})
    torch.cuda.empty_cache()
    # Optional: Log model predictions
    if i+1 % 100 == 0:  # Log every 100th batch
        predictions = model.generate(input_ids)
        wandb.log({
            "val/example_predictions": wandb.Table(
                columns=["Input", "Target", "Prediction"],
                data=[[input_ids[0], target_ids[0], predictions[0]]]
            )
        })

    return final_avg_loss