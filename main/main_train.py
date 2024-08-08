"""
The purpose of this file is for training model
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from train.dataset_for_train import Dataset4Sum_train
from train.process_data import processing_data
import wandb
from tqdm import tqdm
import time
import os
from train.train import train
from train.evaluate import evaluate

def run(model, train_loader, valid_loader, optimizer, scheduler, device, num_epochs, config):
    # Initialize wandb
    best_model_path = False
    run = wandb.init(project="ViT5 base fine-tune abmusu",name="cp18_lr16e-6" ,config=config)
    
    # Watch the model to log gradients and model parameters
    wandb.watch(model, log="all", log_freq=100)
    
    best_val_loss = float('inf')
    start = time.time()
    best_model_artifact = None
    
    # Create a progress bar for epochs
    epoch_bar = tqdm(range(num_epochs), desc="Training Progress")
    
    for epoch in epoch_bar:
        # Training
        train_loss = train(model, valid_loader, optimizer, scheduler, epoch, device)
        torch.cuda.empty_cache()
        # Validation
        val_loss = evaluate(model, valid_loader, epoch, device)
        torch.cuda.empty_cache()
        
        # Update best model if necessary
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save the model locally
            model_path = f"/path/to/your/model_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            if best_model_path and os.path.exists(best_model_path):
                 os.remove(best_model_path) 
            best_model_path = model_path

            # # Create a new artifact
            # artifact = wandb.Artifact(f"best_model_epoch_{epoch+1}", type="model")
            # artifact.add_file(model_path)
            
            # # Log the new artifact
            # run.log_artifact(artifact)
             
            # if best_model_path and os.path.exists(best_model_path):
            #      os.remove(best_model_path)
            # If there's a previous best model, delete it
            # if best_model_artifact:
            #     run.delete_artifact(best_model_artifact)
            
            # Update the best model artifact reference
            best_model_path = model_path
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": train_loss,
#             "val/epoch_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Update progress bar
        epoch_bar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
#             'Val Loss': f'{val_loss:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    end = time.time()
    total_time = end - start
    print(f"Total training time: {total_time:.2f} seconds")
    
    # Log final metrics
    wandb.log({
        "total_training_time": total_time,
#         "best_val_loss": best_val_loss
    })
    
    # Create a summary plot
#     wandb.log({
#         "training_summary": wandb.plot.line_series(
#             xs=[[e+1 for e in range(num_epochs)]] * 2,
#             ys=[run.history["train/epoch_loss"], run.history["val/epoch_loss"]],
#             keys=["Train Loss", "Validation Loss"],
#             title="Training and Validation Loss Over Epochs",
#             xname="Epoch"
#         )
#     })
    
    # Close wandb run
    wandb.finish()



if __name__ == '__main__':
    train_file = "/path/to/train_data"
    valid_file = "/path/to/valid_data"

    train_data = processing_data(train_file)
    valid_data = processing_data(valid_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base-vietnews-summarization")

    model.to(device)
    
    train_data = Dataset4Sum_train(tokenizer, train_data, 2048, 256 )
    valid_data = Dataset4Sum_train(tokenizer, valid_data, 2048, 256 )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=10, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
    wandb.login(key="Insert your key here")
    config = {
        "learning_rate": 16e-6,
        "architecture": "ViT5",
        "dataset": "abmusu",
        "epochs": 2,
        "gamma": 0.98,
        "scheduler": "StepLR",
        "batch_size": 32,
        "model_size": "base"
    }
    run(model, train_loader, valid_loader, optimizer, scheduler, device, num_epochs=config["epochs"], config=config)



