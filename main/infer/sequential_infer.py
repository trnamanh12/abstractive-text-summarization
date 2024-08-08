import torch
import time

@torch.no_grad()
def seq_infer(model, train_loader, device):
        model.eval()
        start = time.time()
        all_summaries = []
        for batch in train_loader:
                summaries = []

                inputs = batch['list_input_ids']
                att_mask = batch['list_att_mask']
#                 target = batch['target'].to(device)

                for i in range(len(inputs)-1):
                    if i == 0:
    # 					inputs[i] = inputs[i].to(device)
                        # att_mask[i] = att_mask[i].to(device)
        
                        summary = model.generate(inputs[i].to(device),attention_mask=att_mask[i].to(device), max_length=200, num_beams=4, num_return_sequences =1)
                        summaries.append(summary)
                    else:
                        
                        inputs[i] = torch.cat([inputs[i].to(device), summaries[i-1].to(device)], dim=1)

                        # att_mask[i] = att_mask[i].to(device)
                        summary = model.generate(inputs[i].to(device), max_length=200, num_beams=4, num_return_sequences =1)
                        summaries.append(summary)

                inputs[-1] = torch.cat([inputs[-1].to(device), summaries[-1]], dim=1).to(device)
                summaries = model.generate(inputs[-1], max_length=300, num_beams=4, num_return_sequences =1)
                for k in summaries:
                    all_summaries.append(k.squeeze().cpu())   
        end = time.time()
        print(f"Time: {end-start}")
        return all_summaries