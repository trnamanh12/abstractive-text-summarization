import torch
import time

@torch.no_grad()
def infer_3_hier(model , train_loader, device):
        model.eval()
        all_summaries = []
        start = time.time()
        for batch in train_loader:
                summaries = []
                summaries_2 = []
                # model.zero_grad(set_to_none=True)
                inputs = batch['list_input_ids']
                att_mask = batch['list_att_mask']
#                 target = batch['target'].to(device)

                # layer 1 of the hierarchy
                for i in range(len(inputs)):
#                     inputs[i] = inputs[i].to(device)
                    # att_mask[i] = att_mask[i].to(device)
                    if torch.all(inputs[i] == 0):
                # If the input is all zeros, skip this iteration
                        continue
                    summary = model.generate(inputs[i].to(device),attention_mask=att_mask[i].to(device), max_length=128, num_beams=4, num_return_sequences =1)
                    summaries.append(summary)

                # layer 2 of the hierarchy	
                for i in range(0, len(summaries), 2):
                    # summaries[i] = summaries[i].to(device)
                    summaries_2.append(model.generate(torch.cat(summaries[i:i+2], dim=1).to(device), max_length=128, num_beams=4, num_return_sequences =1))

                summaries_2 = torch.cat(summaries_2, dim=1).to(device)

                # layer 3 of the hierarchy
                final_summary = model.generate(summaries_2, max_length=256, num_beams=4, num_return_sequences =1)
        
                all_summaries.append(final_summary)
        end = time.time()
        print(f"Time: {end-start}")
        return all_summaries
