import torch
import time
from typing import List
@torch.no_grad()
def infer_2_hierachical(model, train_loader, device, tokenizer):
    model.eval()
    start = time.time()
    all_summaries = []
    for iter in train_loader:
        summaries = []
        inputs = iter['list_input_ids']
        att_mask = iter['list_att_mask']
        
        for i in range(len(inputs)):
            # Check if the input tensor is all zeros
            if torch.all(inputs[i] == 0):
                # If the input is all zeros, skip this iteration
                continue
            else:
                summary = model.generate(inputs[i].to(device),
                                         attention_mask=att_mask[i].to(device),
                                         max_length=128,
                                         num_beams=4,
                                         num_return_sequences=1)
                summaries.append(summary)
        summaries = torch.cat(summaries, dim = 1)
        summaries = model.generate(summaries.to(device),max_length=256, num_beams=4, num_return_sequences=1 )
        all_summaries.append(summaries)

    
    end = time.time()
    print(f"Time: {end-start}")
    return all_summaries