from torch.utils.data import Dataset, DataLoader
import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from infer.dataset_for_infer import Dataset4Summarization
from infer.process_data_infer import process_data_infer, processing_data_infer 


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base-vietnews-summarization")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.load_state_dict(torch.load("path/to/your/model.pth"))

# For other demo purpose, you just need to make sure data is list of documents [document1, document2]

data = process_data_infer("path/to/your/test.jsonl")
dataset = Dataset4Summarization(data, tokenizer)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

# batch_size need to be 1,
@torch.no_grad()
def infer_2_hier(model, data_loader, device, tokenizer):
	model.eval()
	start = time.time()
	all_summaries = []
	for iter in data_loader:
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
											early_stopping=True,
											num_return_sequences=1)
				summaries.append(summary)
		summaries = torch.cat(summaries, dim = 1)

		all_summaries.append(tokenizer.decode(summaries.squeeze(), skip_special_tokens=True))

	
	end = time.time()
	print(f"Time: {end-start}")
	return all_summaries

result = infer_2_hier(model, data_loader, device, tokenizer)
print(result)