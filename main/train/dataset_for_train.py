import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from main.train.process_data import processing_data, process_data

class Dataset4Sum_train(torch.utils.data.Dataset):
	def __init__(self, tokenizer, data, max_input_length=2560, max_output_length=288):
		self.tokenizer = tokenizer
		self.data = data
		self.max_input_length = max_input_length
		self.max_output_length = max_output_length
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, ids):
		data = self.data[ids]
		input_text, target_text = data[0], data[1]
		input_text =  input_text
		tokenized_input = self.tokenizer(input_text, max_length=self.max_input_length, truncation=True, padding='max_length', return_tensors='pt')
		tokenized_target = self.tokenizer(target_text, max_length=self.max_output_length, truncation=True, padding='max_length', return_tensors='pt')
		return {
			'input_ids': tokenized_input['input_ids'].flatten(),
			'attention_mask': tokenized_input['attention_mask'].flatten(),
			'target_ids': tokenized_target['input_ids'].flatten(),
			'target_attention_mask': tokenized_target['attention_mask'].flatten()
		}

# if __name__ == '__main__':
#     train_file = "/path/to/train_data"
#     valid_file = "/path/to/valid_data"
#     train_data = processing_data(train_file)
#     valid_data = processing_data(valid_file)
#     tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
#     train_data = Dataset4Sum_train(tokenizer, train_data, 2048, 256 )
#     valid_data = Dataset4Sum_train(tokenizer, valid_data, 2048, 256 )
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, num_workers=4, pin_memory=True)
#     valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=10, num_workers=4, pin_memory=True)