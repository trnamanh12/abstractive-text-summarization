# create dataset class
from torch.utils.data import Dataset, DataLoader
import torch
import json
from transformers import AutoTokenizer


class Dataset4Summarization(Dataset):
	def __init__(self, data, tokenizer, max_length=1024*3, chunk_length =1024, overlap=32):
		self.data = data
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.chunk_length = chunk_length
		self.overlap = overlap

	def __len__(self):
		return len(self.data)
	
	def chunking(self, text):
		chunks = []
		for i in range(0, self.max_length, self.chunk_length):
			chunks.append(text[i:i+self.chunk_length])
		return chunks

	def __getitem__(self, idx):
		sample = self.data[idx]
		inputs = self.tokenizer(sample, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
		list_chunk = self.chunking(inputs['input_ids'].squeeze())
		list_attention_mask = self.chunking(inputs['attention_mask'].squeeze())


		# del text
		return {
			'list_input_ids': list_chunk,
			'list_att_mask' : list_attention_mask,
		}

