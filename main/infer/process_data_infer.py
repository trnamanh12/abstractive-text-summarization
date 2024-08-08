import json

def process_data_infer(data):
	single_documents = data.get('single_documents', [])
	# print(single_documents)
	# summary = data.get('summary', '')
	# print(summary)
	
	result = []
	for doc in single_documents:
		raw_text = doc.get('raw_text', '')
		result.append(raw_text)
		# print(raw_text)
	
	return " ".join(result)

def processing_data_infer(input_file):
	all_results = []
	
	with open(input_file, 'r', encoding='utf-8') as file:
		for line in file:
			data = json.loads(line.strip())
			result = process_data_infer(data)
			all_results.append(result)
			# print(result)
			# break
	return all_results