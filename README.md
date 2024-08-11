
# Abstractive Text Summarization

This repository contains code for training and inferencing about Abstractive Text Summarization task. 

## Project Structure

The project is structured into three main directories:

**1. `infer`:**

This directory contains all code related to model inference. This includes:

* **`dataset_for_infer.py`:**  Handles loading, preprocessing, tokenizing data specifically for inference.
* **`hierarchical_2_layer_infer.py`:** Implements a 2-layer hierarchical inference method.
* **`hierarchical_3_layer_infer.py`:** Implements a 3-layer hierarchical inference method.
* **`infer_concat.py`:**  Code for an inference method using concatenation.
* **`process_data_infer.py`:**  Preprocesses and load data for the inference pipeline.
* **`sequential_infer.py`:** Implements a sequential inference method.


**2. `train`:**

This directory contains all code related to model training. This includes:

* **`dataset_for_train.py`:** Handles format, preprocessing, tokenizing data specifically for training.
* **`evaluate.py`:**  Contains code for evaluating the trained model.
* **`process_data.py`:**  Preprocesses data and load data for the training pipeline.
* **`train.py`:**  Contains the main training loop and model definition.

**3. Main Directory:**

* **`main_infer.py`:**  Script to run inference using the trained model.
* **`main_train.py`:**  Script to execute the training process.
* **`README.md`:** This file, providing an overview of the repository.

## Getting Started

To use this code, you will need to:

1. **Install Dependencies:**  Make sure you have the necessary Python packages installed. Do this with `pip install -r requirements.txt` 
2. **Train the Model:** Run `python main_train.py` to train the model. ( Make sure you are in the **main** directory and place the right path containing the data, the model and the API wandb key)
3. **Run Inference:** After training, use `python main_infer.py` to perform inference on new data.
( Make sure you are in the **main** directory and place the right path that containing the data and the model)








