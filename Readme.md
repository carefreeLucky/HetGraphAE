# HetGraphAE

This repository contains the code for Anomaly Process Detection via Heterogenous Graph Autoencoder.

# Data

The data set is divided into training data and test data, which are stored in the trainData directory and the testData directory respectively. Contains the adjacency matrix of the graph and the attribute matrix of the node.

## Requirements

- PyTorch 1.0 or higher
- Python 3.6

# Usage

## Training Model

The starting file for training is train.py. After training, the model will be saved in the save_model directory, which already has the trained model in the paper. The model files for training different layers are stored in the directory train_diff_layer

## Testing Model

The result.py will call the test.py for testing, and finally output the metrics of the model. The anomaly score of each node will be saved in the HetGraphAELoss.npy. As for the ROC curve, it can be obtained by running the ROC.py after running the result.py.