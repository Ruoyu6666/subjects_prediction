# Subjects Prediction Cora Dataset

In this task a Graph Convolutional Network (GCN) based approach is implemented for classifying the subjects of scientific papers in the Cora dataset. The model is trained using 10-fold cross-validation and makes predictions for each paper using graph learning techniques.

## Idea
The Cora dataset forms a graph where nodes are papers and edges are citations. Each node has a sparse feature vector representing words in the paper. Graph Convolutional Networks are adopted to propagate information from neighboring nodes to learn embeddings for classification.

Each GCN layer updates node representations using the following operation:

$$
H^{(l+1)} = \sigma\left( \hat{A} H^{(l)} W^{(l)} \right),
\hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}
$$

Where:

- $H^{(l)}$ is the node feature matrix at layer $l$ (with $H^{(0)} = X$, the input features),
- $W^{(l)}$ is the trainable weight matrix for layer $l$,
- $\sigma$ is an activation function (e.g., ReLU),
- $\tilde{A} = A + I$ is the adjacency matrix with self-loops added,
- $\tilde{D}$ is the degree matrix of $\tilde{A}$.

### Model Architecture
- GCN Layer 1: Transforms input features into a hidden representation.
- ReLU Activation
- GCN Layer 2: Maps hidden representation to class scores.
- Softmax is applied to the output for classification.

## Requirements
- dotenv
- pandas
- tensorflow
- networkx
- scikit-learn
- scipy


## Runs

To train and test the network with the CORA dataset.

```bash
python main.py -epochs 50 -lr 0.01
```
## Results

### Output
All predictions are saved to output/predictions.tsv with columns:
- paper_id
- predicted_label
- true_label

The overall accuracy is 86.00%
