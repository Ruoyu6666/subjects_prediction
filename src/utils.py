import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



def load_data(content_path, cites_path):
    '''
    Load the Cora dataset from the given paths.
    '''
    cites_df = pd.read_csv('./cora/cora.cites', sep='\t',header=None, names=["start", "end"]) 
    content_df = pd.read_csv('./cora/cora.content', sep='\t', header=None)
    return content_df, cites_df



def get_graph(content_df, cites_df):
    """
    Get the graph from the content and citations dataframes.
    Returns a graph with nodes having features and labels.
    """
    G = nx.DiGraph()

    # Add nodes with features
    for _, row in content_df.iterrows():
        node_id = row[0]
        features = row[1:-1].values
        label = row.values[-1]
        G.add_node(node_id, feature=features, label=label)

    # Add edges from citations
    for _, row in cites_df.iterrows():
        src = row['start']
        dst = row['end']
        G.add_edge(src, dst)

    return G



def get_node_features(graph):
    """
    Get the node features from the graph.
    """
    # Get the node features
    node_features = []
    for node in graph.nodes():
        node_features.append(graph.nodes[node]['feature'])
    
    # Convert to numpy array
    return np.array(node_features)



def get_node_labels(graph):
    """
    Get the node labels from the graph.
    """
    # Get the node labels
    node_labels = []
    for node in graph.nodes():
        node_labels.append(graph.nodes[node]['label'])
    
    # Convert to numpy array
    return np.array(node_labels)



def get_encoded_labels(graph):
    """
    Get the encoded labels from the graph.
    """
    # Get the node labels
    node_labels = get_node_labels(graph)
    
    # Encode the labels
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_onehot = onehot_encoder.fit_transform(node_labels.reshape(-1, 1))
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(node_labels)

    return y_onehot, y_encoded



def get_adjacency_matrix(G):
    """Get the adjacency matrix from the graph."""
    adj_matrix = nx.to_numpy_array(G, nodelist=np.array(list(G.nodes()))).astype(np.float32)

    return adj_matrix



def get_symmetric_normalized_adjacency(adj_matrix):
    """
    Get the symmetric normalized adjacency matrix.
    """
    A_hat = adj_matrix + np.eye(adj_matrix.shape[0])
    D_inv = tf.linalg.tensor_diag(tf.pow(tf.reduce_sum(A_hat, 0), tf.cast(-0.5, tf.float32)))
    D_inv = tf.where(tf.math.is_inf(D_inv), tf.zeros_like(D_inv), D_inv)

    return  D_inv @ A_hat @ D_inv


def save_predictions(paper_ids, predicted_labels, filename="gcn_predictions.tsv"):
    with open(filename, 'w') as f:
        for pid, label in zip(paper_ids, predicted_labels):
            f.write(f"{pid}\t{label}\n")
    print(f"Predictions saved to {filename}")