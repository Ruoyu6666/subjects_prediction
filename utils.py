import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
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
    return np.array(node_features, dtype=int)



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



def get_encoded_labels(labels):
    """
    Encode the labels using LabelEncoder
    """
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels)
    return labels, label_encoder.classes_



def get_adjacency_matrix(G):
    """Get the adjacency matrix from the graph."""
    adj_matrix = nx.to_numpy_array(G, nodelist=np.array(list(G.nodes()))).astype(np.float32)

    return adj_matrix



def get_symmetric_normalized_adjacency(adj_matrix):
    """
    Get the symmetric normalized adjacency matrix.
    """
    adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])

    # Normalize adjacency matrix
    rowsum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    return  adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

'''
def save_predictions(paper_ids, predicted_labels, filename="gcn_predictions.tsv"):
    with open(filename, 'w') as f:
        for pid, label in zip(paper_ids, predicted_labels):
            f.write(f"{pid}\t{label}\n")
    print(f"Predictions saved to {filename}")
'''