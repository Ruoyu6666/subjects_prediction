import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder



def load_data(content_path, cites_path):
    '''
    Load the Cora dataset from the given paths.
    content_path: Path to the content file.
    cites_path: Path to the citations file.
    '''
    cites_df = pd.read_csv('./cora/cora.cites', sep='\t',header=None, names=["start", "end"]) 
    content_df = pd.read_csv('./cora/cora.content', sep='\t', header=None)
    return content_df, cites_df



def get_graph(content_df, cites_df):
    """
    Get the graph from the content and citations dataframes.
    REturns a directed graph with nodes having features and labels.
    """
    graph = nx.DiGraph()

    # Add nodes with features
    for _, row in content_df.iterrows():
        node_id = row[0]
        features = row[1:-1].values
        label = row[-1]
        graph.add_node(node_id, feature=features, label=label)

    # Add edges from citations
    for _, row in cites_df.iterrows():
        src = row['start']
        dst = row['end']
        graph.add_edge(src, dst)

    return graph



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
    y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(node_labels)
    
    num_classes = y_onehot.shape[1]
    return y_onehot, y_encoded ,num_classes




    # Convert to numpy array
    return np.array(encoded_labels), label_encoder.classes_


def get_adjacency_matrix(graph):
    """Get the adjacency matrix from the graph."""
    # Convert to undirected graph
    G = graph.to_undirected()
    # Get the adjacency matrix
    adj_matrix = nx.to_numpy_array(G, nodelist=np.array(list(G.nodes())))
    return adj_matrix