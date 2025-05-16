import os
from dotenv import load_dotenv
from src.utils import *
from src.gcn_model import GraphConvolution, build_gcn_model
from src.train_eval import train_and_evaluate


load_dotenv()


CONTENT_PATH = os.getenv('CONTENT_PATH')
CITES_PATH = os.getenv('CITES_PATH')
PREDICTIONS_PATH = os.getenv('PREDICTIONS_PATH')



if __name__ == "__main__":

    print("Loading data...")
    content_df, cites_df = load_data(CONTENT_PATH, CITES_PATH)
    paper_ids = content_df.iloc[:, 0].values # list of paper ids
    
    graph = get_graph(content_df, cites_df)

    # Get node features
    node_features = get_node_features(graph)
    N = node_features.shape[0] # number of nodes

    #Get raw labels
    node_labels = get_node_labels(graph)
    label_set = sorted(set(node_labels))
    num_classes = len(label_set)
    
    # Encode labels
    y_onehot, y_encoded = get_encoded_labels(graph)

    # Get adjacency matrix
    adjacency_matrix = get_adjacency_matrix(graph)
    A = get_symmetric_normalized_adjacency(adjacency_matrix)

    for _ in range(10):
        model = build_gcn_model((, ), 64, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        '''
        predictions, predicted_ids, overall_accuracy, fold_accuracies = train_and_evaluate(
            paper_ids, node_features, node_labels, y_onehot, y_encoded, num_classes,
            A, n_splits=10, epochs=200, learning_rate=0.01
        )

        output_df = pd.DataFrame({'paper_id': predicted_ids, 'class_label': predictions})
        output_file = 'cora_gcn_predictions.tsv'
        output_df.to_csv(output_file, sep='\t', index=False, header=False)
        print(f"\nPredictions saved to {output_file}")
        print(f"\nOverall GCN Accuracy: {overall_accuracy:.4f}")
        '''
if __name__ == "__main__":
    main()