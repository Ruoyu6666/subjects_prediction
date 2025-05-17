import os
from dotenv import load_dotenv

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from sklearn.metrics import classification_report
from src.utils import *
from src.gcn_model import *


load_dotenv()

CONTENT_PATH = os.getenv('CONTENT_PATH')
CITES_PATH = os.getenv('CITES_PATH')
PREDICTIONS_PATH = os.getenv('PREDICTIONS_PATH')


if __name__ == "__main__":

    num_folds = 10
    epochs = 50
    learning_rate = 0.01
    rate = 0.2

    print("Loading data...")
    content_df, cites_df = load_data(CONTENT_PATH, CITES_PATH)
    paper_ids = content_df.iloc[:, 0].values # list of paper ids
    graph = get_graph(content_df, cites_df)  # create graph from dataframes

    # Get node features
    features = get_node_features(graph) # get node features, shape (N, F)
    N = features.shape[0]               # number of nodes
    feature_dim = features.shape[1]

    #Get raw labels
    labels = get_node_labels(graph)
    label_set = sorted(set(labels))
    num_classes = len(label_set)

    # Encode labels
    y_encoded, classes = get_encoded_labels(labels)

    # Get adjacency matrix
    A= get_adjacency_matrix(graph)
    A = get_symmetric_normalized_adjacency(A)
    A = tf.convert_to_tensor(A, tf.float32)
    
    
    all_preds = np.zeros(len(labels), dtype=int)
    all_paper_ids = []
    all_pred_labels = []
    all_true_labels = []
    fold_accuracies = []

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(features, y_encoded)):
        print(f"Fold {fold + 1}/{num_folds}")

        train_mask = np.zeros((N,),dtype=bool)
        train_mask[train_index] = True

        test_mask = np.zeros((N,),dtype=bool)
        test_mask[test_index] = True

        train_features, test_features = features[train_mask],  features[test_mask]
        train_labels, test_labels     = y_encoded[train_mask], y_encoded[test_mask]
        train_ids, test_ids           = paper_ids[train_index], paper_ids[test_index]

        # Build the model
        x_input = tf.keras.Input(shape=(feature_dim,))
        gcn1 = GraphConvolution(A, 64, activation='relu')(x_input)
        gcn2 = GraphConvolution(A, num_classes, activation='softmax')(gcn1)
        model = Model(inputs=x_input, outputs=gcn2)

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', weighted_metrics=['acc'])


        # Train the model
        model.fit(features, y_encoded, sample_weight=train_mask, epochs=epochs, batch_size=N, shuffle=False)
        
        # Evaluate the model
        preds = model.predict(features, batch_size=N)
        pred_classes = np.argmax(preds, axis=1)

        report = classification_report(np.argmax(test_labels, axis=1), 
                                        np.argmax(preds[test_mask],axis=1), 
                                        target_names=classes)    
        
        #all_preds[test_index] = predicted_classes
        #all_paper_ids.extend(test_ids)
        #all_predicted_labels.extend([list(label_map.keys())[list(label_map.values()).index(pred)] for pred in predicted_classes])
        
        #print(f"Fold {fold + 1} Test Accuracy: {test_accuracy:.4f}")
        print('GCN Classification Report: \n {}'.format(report))
        #fold_accuracies.append(test_accuracy)

    #save_predictions(all_paper_ids, all_predicts)


    #overall_accuracy = accuracy_score(np.argmax(labels_onehot, axis=1), np.argmax(all_preds, axis=1))
    #print(f"\nOverall GCN Accuracy: {overall_accuracy:.4f}")

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