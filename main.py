import os
import argparse
from dotenv import load_dotenv

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from utils import *
from gcn_model import *


load_dotenv()

CONTENT_PATH = os.getenv('CONTENT_PATH')
CITES_PATH = os.getenv('CITES_PATH')
PREDICTIONS_PATH = os.getenv('PREDICTIONS_PATH')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Convolutional Network for Node Classification")
    parser.add_argument("-epochs", dest="epochs", type=int, default=50,
                        help="Number of epochs to train the model")
    parser.add_argument("-lr", dest="learning_rate", type=float, default=0.01,
                        help="Learning rate for the optimizer")
    args = parser.parse_args()

    num_folds = 10

    print("Loading data...")
    content_df, cites_df = load_data(CONTENT_PATH, CITES_PATH)
    graph = get_graph(content_df, cites_df)  # create graph from dataframes

    paper_ids = content_df.iloc[:, 0].values # list of paper ids
    id_map = {paper_id: i for i, paper_id in enumerate(paper_ids)}

    # Get node features
    features = get_node_features(graph) # get node features, shape (N, F)
    N = features.shape[0]               # number of nodes
    feature_dim = features.shape[1]     # feature dimension

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
    
    
    all_preds = [None] * len(labels)
    predictions = np.zeros(len(labels), dtype=int)
    all_paper_ids = []
    all_pred_labels = []
    all_true_labels = []
    accuracies = []

    # for each fold, train and evaluate the model
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

        optimizer = Adam(learning_rate=args.learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', weighted_metrics=['acc'])

        # Train the model
        model.fit(features, y_encoded, sample_weight=train_mask, epochs=args.epochs, batch_size=N, shuffle=False)
        
        # Evaluate the model
        preds = model.predict(features, batch_size=N)
        pred_classes = np.argmax(preds, axis=1)
        test_ids_map = [id_map[k] for k in test_ids if k in id_map]
        
        acc = accuracy_score(np.argmax(test_labels, axis=1),np.argmax(preds[test_mask],axis=1))
        print(f"Fold {fold + 1} Test Accuracy: {acc:.4f}")
        accuracies.append(acc)
        for id in test_ids_map:
            predictions[id] = pred_classes[id]
            all_preds[id] = (paper_ids[id], classes[pred_classes[id]], labels[id])

with open(PREDICTIONS_PATH, "w") as f:
    f.write("paper_id\tpredicted_label\ttrue_label\n")
    for paper_id, pred_label, true_label in all_preds:
        f.write(f"{paper_id}\t{pred_label}\t{true_label}\n")

#print(f"Average Accuracy: {np.mean(accuracies) * 100:.2f}%")
overall_accuracy = accuracy_score(np.argmax(y_encoded, axis=1), predictions)
print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")
print(f"Predictions saved to {PREDICTIONS_PATH}")