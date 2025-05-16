
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from src.utils import *
from src.gcn_model import *


def train_evaluate(paper_ids, num_nodes, features, 
                    labels, labels_onehot, labels_encode, num_classes, 
                    adj, 
                    n_splits=10, epochs=100, learning_rate=0.01):
    '''
    Train and evaluate the GCN model using K-Fold cross-validation.
    '''
    skf = StratifiedKFold(n_splits=n_splits, random_state=42)
    all_preds = np.zeros(num_nodes, dtype=int)
    #all_paper_ids = []
    #all_predicted_labels = []
    #all_true_labels = []
    fold_accuracies = []

    for fold, (train_index, test_index) in enumerate(skf.split(features, labels_encode)):
        print(f"Fold {fold + 1}/{n_splits}")

        train_features, test_features = features[train_index], features[test_index]
        train_labels, test_labels = labels_onehot[train_index], labels_onehot[test_index]
        train_ids, test_ids = paper_ids[train_index], paper_ids[test_index]

        # Create the model
        model = build_gcn_model(train_features.shape[1], 64, num_classes)
        optimizer = Adam(learning_rate=learning_rate)
        loss_fn = CategoricalCrossentropy()
        accuracy_metric = Accuracy()
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=[accuracy_metric])

        # Train the model
        model.fit([train_features, adj], train_labels, epochs=epochs, batch_size=32, verbose=0)
        '''
        # Evaluate the model
        test_preds = model.predict(test_features)
        test_accuracy = accuracy_score(np.argmax(test_labels, axis=1), np.argmax(test_preds, axis=1))
        fold_accuracies.append(test_accuracy)

        # Store predictions and ids
        all_preds[test_index] = test_preds
        all_paper_ids[test_index] = test_ids
    overall_accuracy = accuracy_score(np.argmax(labels_onehot, axis=1), np.argmax(all_preds, axis=1))
    return all_preds, all_paper_ids, overall_accuracy, fold_accuracies
