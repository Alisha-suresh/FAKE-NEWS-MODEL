# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import os
import re
import sys
import uuid
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import requests
import hashlib
import requests
import time
import logging
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Blockchain class (simplified from your blockchain.py)
class Blockchain:
    def __init__(self, node_url):
        self.node_url = node_url

    def add_transaction(self, transaction):
        response = requests.post(f"{self.node_url}/transactions/new", json=transaction)
        return response.json()

    def mine_block(self):
        response = requests.get(f"{self.node_url}/mine")
        return response.json()

    def get_chain(self):
        response = requests.get(f"{self.node_url}/chain")
        return response.json()
    
# 1. Load and Preprocess the Text Data
def load_and_preprocess_data(dataset_path, max_features):
    import pandas as pd

    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    texts = df['text'].values
    labels = df['labels'].values
    
    # Vectorize the text data using TF-IDF
    logger.info(f"Vectorizing text data with max_features={max_features}")
    vectorizer = TfidfVectorizer(max_features=max_features)  # Limit to max_features
    X = vectorizer.fit_transform(texts).toarray().astype('float32')  # Convert features to float32
    y = labels.astype('float32')  # Convert labels to float32

    logger.info(f"Preprocessed data shape: X={X.shape}, y={y.shape}")
    return X, y, vectorizer

# 2. Split Data into Train/Test and Clients for Federated Learning
def create_federated_data(X, y, num_clients=10):
    logger.info(f"Splitting data into train/test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Split data into clients
    logger.info(f"Creating federated data for {num_clients} clients")
    data_per_client = len(X_train) // num_clients
    federated_data = []
    
    for i in range(num_clients):
        client_X = X_train[i * data_per_client: (i + 1) * data_per_client]
        client_y = y_train[i * data_per_client: (i + 1) * data_per_client]
        
        # Convert to tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((client_X, client_y))
        dataset = dataset.batch(32)  # Adjust batch size as needed
        federated_data.append(dataset)

    logger.info(f"Created {len(federated_data)} federated datasets")
    return federated_data, (X_test, y_test)

# 3. Load Model Parameters
def load_model_structure(structure_path):
    logger.info(f"Loading model structure from {structure_path}")
    with open(structure_path, 'r') as f:
        model_structure = json.load(f)
    return model_structure

# 4. Define a Simplified Neural Network Model
def create_text_classification_model(input_dim):
    logger.info(f"Creating text classification model with input_dim={input_dim}")
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# 5. Federated Learning Setup
def model_fn(input_dim):
    model = create_text_classification_model(input_dim)
    return tff.learning.from_keras_model(
        model,
        input_spec=(
            tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.float32)
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# 6. Learning Rate Schedule
def learning_rate_schedule():
    initial_lr = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    return lr_schedule

# FedAvg process for Federated Learning with Client Sampling
def build_fedavg_process(model_fn, input_dim):
    return tff.learning.build_federated_averaging_process(
        model_fn=lambda: model_fn(input_dim),
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule()),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.5)
    )

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif tf.is_tensor(obj):
            return obj.numpy().tolist()
        return super(NumpyEncoder, self).default(obj)

def run_federated_training(federated_data, blockchain, num_rounds=360, input_dim=100, clients_per_round=8):
    learning_process = build_fedavg_process(model_fn, input_dim)
    state = learning_process.initialize()

    for round_num in range(1, num_rounds + 1):
        start_time = time.time()
        logger.info(f"Starting round {round_num}/{num_rounds}")
        sampled_clients = federated_data[:clients_per_round]
        state, metrics = learning_process.next(state, sampled_clients)
        
        # Convert metrics to a standard Python dictionary
        metrics_dict = {key: value.numpy().tolist() if hasattr(value, 'numpy') else value for key, value in metrics.items()}
        
        # Serialize and hash the model weights
        model_weights = state.model.trainable
        model_weights_str = json.dumps(model_weights, cls=NumpyEncoder)
        model_hash = hashlib.sha256(model_weights_str.encode()).hexdigest()
        
        # Calculate computing time
        computing_time = time.time() - start_time
        
        # Create a blockchain transaction for this round
        transaction = {
            'round': round_num,
            'model_hash': model_hash,
            'metrics': metrics_dict,
            'timestamp': time.time(),
            'client': str(uuid.uuid4()),
            'baseindex': round_num - 1,
            'update': model_hash,
            'datasize': sum(len(client_data) for client_data in sampled_clients),
            'computing_time': computing_time
        }
        
        # Use the custom encoder to serialize the transaction
        transaction_json = json.dumps(transaction, cls=NumpyEncoder)
        response = blockchain.add_transaction(json.loads(transaction_json))
        
        # Check if the transaction was successful
        if 'message' in response and re.match(r'Transaction will be added to Block \d+', response['message']):
            logger.info(f"Transaction for round {round_num} added successfully: {response['message']}")
        else:
            logger.error(f"Failed to add transaction for round {round_num}: {response}")
        
        # Mine a new block
        mine_response = blockchain.mine_block()
        logger.info(f"Mining response for round {round_num}: {mine_response}")
        
        logger.info(f'Round {round_num} completed. Metrics: {metrics_dict}')
        logger.info(f'Model hash for round {round_num}: {model_hash}')

    return state

# 8. Evaluate the Model
def evaluate_model(global_model, X_test, y_test):
    logger.info("Evaluating the final model")
    y_pred = global_model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-score: {f1:.4f}")
    logger.info(f"AUC: {auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


# Main Flow
if __name__ == "__main__":
    logger.info("Starting the Blockchain-Integrated Federated Learning process")
    
    #  Define the blockchain node URL
    blockchain_url = "http://localhost:5000"  # Adjust the URL as needed

    # Initialize blockchain
    blockchain = Blockchain(blockchain_url) # Adjust the URL as needed
    logger.info("Blockchain initialized")

    dataset_path = 'datasets/combined_dataset.csv'
    structure_path = 'model/stacking_ensemble_structure.json'

    model_structure = load_model_structure(structure_path)
    input_dim = model_structure['n_features']

    logger.info(f"Loading and preprocessing data from {dataset_path}")
    X, y, vectorizer = load_and_preprocess_data(dataset_path, max_features=input_dim)

    logger.info("Creating federated datasets")
    federated_train_data, (X_test, y_test) = create_federated_data(X, y, num_clients=10)

    logger.info("Starting federated training")
    state = run_federated_training(federated_train_data, blockchain, num_rounds=360, input_dim=input_dim, clients_per_round=8)

    logger.info("Federated training completed")

    # Load the global model and set its weights to the ones obtained after federated learning
    global_model = create_text_classification_model(input_dim)
    global_model.set_weights(state.model.trainable)

# Evaluate the final global model
    final_metrics = evaluate_model(global_model, X_test, y_test)
    # Add final evaluation to blockchain
    final_transaction = {
        'type': 'final_evaluation',
        'metrics': {
            'accuracy': final_metrics['accuracy'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'f1': final_metrics['f1'],
            'auc': final_metrics['auc']
        },
        'timestamp': time.time(),
        'client': 'final_evaluation',  # Placeholder value
        'baseindex': 360,              # Assuming it's the final round
        'update': 'final_evaluation',  # Placeholder value
        'datasize': 0,                 # No data associated
        'computing_time': 0            # No specific computing time
    }
    blockchain.add_transaction(final_transaction)
    blockchain.mine_block()

    logger.info("Final evaluation added to blockchain")

    # Save additional information for dissertation
    report = {
        'final_metrics': final_metrics,
        'total_rounds': 360,
        'clients_per_round': 8,
        'input_dim': input_dim,
        'dataset_path': dataset_path,
        'model_structure_path': structure_path
    }

    os.makedirs('results', exist_ok=True)
    with open('results/federated_learning_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    logger.info("Federated learning report saved to results/federated_learning_report.json")
    logger.info("Blockchain-Integrated Federated Learning process completed")