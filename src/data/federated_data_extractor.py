import os
import time
import pandas as pd
import numpy as np
import pickle
import sys
import hashlib
import requests
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Start timing the preprocessing
start_time = time.time()

class BlockchainClient:
    def __init__(self, node_url):
        self.node_url = node_url
        logger.info(f"BlockchainClient initialized with URL: {self.node_url}")

    def add_transaction(self, transaction):
        try:
            # Ensure all required fields are present
            required_fields = {'client': 'FederatedDataExtractor', 
                               'baseindex': 0, 
                               'update': 'data_operation', 
                               'datasize': 0, 
                               'computing_time': 0}
            
            for key, default_value in required_fields.items():
                if key not in transaction:
                    transaction[key] = default_value

            # Add additional fields from the original transaction
            if 'type' in transaction:
                transaction['update'] = transaction['type']
            if 'file_name' in transaction:
                transaction['datasize'] = os.path.getsize(transaction['file_name'])
            
            logger.debug(f"Sending transaction: {transaction}")
            response = requests.post(f"{self.node_url}/transactions/new", json=transaction)
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error adding transaction: {str(e)}")
            return None

    def mine_block(self):
        try:
            response = requests.get(f"{self.node_url}/mine")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error mining block: {str(e)}")
            return None

class FederatedDataExtractor:
    def __init__(self, blockchain_url):
        self.transactions = []
        self.blockchain = BlockchainClient(blockchain_url)
        logger.info(f"FederatedDataExtractor initialized with blockchain URL: {blockchain_url}")

    def load_and_preprocess_data(self, dataset_path, max_features=1000):
        logger.info(f"Attempting to load dataset from: {dataset_path}")
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found at {dataset_path}")
            sys.exit(1)

        logger.info("Loading CSV file...")
        df = pd.read_csv(dataset_path)
        logger.info(f"CSV loaded. Shape: {df.shape}")
        
        texts = df['text'].values
        labels = df['labels'].values
        
        logger.info("Vectorizing text data...")
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(texts).toarray().astype('float32')
        y = labels.astype('float32')
        
        logger.info("Splitting into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        dataset = {
            "train_data": X_train,
            "train_labels": y_train,
            "test_data": X_test,
            "test_labels": y_test,
            "vectorizer": vectorizer
        }

        # Add preprocessing details to blockchain
        preprocessing_transaction = {
            'client': 'FederatedDataExtractor',
            'baseindex': 0,  # Assuming this is the first transaction
            'update': 'preprocessing',
            'datasize': X_train.shape[0],
            'computing_time': time.time() - start_time
        }
        transaction_result = self.blockchain.add_transaction(preprocessing_transaction)
        if transaction_result:
            logger.info("Preprocessing details added to blockchain")
            mining_result = self.blockchain.mine_block()
            if mining_result:
                logger.info("New block mined")
            else:
                logger.warning("Failed to mine new block")
        else:
            logger.warning("Failed to add preprocessing details to blockchain")
        
        logger.info("Dataset preprocessing complete.")
        return dataset

    def hash_data(self, data):
        return hashlib.sha256(pickle.dumps(data)).hexdigest()

    def save_data(self, dataset, name="combined_dataset.pkl"):
        print(f"Attempting to save dataset to: {name}")
        try:
            os.makedirs(os.path.dirname(name), exist_ok=True)
            
            save_data = {
                'train_data': dataset['train_data'],
                'train_labels': dataset['train_labels'],
                'test_data': dataset['test_data'],
                'test_labels': dataset['test_labels'],
                'vectorizer_vocab': dataset['vectorizer'].vocabulary_,
                'vectorizer_idf': dataset['vectorizer'].idf_
            }
            
            print("Dataset keys before saving:", save_data.keys())
            print("Sample of train_data before saving:", save_data['train_data'][:5])
            
            with open(name, "wb") as f:
                pickle.dump(save_data, f)
            
            data_hash = self.hash_data(save_data)
            save_transaction = {
                'type': 'data_save',
                'file_name': name,
                'data_hash': data_hash,
                'computing_time': time.time() - start_time
            }
            self.blockchain.add_transaction(save_transaction)
            self.blockchain.mine_block()
            logger.info(f"Dataset successfully saved to {name}")
            logger.info(f"Data hash: {data_hash}")
        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
            sys.exit(1)

    def load_data(self, name="combined_dataset.pkl"):
        print(f"Attempting to load data from: {name}")
        if not os.path.exists(name):
            print(f"Error: File not found at {name}")
            return None

        try:
            with open(name, "rb") as f:
                load_data = pickle.load(f)
            
            vectorizer = TfidfVectorizer()
            vectorizer.vocabulary_ = load_data['vectorizer_vocab']
            vectorizer.idf_ = load_data['vectorizer_idf']
            
            dataset = {
                'train_data': load_data['train_data'],
                'train_labels': load_data['train_labels'],
                'test_data': load_data['test_data'],
                'test_labels': load_data['test_labels'],
                'vectorizer': vectorizer
            }
            
            print("Dataset keys after loading:", dataset.keys())
            print("Sample of train_data after loading:", dataset['train_data'][:5])
            
            data_hash = self.hash_data(load_data)
            print(f"Loaded data hash: {data_hash}")

            # Verify data integrity using blockchain
            blockchain_record = self.get_latest_data_save(name)
            
            if blockchain_record:
                print(f"Blockchain record hash: {blockchain_record['data_hash']}")
                if blockchain_record['data_hash'] == data_hash:
                    print(f"Data integrity verified for {name}")
                    return dataset
                else:
                    print(f"Data integrity check failed for {name}")
                    print("Hashes do not match.")
                    return None
            else:
                print("No blockchain record found. This might be the first save.")
                return dataset
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None

    def get_latest_data_save(self, file_name):
        for transaction in reversed(self.transactions):
            if transaction['type'] == 'data_save' and transaction['file_name'] == file_name:
                return transaction
        return None


    def get_dataset_details(self, dataset):
        print("Dataset details:")
        for k in dataset.keys():
            if k != 'vectorizer':
                print(f"{k}: shape {dataset[k].shape}")
        logger.info(f"Vectorizer features: {len(dataset['vectorizer'].get_feature_names())}")

    def split_dataset(self, dataset, split_count):
        logger.info(f"Splitting dataset into {split_count} parts...")
        datasets = []
        split_data_length = len(dataset["train_data"]) // split_count
        for i in range(split_count):
            d = dict()
            d["test_data"] = dataset["test_data"][:]
            d["test_labels"] = dataset["test_labels"][:]
            d["train_data"] = dataset["train_data"][i*split_data_length:(i+1)*split_data_length]
            d["train_labels"] = dataset["train_labels"][i*split_data_length:(i+1)*split_data_length]
            d["vectorizer"] = dataset["vectorizer"]
            datasets.append(d)
            
            split_hash = self.hash_data({
                'train_data': d['train_data'],
                'train_labels': d['train_labels'],
                'test_data': d['test_data'],
                'test_labels': d['test_labels'],
                'vectorizer_vocab': d['vectorizer'].vocabulary_,
                'vectorizer_idf': d['vectorizer'].idf_
            })
            split_transaction = {
                'type': 'data_split',
                'split_index': i,
                'split_hash': split_hash,
                'computing_time': time.time() - start_time
            }
            self.transactions.append(split_transaction)
            self.blockchain.add_transaction(split_transaction)
            self.blockchain.mine_block()
            logger.info(f"Split {i} created. Hash: {split_hash}")
        
        return datasets

    def verify_split_integrity(self, split_index, split_data):
        print(f"Verifying integrity of split {split_index}...")
        split_hash = self.hash_data({
            'train_data': split_data['train_data'],
            'train_labels': split_data['train_labels'],
            'test_data': split_data['test_data'],
            'test_labels': split_data['test_labels'],
            'vectorizer_vocab': split_data['vectorizer'].vocabulary_,
            'vectorizer_idf': split_data['vectorizer'].idf_
        })
        blockchain_record = self.get_latest_data_split(split_index)
        
        if blockchain_record:
            print(f"Blockchain record hash: {blockchain_record['split_hash']}")
            print(f"Calculated split hash: {split_hash}")
            if blockchain_record['split_hash'] == split_hash:
                print(f"Split {split_index} integrity verified")
                return True
            else:
                print(f"Split {split_index} integrity check failed")
                return False
        else:
            print(f"No blockchain record found for split {split_index}")
            return False

    def get_latest_data_split(self, split_index):
        for transaction in reversed(self.transactions):
            if transaction['type'] == 'data_split' and transaction['split_index'] == split_index:
                return transaction
        return None

if __name__ == '__main__':
    logger.info("Starting FederatedDataExtractor process")
    if len(sys.argv) < 2:
        print("Please provide the number of clients.")
        sys.exit(1)
    
    num_clients = int(sys.argv[1])

    # Update the blockchain URL to match the running server
    blockchain_url = "http://localhost:5000"
    extractor = FederatedDataExtractor(blockchain_url)
    
    dataset_path = os.path.join("..", "datasets", "combined_dataset.csv")
    print(f"Checking for dataset at: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        print("Please ensure the combined_dataset.csv file is in the 'datasets' directory.")
        sys.exit(1)

    print("\nLoading and preprocessing the combined dataset...")
    dataset = extractor.load_and_preprocess_data(dataset_path)
    
    processed_data_path = os.path.join("data", "processed_dataset.pkl")
    print(f"\nSaving processed dataset to: {processed_data_path}")
    extractor.save_data(dataset, processed_data_path)
    
    print("\nLoading and verifying processed dataset...")
    loaded_dataset = extractor.load_data(processed_data_path)
    if loaded_dataset:
        print("Processed dataset loaded successfully. Details:")
        extractor.get_dataset_details(loaded_dataset)

        print(f"\nSplitting dataset into {num_clients} parts...")
        split_datasets = extractor.split_dataset(loaded_dataset, num_clients)
    
        for n, d in enumerate(split_datasets):
            file_name = os.path.join("data", f"federated_data_{n}.pkl")
            print(f"\nSaving split {n} to: {file_name}")
            extractor.save_data(d, file_name)
            
            print(f"Verifying split {n}...")
            loaded_split = extractor.load_data(file_name)
            if loaded_split:
                integrity_verified = extractor.verify_split_integrity(n, loaded_split)
                if integrity_verified:
                    print(f"Split {n} details:")
                    extractor.get_dataset_details(loaded_split)
                else:
                    print(f"Split {n} integrity check failed. Details may be incorrect.")
            else:
                print(f"Failed to load split {n}")

print("\nScript execution completed.")