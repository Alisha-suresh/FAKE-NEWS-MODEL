import hashlib
import json
import time
from flask import Flask, jsonify, request
from uuid import uuid4
from urllib.parse import urlparse
import requests
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Block:
    def __init__(self, index, transactions, timestamp, previous_hash, nonce=0):
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    @classmethod
    def from_dict(cls, block_dict):
        block = cls(
            index=block_dict['index'],
            transactions=block_dict['transactions'],
            timestamp=block_dict['timestamp'],
            previous_hash=block_dict['previous_hash'],
            nonce=block_dict['nonce']
        )
        block.hash = block_dict['hash']
        return block

class Blockchain:
    def __init__(self):
        self.chain = []
        self.transactions = []
        self.nodes = set()
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, [], 1624050000, "0")  # Use a fixed timestamp for genesis block
        self.chain.append(genesis_block)
        logger.info(f"Genesis block created: {genesis_block.__dict__}")

    def get_last_block(self):
        return self.chain[-1]

    def add_block(self, block):
        self.chain.append(block)
        logger.info(f"New block added: {block.__dict__}")

    def add_transaction(self, transaction):
        self.transactions.append(transaction)
        return self.get_last_block().index + 1

    def hash(self, block):
        block_string = json.dumps(block.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def proof_of_work(self, block):
        block.nonce = 0
        computed_hash = block.calculate_hash()
        while not computed_hash.startswith('0000'):
            block.nonce += 1
            computed_hash = block.calculate_hash()
        logger.debug(f"Proof of work found. Nonce: {block.nonce}, Hash: {computed_hash}")
        return computed_hash

    def add_node(self, address):
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)
    
    def get_nodes(self):
        return list(self.nodes)

    def is_chain_valid(self, chain):
        for i in range(1, len(chain)):
            current = chain[i]
            previous = chain[i-1]
            logger.debug(f"Validating block {i}")
            logger.debug(f"Current block: {current.__dict__}")
            logger.debug(f"Previous block: {previous.__dict__}")

            if current.previous_hash != previous.hash:
                logger.warning(f"Invalid previous hash for block {i}")
                return False
            
            if not current.hash.startswith('0000'):
                logger.warning(f"Invalid proof of work for block {i}")
                return False
            
            logger.debug(f"Block {i} is valid")
        return True

    def resolve_conflicts(self):
        neighbors = self.nodes
        new_chain = None
        max_length = len(self.chain)
        logger.info(f"Current chain length: {max_length}")
        
        for node in neighbors:
            logger.info(f"Checking node: {node}")
            response = requests.get(f'http://{node}/chain')
            if response.status_code == 200:
                length = response.json()['length']
                chain = [Block.from_dict(block) for block in response.json()['chain']]
                logger.info(f"Node {node} chain length: {length}")
                
                if length > max_length and self.is_chain_valid(chain):
                    max_length = length
                    new_chain = chain
                    logger.info(f"New valid chain accepted with length: {max_length}")
                else:
                    logger.warning(f"Chain from node {node} is invalid or not longer")
            else:
                logger.warning(f"Failed to get chain from node {node}")
        
        if new_chain:
            self.chain = new_chain
            logger.info("Chain replaced")
            return True
        
        logger.info("Current chain is up to date")
        return False

    def mine_block(self):
        last_block = self.get_last_block()
        new_block = Block(index=last_block.index + 1,
                          transactions=self.transactions,
                          timestamp=time.time(),
                          previous_hash=last_block.hash)
        new_block.hash = self.proof_of_work(new_block)
        self.add_block(new_block)
        self.transactions = []
        return new_block

# Initialize Flask app
app = Flask(__name__)

# Generate a globally unique address for this node
# node_identifier = str(uuid4()).replace('-', '')


# Initialize Blockchain
blockchain = Blockchain()

@app.route('/mine', methods=['GET'])
def mine():
    block = blockchain.mine_block()
    response = {
        'message': "New Block Forged",
        'index': block.index,
        'transactions': block.transactions,
        'proof': block.nonce,
        'previous_hash': block.previous_hash,
    }
    return jsonify(response), 200

@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.get_json()
    logger.debug(f"Received transaction data: {values}")

    required = ['client', 'baseindex', 'update', 'datasize', 'computing_time']
    if not all(k in values for k in required):
        missing_fields = [k for k in required if k not in values]
        logger.error(f"Missing values in transaction: {missing_fields}")
        return 'Missing values', 400

    # Create a new Transaction
    index = blockchain.add_transaction(values)

    response = {'message': f'Transaction will be added to Block {index}'}
    return jsonify(response), 201

@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': [vars(block) for block in blockchain.chain],
        'length': len(blockchain.chain),
    }
    return jsonify(response), 200

@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Please supply a valid list of nodes", 400

    for node in nodes:
        blockchain.add_node(node)

    response = {
        'message': 'New nodes have been added',
        'total_nodes': list(blockchain.nodes),
    }
    return jsonify(response), 201

@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    try:
        replaced = blockchain.resolve_conflicts()
        if replaced:
            response = {
                'message': 'Our chain was replaced',
                'new_chain': [vars(block) for block in blockchain.chain]
            }
        else:
            response = {
                'message': 'Our chain is authoritative',
                'chain': [vars(block) for block in blockchain.chain]
            }
        return jsonify(response), 200
    except Exception as e:
        print(f"Error in consensus: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    response = {
        'status': "receiving",
        'last_model_index': blockchain.get_last_block().index
    }
    return jsonify(response), 200

@app.route('/length', methods=['GET'])
def chain_length():
    return jsonify({'length': len(blockchain.chain)}), 200

@app.route('/nodes/list', methods=['GET'])
def list_nodes():
    nodes = blockchain.get_nodes()
    response = {
        'nodes': nodes,
        'total_nodes': len(nodes)
    }
    return jsonify(response), 200


# if __name__ == '__main__':
#     from argparse import ArgumentParser
#     parser = ArgumentParser()
#     parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
#     args = parser.parse_args()
#     port = args.port
#     app.run(host='0.0.0.0', port=port, debug=True)
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port
    # Generate a globally unique address for this node
    node_identifier = f"http://localhost:{port}"
    app.run(host='localhost', port=port, debug=True)