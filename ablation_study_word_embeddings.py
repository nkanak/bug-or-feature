# Standard Library Imports
import json
import random
import sys

# Data Manipulation and Visualization Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Text Processing Imports
from gensim.parsing.preprocessing import remove_stopwords
import contractions
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.manifold import TSNE


# PyTorch and PyTorch Geometric Imports
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GATConv

# NetworkX Import for Graph Manipulation
import networkx as nx

# Progress Bar Import
from tqdm import tqdm

# Set the theme for seaborn plots
sns.set_theme()

# Convert GloVe format to Word2Vec format
glove2word2vec('glove.twitter.27B/glove.twitter.27B.200d.txt', 'tmpfile_glove')

# Constants
EMBEDDING_SIZE = 200
figure_name = 'figures/documents_ablation_study_visualization_glove_twitter_200_27b.jpg'


# Function to calculate accuracy, precision, and recall
def calculate_accuracy_precision_recall(true_labels, predicted_labels):
    """
    Calculate accuracy, precision, and recall.

    Parameters:
    true_labels (list): List of true labels.
    predicted_labels (list): List of predicted labels.

    Returns:
    tuple: Accuracy, precision, and recall.
    """
    # Calculate and return accuracy, precision, and recall
    return (accuracy_score(true_labels, predicted_labels),
            precision_score(true_labels, predicted_labels),
            recall_score(true_labels, predicted_labels))


# Function to print evaluation results
def print_evaluation_results(results):
    """
    Print the average accuracy, precision, and recall with their standard deviations.

    Parameters:
    results (list of tuples): A list where each tuple contains accuracy, precision, and recall values.
    """
    print('Avg accuracy | Avg precision | Avg recall')
    
    # Calculate averages and standard deviations
    avg_accuracy, avg_precision, avg_recall = np.mean(results, axis=0)
    std_accuracy, std_precision, std_recall = np.std(results, axis=0)
    
    # Print results in a formatted manner
    print(f'{avg_accuracy:.4f}+-{std_accuracy:.4f}, {avg_precision:.4f}+-{std_precision:.4f}, {avg_recall:.4f}+-{std_recall:.4f}')


# Function to generate a random number
def get_random_number():
    """
    Generate a random integer between 0 and 10000.

    Returns:
    int: A random integer.
    """
    return random.randint(0, 10000)

# Generate global random numbers
global_random_number = get_random_number()
global_random_numbers = [get_random_number() for _ in range(10)]

# Load and preprocess the dataset
df = pd.read_csv('datasets/samples.csv')
# Filter to keep only relevant labels (bug == 0 and feature == 1)
df = df[(df['label'] == 0) | (df['label'] == 1)]

# Add a specific contraction fix for a token
contractions.add('__label__', 'REMOVED_TOKEN')

# Apply contraction fixes and remove stopwords
df['title'] = df['title'].apply(contractions.fix).apply(remove_stopwords)
df['body'] = df['body'].apply(contractions.fix).apply(remove_stopwords)

# Load GloVe embeddings
glove_embeddings_model = KeyedVectors.load_word2vec_format('tmpfile_glove')


# Function to get GloVe embedding for a word
def get_word_glove_embedding(word):
    """
    Retrieve the GloVe embedding for a given word.

    Parameters:
    word (str): The word to retrieve the embedding for.

    Returns:
    np.array: The GloVe embedding.
    """
    # Return the embedding if the word exists, otherwise return a zero vector
    if word not in glove_embeddings_model:
        return np.zeros(EMBEDDING_SIZE, dtype='float32')
    return glove_embeddings_model.get_vector(word)

# Set the device to use for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_graph_of_words(text, window_size):
    """
    Create a graph of words from the given text using a sliding window approach.
    
    Parameters:
    text (str): The input text.
    window_size (int): The size of the sliding window.
    
    Returns:
    networkx.Graph: The graph of words.
    """

    # Split the text into words
    text = text.split()
    # Initialize an empty graph
    G = nx.Graph()


    for i, word in enumerate(text):
        
        # Get the word embedding or use a zero vector if the word is not found
        embedding = get_word_glove_embedding(word)
        
        # Add the word as a node with its embedding as an attribute
        G.add_node(word, x=embedding)

        # Connect the word to the next words in the window
        for j in range(i + 1, i + window_size):
            if j < len(text):
                G.add_edge(word, text[j])
    return G

def create_graph_of_words_for_pytorch(text, window_size):
    """
    Convert the graph of words to a PyTorch Geometric graph.
    
    Parameters:
    text (str): The input text.
    window_size (int): The size of the sliding window.
    
    Returns:
    torch_geometric.data.Data: The PyTorch Geometric graph.
    """
    return from_networkx(create_graph_of_words(text, window_size))


def generate_pytorch_geometric_graphs(window_size):
    """
    Generate PyTorch Geometric graphs for all texts in the DataFrame.
    
    Parameters:
    window_size (int): The size of the sliding window.
    
    Returns:
    list: A list of PyTorch Geometric graphs with labels.
    """

    # Initialize an empty list to store the graphs
    pyg_graphs = []

    for s in tqdm(df['body'].values):
        
        # Convert each text to a PyTorch Geometric graph and append to the list
        pyg_graphs.append(create_graph_of_words_for_pytorch(s, window_size))
    
    print('Finished generating graphs...')

    # Add labels to the graphs
    for i, label in enumerate(df['label'].values):
        pyg_graphs[i].y = torch.tensor(label).float()
    
    # Filter out graphs with no nodes
    pyg_graphs = [g for g in pyg_graphs if g.num_nodes != 0]
    
    return pyg_graphs


class GATClassifier(torch.nn.Module):
    def __init__(self):
        """
        Initialize the GATClassifier model.
        """

        super().__init__()

        # Define the first GAT layer with input features of dimensionality EMBEDDING_SIZE, 10 output features, and 3 heads        
        self.conv1 = GATConv(EMBEDDING_SIZE, 10, heads=5)
        # Define a linear layer to reduce the output to 1 feature
        self.linear1 = torch.nn.Linear(10*5, 1)
        # Define the sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data, batch):
        """
        Define the forward pass of the GATClassifier model.
        
        Parameters:
        data (torch_geometric.data.Data): The input data.
        batch (torch.Tensor): The batch tensor.
        
        Returns:
        tuple: Hidden representations and output predictions.
        """
        # Get the node features and edge indices from the data
        x, edge_index = data.x, data.edge_index

        # Apply the first GAT layer with ELU activation
        x = F.elu(self.conv1(x, edge_index))
        # Perform global mean pooling to get a graph-level representation
        x = global_mean_pool(x, batch)
        # Apply dropout with a probability of 0.5 during training
        h = F.dropout(x, p=0.5, training=self.training)
        # Pass the pooled representation through the linear layer
        x = self.linear1(h)
        # Apply the sigmoid activation function
        x = self.sigmoid(x)
               
        return h, x


def run_gat_classifier(train_pyg_graphs, test_pyg_graphs, train_batch_size=300, learning_rate=0.001, num_epoch=10):
    """
    Train and evaluate the GATClassifier model on the dataset.
    
    Parameters:
    train_pyg_graphs (list): List of PyTorch Geometric graphs for training.
    test_pyg_graphs (list): List of PyTorch Geometric graphs for testing.
    train_batch_size (int): Batch size for training.
    learning_rate (float): Learning rate for the optimizer.
    num_epoch (int): Number of epochs for training.
    
    Returns:
    dict: A dictionary containing the trained model and evaluation results.
    """

    # Create DataLoader for training and testing datasets
    train_loader = DataLoader(train_pyg_graphs, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_pyg_graphs, batch_size=200, shuffle=False)
    
    # Initialize the GATClassifier model and move it to the selected device
    gat_model = GATClassifier().to(device)
    print(gat_model)

    # Define the loss function and optimizer
    loss_function = F.binary_cross_entropy
    optimizer = torch.optim.Adam(gat_model.parameters(), lr=learning_rate)
    
    # Set the model to training mode
    gat_model.train()
    for epoch in range(0, num_epoch):

        # Iterate in batches over the training dataset
        for i, data in enumerate(train_loader):
            data = data.to(device)
            try:
                # Perform a single forward pass
                _, out = gat_model(data, data.batch)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                print(data)
                print(data.x)
                print(data.y)
            
            out = out.squeeze()
            y = data.y.squeeze()
            
            # Compute the loss            
            loss = loss_function(out, y)
            # Perform backward pass            
            loss.backward()
            # Update parameters based on gradients
            optimizer.step()
            # Clear gradients
            optimizer.zero_grad()  # Clear gradients.
        
        # Print the loss for each epoch
        print(f'Epoch: {epoch}, Epoch loss {loss.item()}')

    # Training process is complete
    print('Training process has finished.')
    print('Final loss', loss.item())
    
    # Initialize lists to store true and predicted labels
    true_labels = []
    pred_labels = []

    # Set the model to evaluation mode
    with torch.no_grad():
        gat_model.eval()

        # Iterate in batches over the testing dataset
        for i, data in enumerate(test_loader):
            data = data.to(device)
            # Perform a forward pass
            _, out = gat_model(data, data.batch)
            # Store the predicted and true labels
            pred_labels.extend(torch.round(out.squeeze()).tolist())
            true_labels.extend(data.y.tolist())

    # Calculate accuracy, precision, and recall
    results = calculate_accuracy_precision_recall(true_labels, pred_labels)

    # Print and return the evaluation results    
    print(results)
    return {
        'model': gat_model,
        'results': results
    }


def run_document_visualization_experiment():
    """
    Run the document visualization experiment.

    Returns:
    tuple: Document embeddings and predicted labels.
    """

    # Set the sliding window size for graph construction
    sw = 7
    print('Window size:', sw)

    # Generate PyTorch Geometric graphs using the specified window size
    pytorch_geometric_graphs = generate_pytorch_geometric_graphs(window_size=sw)

    # Split the dataset into training and testing sets (67% train, 33% test)
    train_pyg_graphs, test_pyg_graphs = train_test_split(pytorch_geometric_graphs, test_size=0.33, random_state=42)

    # Train the GAT classifier on the training data
    model = run_gat_classifier(train_pyg_graphs, test_pyg_graphs)['model']

    # Create a DataLoader for the entire dataset to extract embeddings
    loader = DataLoader(pytorch_geometric_graphs, batch_size=100, shuffle=False)
    # List to store document embeddings
    documents_embeddings = []
    # List to store predicted labels   
    pred_labels = []

    # Extract embeddings and predictions using the trained model
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        for i, data in enumerate(loader):
            data = data.to(device)
            embeddings, out = model(data, data.batch)
            documents_embeddings.extend(embeddings.tolist())
            pred_labels.extend(torch.round(out.squeeze()).tolist())
    
    # Print the number of documents and dimensions of each embedding
    print('Number of documents:', len(documents_embeddings))
    print('Number of dimensions per document:', len(documents_embeddings[0]))

    # Return the document embeddings and predicted labels
    return documents_embeddings, pred_labels

# Run the document visualization experiment
documents_embeddings, pred_labels = run_document_visualization_experiment()

# Convert document embeddings to numpy array
documents_embeddings = np.array(documents_embeddings)

# Perform t-SNE for dimensionality reduction
visualization_x_y = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42).fit_transform(np.array(documents_embeddings))

# Plot the results
fig = sns.scatterplot(x=visualization_x_y[:, 0], y=visualization_x_y[:, 1], hue=pred_labels)
fig.set_xlabel('')
fig.set_ylabel('')
fig.get_figure().savefig(figure_name, dpi=500)