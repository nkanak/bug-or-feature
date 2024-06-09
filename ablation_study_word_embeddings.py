import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import remove_stopwords
import contractions
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import torch.nn.functional as F
import torch

import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from torch_geometric.nn import GATConv

from tqdm import tqdm

import random

sns.set_theme()

glove2word2vec('glove.twitter.27B/glove.twitter.27B.200d.txt', 'tmpfile_glove')
EMBEDDING_SIZE = 200
figure_name = 'figures/documents_ablation_study_visualization_glove_twitter_200_27b.jpg'

# Set fixed random number seed
#torch.manual_seed(42)

def calculate_accuracy_precision_recall(true_labels, predicted_labels):
    return (accuracy_score(true_labels, predicted_labels),
           precision_score(true_labels, predicted_labels),
           recall_score(true_labels, predicted_labels))

def print_evaluation_results(results):
    print('Avg accuracy | Avg precision | Avg recall')
    avg_accuracy, avg_precision, avg_recall = np.mean(results, axis=0)
    std_accuracy, std_precision, std_recall = np.std(results, axis=0)
    print(f'{avg_accuracy:.4f}+-{std_accuracy:.4f}, {avg_precision:.4f}+-{std_precision:.4f}, {avg_recall:.4f}+-{std_recall:.4f}')

def get_random_number():
    return random.randint(0, 10000)

global_random_number = get_random_number()
global_random_numbers = [get_random_number() for _ in range(10)]

df = pd.read_csv('samples.csv')
# bug == 0 and feature == 1
df = df[(df['label'] == 0) | (df['label'] == 1)]
#df = df[:500]
len(df)

contractions.add('__label__', 'REMOVED_TOKEN')
# fix contractions
df['title'] = df['title'].apply(contractions.fix)
df['body'] = df['body'].apply(contractions.fix)

# removal of stopwords
df['title'] = df['title'].apply(remove_stopwords)
df['body'] = df['body'].apply(remove_stopwords)

glove_embeddings_model = KeyedVectors.load_word2vec_format('tmpfile_glove')

def get_word_glove_embedding(word):
    if word not in glove_embeddings_model:
        return np.zeros(EMBEDDING_SIZE, dtype='float32')
    return glove_embeddings_model.get_vector(word)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def create_graph_of_words(text, window_size):
    text = text.split()
    G = nx.Graph()
    for i, word in enumerate(text):
        #embedding = fasttext_model.get_word_vector(word)
        #embedding = embeddings_lookup.get(word, np.zeros(100, dtype='float32'))
        embedding = get_word_glove_embedding(word)
        G.add_node(word, x=embedding)
        for j in range(i + 1, i + window_size):
            if j < len(text):
                G.add_edge(word, text[j])
    return G

def create_graph_of_words_for_pytorch(text, window_size):
    return from_networkx(create_graph_of_words(text, window_size))

def generate_pytorch_geometric_graphs(window_size):
    pyg_graphs = []
    for s in tqdm(df['body'].values):
        pyg_graphs.append(create_graph_of_words_for_pytorch(s, window_size))
    print('finished...')
    for i, label in enumerate(df['label'].values):
        pyg_graphs[i].y = torch.tensor(label).float()
    
    pyg_graphs = [g for g in pyg_graphs if g.num_nodes != 0]
    return pyg_graphs

class GATClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #torch.manual_seed(12345)
        
        self.conv1 = GATConv(EMBEDDING_SIZE, 10, heads=5)
        #self.conv1 = SGConv(100, 50, K=1)
        self.linear1 = torch.nn.Linear(10*5, 1)
        
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        
        x = F.elu(self.conv1(x, edge_index))
        x = global_mean_pool(x, batch)
        h = F.dropout(x, p=0.5, training=self.training)
        x = self.linear1(h)
        x = self.sigmoid(x)
        
        return h, x

def run_gat_classifier(train_pyg_graphs, test_pyg_graphs, train_batch_size=300, learning_rate=0.001, num_epoch=10):
    train_loader = DataLoader(train_pyg_graphs, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_pyg_graphs, batch_size=200, shuffle=False)
    
    gat_model = GATClassifier().to(device)
    print(gat_model)
    # Define the loss function and optimizer
    loss_function = F.binary_cross_entropy
    optimizer = torch.optim.Adam(gat_model.parameters(), lr=learning_rate)

    gat_model.train()
    for epoch in range(0, num_epoch):
        for i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
            data = data.to(device)
            try:
                _, out = gat_model(data, data.batch)  # Perform a single forward pass.
            except Exception as e:
                print(data)
                print(data.x)
                print(data.y)
            out = out.squeeze()
            y = data.y.squeeze()
            loss = loss_function(out, y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        print(f'Epoch: {epoch}, Epoch loss {loss.item()}')

    print('Training process has finished.')
    print('Final loss', loss.item())
    
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        gat_model.eval()
        for i, data in enumerate(test_loader):
            data = data.to(device)
            _, out = gat_model(data, data.batch)
            pred_labels.extend(torch.round(out.squeeze()).tolist())
            true_labels.extend(data.y.tolist())
            
    #print('true labels ----')
    #print(true_labels)
    #print('pred labels ----')
    #print(pred_labels)
    
    results = calculate_accuracy_precision_recall(true_labels, pred_labels)
    
    print(results)
    return {
        'model': gat_model,
        'results': results
    }

from sklearn.manifold import TSNE
def run_document_visualization_experiment():
    sw = 7
    print('Window size:', sw)
    pytorch_geometric_graphs = generate_pytorch_geometric_graphs(window_size=sw)
    train_pyg_graphs, test_pyg_graphs = train_test_split(pytorch_geometric_graphs, test_size=0.33, random_state=42)
    model = run_gat_classifier(train_pyg_graphs, test_pyg_graphs)['model']
    
    loader = DataLoader(pytorch_geometric_graphs, batch_size=100, shuffle=False)
    documents_embeddings = []
    
    pred_labels = []
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(loader):
            data = data.to(device)
            embeddings, out = model(data, data.batch)
            documents_embeddings.extend(embeddings.tolist())
            pred_labels.extend(torch.round(out.squeeze()).tolist())
    
    print('Number of documents:', len(documents_embeddings))
    print('Number of dimensions per document:', len(documents_embeddings[0]))
    return documents_embeddings, pred_labels

documents_embeddings, pred_labels = run_document_visualization_experiment()
documents_embeddings = np.array(documents_embeddings)
visualization_x_y = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42).fit_transform(np.array(documents_embeddings))

fig = sns.scatterplot(x=visualization_x_y[:, 0], y=visualization_x_y[:, 1], hue=pred_labels)
fig.set_xlabel('')
fig.set_ylabel('')
fig.get_figure().savefig(figure_name, dpi=500)