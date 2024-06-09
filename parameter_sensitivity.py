import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import remove_stopwords
import contractions
from sklearn.model_selection import train_test_split
import numpy as np
import fasttext
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import torch.nn.functional as F
import torch
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv
from torch_geometric.nn import SGConv
from tqdm import tqdm
import random
sns.set_theme()


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

x_train, x_test = train_test_split(df.body, test_size=0.33, random_state=42)
print('Train size', len(x_train))
print('Test size', len(x_test))

df['fasttext_input'] = '__label__' + df['label'].map(str) + ' ' + df['title'] + ' ' + df['body']
train_input, test_input = train_test_split(df.fasttext_input.values, test_size=0.33, random_state=42)
np.savetxt('train.txt', train_input, fmt='%s')
np.savetxt('test.txt', test_input, fmt='%s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def create_graph_of_words(text, window_size, embeddings_lookup, embedding_dim):
    text = text.split()
    G = nx.Graph()
    for i, word in enumerate(text):
        #embedding = fasttext_model.get_word_vector(word)
        embedding = embeddings_lookup.get(word, np.zeros(embedding_dim, dtype='float32'))
        G.add_node(word, x=embedding)
        for j in range(i + 1, i + window_size):
            if j < len(text):
                G.add_edge(word, text[j])
    return G

def create_graph_of_words_for_pytorch(text, window_size, embeddings_lookup, embedding_dim):
    return from_networkx(create_graph_of_words(text, window_size, embeddings_lookup, embedding_dim))

def generate_pytorch_geometric_graphs(window_size, embeddings_lookup, embedding_dim):
    pyg_graphs = []
    for s in tqdm(df['body'].values):
        pyg_graphs.append(create_graph_of_words_for_pytorch(s, window_size, embeddings_lookup, embedding_dim))
    print('finished...')
    for i, label in enumerate(df['label'].values):
        pyg_graphs[i].y = torch.tensor(label).float()
    
    pyg_graphs = [g for g in pyg_graphs if g.num_nodes != 0]
    return pyg_graphs

class GATClassifier(torch.nn.Module):
    def __init__(self, input_dim, attention_heads=5, gat_layer_output_size=10):
        super().__init__()
        #torch.manual_seed(12345)
        
        self.conv1 = GATConv(input_dim, gat_layer_output_size, heads=attention_heads)
        #self.conv1 = SGConv(100, 50, K=1)
        self.linear1 = torch.nn.Linear(gat_layer_output_size*attention_heads, 1)
        
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        
        x = F.elu(self.conv1(x, edge_index))
        x = global_mean_pool(x, batch)
        h = F.dropout(x, p=0.5, training=self.training)
        x = self.linear1(h)
        x = self.sigmoid(x)
        
        return h, x

def run_gat_classifier(train_pyg_graphs, test_pyg_graphs, train_batch_size=300, learning_rate=0.001, num_epoch=10, input_dim=100, attention_heads=5, gat_layer_output_size=10):
    train_loader = DataLoader(train_pyg_graphs, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_pyg_graphs, batch_size=200, shuffle=False)
    
    gat_model = GATClassifier(input_dim, attention_heads=attention_heads, gat_layer_output_size=gat_layer_output_size).to(device)
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

    results = calculate_accuracy_precision_recall(true_labels, pred_labels)
    
    print(results)
    return {
        'model': gat_model,
        'results': results
    }

torch.manual_seed(12345)

#### 1. Sensitivity analysis for the embedding size of the fastText embeddings. ####

def run_embedding_dimensionality_experiments():
    dims = [32, 64, 100, 128, 150]
    
    dim_results = []
    for dim in dims:
        # Train fasttext embeddings
        fasttext_model = fasttext.train_supervised('train.txt', dim=dim, epoch=5)
        fasttext_model.test('test.txt')
        embeddings_lookup = {word: fasttext_model.get_word_vector(word) for word in fasttext_model.get_words()}
        
        print('Dim:', dim)
        pytorch_geometric_graphs = generate_pytorch_geometric_graphs(window_size=7, embeddings_lookup=embeddings_lookup, embedding_dim=dim)
        train_pyg_graphs, test_pyg_graphs = train_test_split(pytorch_geometric_graphs, test_size=0.33, random_state=42)
        dim_results.append((dim, run_gat_classifier(train_pyg_graphs, test_pyg_graphs, input_dim=dim)['results']))
    return dim_results

dim_results = run_embedding_dimensionality_experiments()
dim_results = [[dim, results[0]] for dim, results in dim_results]

plot_x = [res[0] for res in dim_results]
plot_y = [res[1] for res in dim_results]
print(dim_results)

fig = sns.lineplot(x=plot_x, y=plot_y)
sns.scatterplot(x=plot_x, y=plot_y)
fig.set_xlabel('embedding size')
fig.set_ylabel('accuracy')
fig.get_figure().savefig('figures/plot_embedding_dimensionality.pdf', dpi=500)

#### Calcualte default fastText model and training graphs.

fasttext_model = fasttext.train_supervised('train.txt', dim=100, epoch=5)
fasttext_model.test('test.txt')
df.drop('fasttext_input', axis=1, inplace=True)
embeddings_lookup = {word: fasttext_model.get_word_vector(word) for word in fasttext_model.get_words()}

pytorch_geometric_graphs = generate_pytorch_geometric_graphs(window_size=7, embeddings_lookup=embeddings_lookup, embedding_dim=100)
train_pyg_graphs, test_pyg_graphs = train_test_split(pytorch_geometric_graphs, test_size=0.33, random_state=42)

#### 2. Sensitivity analysis for the number of training epochs. ####

def run_training_epochs_experiments():
    num_epochs = [5, 10, 20, 30, 50, 100, 200]

    pytorch_geometric_graphs = generate_pytorch_geometric_graphs(window_size=7, embeddings_lookup=embeddings_lookup, embedding_dim=100)
    train_pyg_graphs, test_pyg_graphs = train_test_split(pytorch_geometric_graphs, test_size=0.33, random_state=42)    
    epoch_results = []
    for epochs in num_epochs:
        print('Epochs:', epochs)
        epoch_results.append((epochs, run_gat_classifier(train_pyg_graphs, test_pyg_graphs, num_epoch=epochs, input_dim=100)['results']))
    return epoch_results

epoch_results = run_training_epochs_experiments()
epoch_results = [[epochs, results[0]] for epochs, results in epoch_results]

plot_x = [res[0] for res in epoch_results]
plot_y = [res[1] for res in epoch_results]
print(epoch_results)

plt.clf()
fig = sns.lineplot(x=plot_x, y=plot_y)
sns.scatterplot(x=plot_x, y=plot_y)
fig.set_xlabel('epochs')
fig.set_ylabel('accuracy')
fig.get_figure().savefig('figures/plot_epochs.pdf', dpi=500)


#### 3. Sensitivity analysis for the number of attention heads. ####

def run_attention_heads_experiments():
    attention_heads = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]

    pytorch_geometric_graphs = generate_pytorch_geometric_graphs(window_size=7, embeddings_lookup=embeddings_lookup, embedding_dim=100)
    train_pyg_graphs, test_pyg_graphs = train_test_split(pytorch_geometric_graphs, test_size=0.33, random_state=42)
    attention_results = []
    for ah in attention_heads:
        print('Attention heads:', ah)
        attention_results.append((ah, run_gat_classifier(train_pyg_graphs, test_pyg_graphs, num_epoch=10, input_dim=100, attention_heads=ah)['results']))
    return attention_results

attention_results = run_attention_heads_experiments()
attention_results = [[ah, results[0]] for ah, results in attention_results]

plot_x = [res[0] for res in attention_results]
plot_y = [res[1] for res in attention_results]
print(attention_results)

plt.clf()
fig = sns.lineplot(x=plot_x, y=plot_y)
sns.scatterplot(x=plot_x, y=plot_y) 
fig.set_xlabel('number of attention heads')
fig.set_ylabel('accuracy')
fig.get_figure().savefig('figures/plot_attention.pdf', dpi=500)


#### 4. Sensitivity analysis for the GAT attention layer size. ####

def run_attention_layer_sizes_experiments():
    attention_layer_sizes = [5, 10, 15, 20, 25, 50, 60, 100, 150]
    pytorch_geometric_graphs = generate_pytorch_geometric_graphs(window_size=7, embeddings_lookup=embeddings_lookup, embedding_dim=100)
    train_pyg_graphs, test_pyg_graphs = train_test_split(pytorch_geometric_graphs, test_size=0.33, random_state=42)

    gat_layer_size_results = []
    for gs in attention_layer_sizes:
        print('GAT layer output size:', gs)
        gat_layer_size_results.append((gs, run_gat_classifier(train_pyg_graphs, test_pyg_graphs, num_epoch=10, input_dim=100, attention_heads=5, gat_layer_output_size=gs)['results']))
    return gat_layer_size_results

gat_layer_size_results = run_attention_layer_sizes_experiments()
gat_layer_size_results = [[gs, results[0]] for gs, results in gat_layer_size_results]

plot_x = [res[0] for res in gat_layer_size_results]
plot_y = [res[1] for res in gat_layer_size_results]
print(gat_layer_size_results)

plt.clf()
fig = sns.lineplot(x=plot_x, y=plot_y)
sns.scatterplot(x=plot_x, y=plot_y) 
fig.set_xlabel('size of the GAT layer')
fig.set_ylabel('accuracy')
fig.get_figure().savefig('figures/plot_size_of_the_gat.pdf', dpi=500)