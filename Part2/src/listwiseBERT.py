import pandas as pd
# set random seeds
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, BertTokenizer
from transformers import DistilBertModel, DistilBertTokenizer

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Step 1: Parse Documents
def load_documents(filepath):
    doc_df = pd.read_csv(filepath, sep='\t', names=['ID', 'URL', 'TITLE', 'ABSTRACT'], header=None)
    documents = {row['ID']: (row['TITLE'], row['ABSTRACT']) for _, row in doc_df.iterrows()}
    return documents

documents = load_documents('../nfcorpus/raw/doc_dump.txt')

# Step 2: Load Queries
def load_queries(filepath):
    queries_df = pd.read_csv(filepath, sep='\t', names=['QUERY_ID', 'QUERY_TEXT'], header=None)
    return queries_df

train_queries = load_queries('../nfcorpus/train.titles.queries')
test_queries = load_queries('../nfcorpus/test.titles.queries')

# Step 3: Load Relevance Judgments
def load_qrels(filepath):
    qrels_df = pd.read_csv(filepath, sep='\t', names=['QUERY_ID', 'IGNORE', 'DOC_ID', 'RELEVANCE_LEVEL'], header=None)
    relevance = {}
    for _, row in qrels_df.iterrows():
        if row['QUERY_ID'] not in relevance:
            relevance[row['QUERY_ID']] = []
        relevance[row['QUERY_ID']].append((row['DOC_ID'], row['RELEVANCE_LEVEL']))
    return relevance

relevance = load_qrels('../nfcorpus/merged.qrel')

# Step 4: Combine into a usable format
def prepare_data(queries, documents, relevance):
    data = []
    for _, row in queries.iterrows():
        query_id = row['QUERY_ID']
        query_text = row['QUERY_TEXT']
        if query_id in relevance:
            doc_texts = []
            rel_scores = []
            for doc_id, rel_score in relevance[query_id]:
                if doc_id in documents:
                    title, abstract = documents[doc_id]
                    doc_texts.append((title, abstract))
                    rel_scores.append(rel_score)
            data.append((query_text, doc_texts, rel_scores))
    return data

train_data = prepare_data(train_queries, documents, relevance)
test_data = prepare_data(test_queries, documents, relevance)

# Example of data structure
print("Sample data format:", train_data[0])
print(len(train_data))


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'Using device: {device}')

class ListNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ListNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ListwiseDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
        # Device setup
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        
        # Initialize model and move it to the specified device
        # self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.bert_model = BertModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4').to(self.device)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query_text, docs, rel_scores = self.data[idx]
        inputs = self.tokenizer([title + " " + abstract for title, abstract in docs], padding=True, truncation=True, return_tensors="pt", max_length=256)
        
        # Move input tensors to the same device as the model
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
        # Use the mean of the last hidden states as document embeddings
        docs_tensor = outputs.last_hidden_state.mean(dim=1)
        rel_scores_tensor = torch.tensor(rel_scores, dtype=torch.float).to(self.device)
        
        return docs_tensor, rel_scores_tensor


train_dataset = ListwiseDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Batch size of 1 for listwise approach

def listnet_loss(y_i, z_i):
    """
    ListNet loss function.
    y_i: True probabilities (normalized relevance scores)
    z_i: Predicted scores by the model
    """
    P_y_i = torch.softmax(y_i, dim=0)
    P_z_i = torch.softmax(z_i, dim=0)
    return -torch.sum(P_y_i * torch.log(P_z_i))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
model = ListNet(input_size=256, hidden_size=128).to(device) 
optimizer = optim.Adam(model.parameters(), lr=0.0005)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for documents, scores in train_loader:
        documents, scores = documents.to(device), scores.to(device)
        optimizer.zero_grad()
        output = model(documents.squeeze(0))  # squeeze because each batch is size 1
        loss = listnet_loss(scores, output.squeeze())
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Step Loss: {loss.item()}')  # Print loss for each batch
    print(f'Epoch {epoch+1} completed.') 


import numpy as np

def dcg_at_k(scores, k=5):
    scores = np.array(scores)[:k]
    if len(scores) == 0:
        return 0  # If no scores, DCG is zero
    return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

def ndcg_at_k(true_scores, pred_scores, k=5):
    if len(pred_scores) == 0:
        return 0  # Handle case where there are no predictions
    
    # Adjust k if there are fewer scores than k
    k = min(k, len(pred_scores), len(true_scores))
    if k == 0:
        return 0

    order = np.argsort(pred_scores)[::-1]
    true_scores_sorted = np.array(true_scores)[order]
    
    ideal_order = np.argsort(true_scores)[::-1]
    ideal_scores_sorted = np.array(true_scores)[ideal_order]

    dcg = dcg_at_k(true_scores_sorted, k)
    idcg = dcg_at_k(ideal_scores_sorted, k)

    return (dcg / idcg) if idcg > 0 else 0

def evaluate_model(model, data_loader, device):
    model.eval()
    ndcg_scores = []
    with torch.no_grad():
        for documents, scores in data_loader:
            if documents.size(0) == 0:
                continue  # Skip this batch if there are no documents
            documents, scores = documents.to(device), scores.to(device)
            pred_scores = model(documents.squeeze(0))
            pred_scores = pred_scores.squeeze().detach().cpu().numpy() 

            # Ensure scores are in a list format even if there's only one score
            true_scores = scores.squeeze().detach().cpu().numpy()
            true_scores = true_scores if true_scores.ndim > 0 else [true_scores.item()]  # Handle single float scenario
            pred_scores = pred_scores if pred_scores.ndim > 0 else [pred_scores.item()]

            if len(pred_scores) != len(true_scores):
                print(f"Skipping mismatched lengths. True: {len(true_scores)}, Pred: {len(pred_scores)}")
                continue

            try:
                ndcg_score = ndcg_at_k(true_scores, pred_scores, k=5)
                ndcg_scores.append(ndcg_score)
            except Exception as e:
                print(f"Error processing nDCG: {e}")
                print(f"True Scores: {true_scores}, Pred Scores: {pred_scores}")

    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
    return mean_ndcg


test_dataset = ListwiseDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Non-shuffled for evaluation

# Evaluate the model
mean_ndcg = evaluate_model(model, test_loader, device)
print(f'Mean nDCG score on the test set: {mean_ndcg}')
