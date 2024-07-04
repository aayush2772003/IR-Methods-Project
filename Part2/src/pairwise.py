import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example of loading and preparing data
# You'll replace this with actual loading code
queries = pd.read_csv("../nfcorpus/train.queries.ids", sep="\t", header=None, names=["QUERY_ID", "QUERY_TEXT"])
docs = pd.read_csv("../nfcorpus/raw/doc_dump.txt", sep="\t", header=None, names=["DOC_ID", "URL", "TITLE", "ABSTRACT"])
qrels = pd.read_csv("../nfcorpus/merged.qrel", sep="\t", header=None, names=["QUERY_ID", "ZERO", "DOC_ID", "RELEVANCE_LEVEL"])

# Join the dataframes to create a single dataframe containing all relevant information
data = queries.merge(qrels, on="QUERY_ID").merge(docs, on="DOC_ID")

# Feature extraction for documents using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
doc_features = tfidf_vectorizer.fit_transform(data['ABSTRACT'])

# Create pairs of documents for each query
# Simplistic pairing approach
# More efficient and complex strategies might be needed for larger datasets
# Function to generate pairs with sampling
# Function to generate sampled pairs with correct indexing
# Generate a limited number of pairs for demonstration
def generate_limited_pairs(data, num_pairs=10):
    pair_data = []
    labels = []

    for query_id in data['QUERY_ID'].unique():
        sub_data = data[data['QUERY_ID'] == query_id]
        if len(sub_data) > 1:  # Ensure there are at least two documents to compare
            for _ in range(min(num_pairs, len(sub_data)*(len(sub_data)-1)//2)):  # Limit the number of pairs per query
                i, j = np.random.choice(len(sub_data), 2, replace=False)  # Select two random docs without replacement
                if sub_data.iloc[i]['RELEVANCE_LEVEL'] > sub_data.iloc[j]['RELEVANCE_LEVEL']:
                    pair_data.append((i, j))
                    labels.append(1)
                else:
                    pair_data.append((j, i))
                    labels.append(0)

    return pair_data, labels

# Generate limited pairs
pair_data, labels = generate_limited_pairs(data, num_pairs=5)  # Adjust number of pairs as needed

# Convert pairs to feature vectors
pair_features = np.array([np.abs(doc_features[i].toarray() - doc_features[j].toarray()).flatten() for i, j in pair_data])

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(pair_features, labels, test_size=0.2, random_state=42)

# Train the model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the model:", accuracy)

import numpy as np
from sklearn.metrics import ndcg_score

# Function to calculate NDCG
def calculate_ndcg(y_true, scores, k=None):
    """
    Calculate the NDCG score for given true labels and predicted scores.
    :param y_true: The true relevance labels.
    :param scores: The predicted scores or relevance values.
    :param k: The number of top-k items to consider in the ranking.
    :return: The NDCG score.
    """
    # Ensuring the true labels and scores are numpy arrays
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    # Sorting indices by scores in descending order
    ranked_indices = np.argsort(scores)[::-1]

    # Getting the top-k indices, if k is specified
    if k is not None:
        ranked_indices = ranked_indices[:k]

    # Calculate DCG@k
    gains = 2 ** y_true[ranked_indices] - 1  # Transform relevance into gains
    discounts = np.log2(np.arange(len(ranked_indices)) + 2)
    dcg = np.sum(gains / discounts)

    # Calculate IDCG@k (ideal DCG, maximum possible DCG)
    ideal_indices = np.argsort(y_true)[::-1]
    if k is not None:
        ideal_indices = ideal_indices[:k]
    ideal_gains = 2 ** y_true[ideal_indices] - 1
    idcg = np.sum(ideal_gains / discounts)

    # Handle the case where IDCG is zero (no relevant documents)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg

# Example usage in your testing pipeline
# Assume y_test contains the true relevance labels, and predictions are the scores from your model
ndcg_value = calculate_ndcg(y_test, predictions, k=10)  # Evaluate NDCG@10
print("NDCG@10 Score:", ndcg_value)