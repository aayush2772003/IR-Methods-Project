import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from rank_bm25 import BM25Okapi
import nltk
import math

nltk.download('punkt')
nltk.download('stopwords')

# Read the document and queries
docs = pd.read_csv('../nfcorpus/raw/doc_dump.txt', sep='\t', names=['ID', 'URL', 'TITLE', 'ABSTRACT'])
queries = pd.read_csv('../nfcorpus/train.titles.queries', sep='\t', names=['QUERY_ID', 'QUERY_TEXT'])

# Set up preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    words = word_tokenize(text.lower())
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalnum()]
    return filtered_words

docs['PROCESSED_TEXT'] = docs['ABSTRACT'].apply(preprocess)
queries['PROCESSED_TEXT'] = queries['QUERY_TEXT'].apply(preprocess)

# Calculate document-term frequencies
from collections import defaultdict

doc_term_freqs = []
total_term_freq = defaultdict(int)

for index, row in docs.iterrows():
    doc_freq = defaultdict(int)
    for word in row['PROCESSED_TEXT']:
        doc_freq[word] += 1
        total_term_freq[word] += 1
    doc_term_freqs.append(doc_freq)
    docs.at[index, 'DOC_LENGTH'] = sum(doc_freq.values())

# Set the smoothing parameter
lambda_ = 0.1
total_terms = sum(total_term_freq.values())

def ptd_jelinek(word, doc_id):
    tf = doc_term_freqs[doc_id].get(word, 0)
    doc_length = docs.at[doc_id, 'DOC_LENGTH']
    if doc_length == 0:
        return lambda_ * (total_term_freq[word] / total_terms)
    return (1 - lambda_) * (tf / doc_length) + lambda_ * (total_term_freq[word] / total_terms)

def score_query_lm(query):
    scores = []
    for index, doc in docs.iterrows():
        log_score = 0
        for word in query:
            probability = ptd_jelinek(word, index)
            log_score += math.log(probability + 1e-10)  # small constant to prevent log(0)
        scores.append(log_score)
    return scores

tokenized_corpus = [doc['PROCESSED_TEXT'] for index, doc in docs.iterrows()]
bm25 = BM25Okapi(tokenized_corpus)

def score_query_bm25(query):
    tokenized_query = preprocess(query)
    scores = bm25.get_scores(tokenized_query)
    return scores

# Read relevance judgments
relevance_judgments = pd.read_csv('../nfcorpus/merged.qrel', sep='\t', names=['QUERY_ID', 'USELESS', 'DOC_ID', 'RELEVANCE_LEVEL'])

def evaluate_model(scores, query_id, top_k=10):
    ranked_docs = np.argsort(scores)[::-1][:top_k]  # Top k scores in descending order
    top_ranked_docs_ids = docs.iloc[ranked_docs]['ID'].values  # Getting the document IDs
    relevant_docs = relevance_judgments[(relevance_judgments['QUERY_ID'] == query_id) & (relevance_judgments['RELEVANCE_LEVEL'] > 0)]['DOC_ID'].values
    true_positives = np.isin(top_ranked_docs_ids, relevant_docs)
    precision = np.mean(true_positives)
    recall = np.sum(true_positives) / len(relevant_docs) if len(relevant_docs) > 0 else 0
    return precision, recall, scores[ranked_docs[:top_k]]

# Testing and evaluating the first 20 queries
results = []
for index, row in queries.head(20).iterrows():
    lm_scores = np.array(score_query_lm(row['PROCESSED_TEXT']))
    bm25_scores = score_query_bm25(row['QUERY_TEXT'])
    
    lm_precision, lm_recall, lm_top_scores = evaluate_model(lm_scores, row['QUERY_ID'])
    bm25_precision, bm25_recall, bm25_top_scores = evaluate_model(bm25_scores, row['QUERY_ID'])
    
    results.append({
        'Query ID': row['QUERY_ID'],
        'LM Precision': lm_precision,
        'LM Recall': lm_recall,
        'LM Top Scores': lm_top_scores,
        'BM25 Precision': bm25_precision,
        'BM25 Recall': bm25_recall,
        'BM25 Top Scores': bm25_top_scores
    })

# Display results
results_df = pd.DataFrame(results)
print(results_df)
