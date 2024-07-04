import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_documents(file_path):
    docs = []
    doc_ids = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')  # assuming tab-separated values
            if len(parts) < 4:
                continue  # skip incomplete lines
            doc_id, url, title, abstract = parts[0], parts[1], parts[2], parts[3]
            doc_ids.append(doc_id)
            # Combine title and abstract for content
            docs.append(title + " " + abstract)
    return doc_ids, docs

def rocchio(original_query_vec, docs_vectors, top_n_indices, alpha=1.0, beta=0.73):
    relevant_vecs = docs_vectors[top_n_indices]
    mean_relevant_vec = np.mean(relevant_vecs, axis=0)
    new_query_vec = alpha * original_query_vec + beta * mean_relevant_vec
    return np.asarray(new_query_vec)  # Ensure output is ndarray

# Load documents
doc_ids, documents = load_documents('../nfcorpus/raw/doc_dump.txt')

# Initialize vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
doc_vectors = vectorizer.fit_transform(documents)

# # Example initial query
# query = "effects of vitamin C on health"
# query_vec = vectorizer.transform([query])

# # Perform initial retrieval (find top 3 documents)
# cos_similarities = cosine_similarity(query_vec, doc_vectors).flatten()
# top_n = 3
# top_n_indices = np.argsort(cos_similarities)[-top_n:][::-1]

# # After applying Rocchio algorithm to refine the query
# new_query_vec = rocchio(query_vec, doc_vectors, top_n_indices)

# # Ensure the new query vector is an ndarray before passing to cosine_similarity
# new_query_vec = np.asarray(new_query_vec)

# # Re-run the query with the refined vector
# new_cos_similarities = cosine_similarity(new_query_vec, doc_vectors).flatten()
# new_top_indices = np.argsort(new_cos_similarities)[-top_n:][::-1]

# # Output results
# print("Top-3 documents after query refinement:")
# for idx in new_top_indices:
#     print(doc_ids[idx])

def load_test_queries(file_path):
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            queries[query_id] = query_text
    return queries

test_queries = load_test_queries('../nfcorpus/test.titles.queries')

def load_relevance_judgments(file_path):
    relevance = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            query_id, _, doc_id, relevance_score = parts
            if query_id not in relevance:
                relevance[query_id] = []
            relevance[query_id].append((doc_id, int(relevance_score)))
    return relevance

relevance_judgments = load_relevance_judgments('../nfcorpus/merged.qrel')

def precision_at_k(retrieved_docs, relevant_docs, k=10):
    relevant_set = set([doc_id for doc_id, _ in relevant_docs[:k]])
    retrieved_set = set(retrieved_docs[:k])
    relevant_retrieved = relevant_set.intersection(retrieved_set)
    return len(relevant_retrieved) / k

results_before = {}
results_after = {}

for query_id, query_text in test_queries.items():
    query_vec = vectorizer.transform([query_text])
    initial_similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    top_n_indices = np.argsort(initial_similarities)[-3:][::-1]
    
    relevant_docs = relevance_judgments.get(query_id, [])
    results_before[query_id] = precision_at_k([doc_ids[idx] for idx in top_n_indices], relevant_docs)

    new_query_vec = rocchio(query_vec, doc_vectors, top_n_indices)
    new_similarities = cosine_similarity(new_query_vec, doc_vectors).flatten()
    new_top_indices = np.argsort(new_similarities)[-10:][::-1]
    
    results_after[query_id] = precision_at_k([doc_ids[idx] for idx in new_top_indices], relevant_docs)

# Print results to compare
for query_id in test_queries:
    print(f"Query ID: {query_id}, Precision before: {results_before[query_id]}, Precision after: {results_after[query_id]}")

import numpy as np

def dcg_at_k(scores, k=10):
    """Calculate DCG for the given scores up to position k."""
    scores = np.asfarray(scores)[:k]
    if scores.size:
        return scores[0] + np.sum(scores[1:] / np.log2(np.arange(2, scores.size + 1)))
    return 0.0

def ndcg_at_k(scores, ideal_scores, k=10):
    """Calculate NDCG for the given scores and ideal scores up to position k."""
    actual_dcg = dcg_at_k(scores, k)
    ideal_dcg = dcg_at_k(ideal_scores, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

def get_scores(retrieved_docs, relevant_docs):
    """Return the relevance scores for the retrieved documents."""
    doc_to_score = {doc_id: score for doc_id, score in relevant_docs}
    scores = [doc_to_score.get(doc, 0) for doc in retrieved_docs]
    return scores

results_before_ndcg = {}
results_after_ndcg = {}
sum_ndcg_before = 0
sum_ndcg_after = 0
count_queries_with_scores = 0

for query_id, query_text in test_queries.items():
    query_vec = vectorizer.transform([query_text])
    initial_similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    top_n_indices = np.argsort(initial_similarities)[-3:][::-1]
    
    relevant_docs = relevance_judgments.get(query_id, [])
    retrieved_docs_before = [doc_ids[idx] for idx in top_n_indices]
    scores_before = get_scores(retrieved_docs_before, relevant_docs)
    ideal_scores = sorted([score for _, score in relevant_docs], reverse=True)
    
    results_before_ndcg[query_id] = ndcg_at_k(scores_before, ideal_scores)
    sum_ndcg_before += results_before_ndcg[query_id]

    new_query_vec = rocchio(query_vec, doc_vectors, top_n_indices)
    new_similarities = cosine_similarity(new_query_vec, doc_vectors).flatten()
    new_top_indices = np.argsort(new_similarities)[-3:][::-1]
    
    retrieved_docs_after = [doc_ids[idx] for idx in new_top_indices]
    scores_after = get_scores(retrieved_docs_after, relevant_docs)
    
    results_after_ndcg[query_id] = ndcg_at_k(scores_after, ideal_scores)
    sum_ndcg_after += results_after_ndcg[query_id]

    if ideal_scores:  # Only count queries where NDCG could be calculated (non-empty ideal scores)
        count_queries_with_scores += 1

# Calculate average NDCG
average_ndcg_before = sum_ndcg_before / count_queries_with_scores if count_queries_with_scores else 0
average_ndcg_after = sum_ndcg_after / count_queries_with_scores if count_queries_with_scores else 0

# Print average NDCG results to compare
print(f"Average NDCG before applying Rocchio: {average_ndcg_before:.4f}")
print(f"Average NDCG after applying Rocchio: {average_ndcg_after:.4f}")

