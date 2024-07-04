import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from nltk.stem.porter import PorterStemmer
import os
import json
import math

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Tokenize, normalize, and stem words."""
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lower case
    processed_tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return processed_tokens

def read_document(file_path):
    """Read documents from a file and preprocess them."""
    documents = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue  # Skip malformed lines
            doc_id, url, title, abstract = parts[:4]
            content = ' '.join(parts[2:])  # Consider title and abstract
            documents[doc_id] = preprocess(content)
    return documents

def build_index(documents):
    """Build an inverted index from the processed documents with raw term frequencies."""
    inverted_index = defaultdict(list)
    doc_length = {}

    for doc_id, tokens in documents.items():
        term_freq = Counter(tokens)
        # Store the length of the document in terms of total term count (for possible use in normalization later)
        doc_length[doc_id] = sum(term_freq.values())

        for term, count in term_freq.items():
            inverted_index[term].append((doc_id, count))

    return inverted_index, doc_length

doc_dump_path = '../nfcorpus/raw/doc_dump.txt'
json_index_path = '../inverted_index.json'

def save_index_with_json(index, file_path):
    # Convert tuples in lists to lists for JSON serialization
    json_ready_index = {k: [list(item) for item in v] for k, v in index.items()}
    with open(file_path, 'w') as f:
        json.dump(json_ready_index, f)

def save_index_with_custom_json(index, file_path):
    # Convert tuples in lists to lists for JSON serialization
    json_ready_index = {k: [list(item) for item in v] for k, v in index.items()}
    
    with open(file_path, 'w') as f:
        # Begin the JSON object
        f.write('{\n')
        # Serialize each key-value pair to be on a single line
        entries = []
        for key, value in json_ready_index.items():
            # JSON serialize the key and the value
            json_key = json.dumps(key)
            json_value = json.dumps(value)
            entry = f'  {json_key}: {json_value}'
            entries.append(entry)
        # Join all entries with commas and new lines
        f.write(',\n'.join(entries))
        # End the JSON object
        f.write('\n}\n')

def load_index_with_json(file_path):
    with open(file_path, 'r') as f:
        json_loaded_index = json.load(f)
    # Convert lists back to tuples if necessarynt
    return {k: [tuple(item) for item in v] for k, v in json_loaded_index.items()}

def save_document_count(total_documents, file_path):
    with open(file_path, 'w') as f:
        json.dump(total_documents, f)

# if not os.path.exists(json_index_path):
#     print("Index file not found. Building index from scratch...")
#     doc_documents = read_document(doc_dump_path)
#     total_documents = len(doc_documents)
#     save_document_count(total_documents, 'total_docs.json')
#     inverted_index, doc_lengths = build_index(doc_documents)
#     save_index_with_json(inverted_index, json_index_path)

def load_document_count(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# if os.path.exists(json_index_path):
#     print("Index file found. Loading index...")
#     inverted_index = load_index_with_json(json_index_path)
#     total_documents = load_document_count('total_docs.json')

if os.path.exists(json_index_path):
    print("Index file found. Loading index...")
    inverted_index = load_index_with_json(json_index_path)
    total_documents = load_document_count('../total_docs.json')
    # Load doc lengths
    with open('../doc_lengths.json', 'r') as f:
        doc_lengths = json.load(f)
else:
    print("Index file not found. Building index from scratch...")
    doc_documents = read_document(doc_dump_path)
    total_documents = len(doc_documents)
    save_document_count(total_documents, 'total_docs.json')
    inverted_index, doc_lengths = build_index(doc_documents)
    # save doc lengths
    with open('doc_lengths.json', 'w') as f:
        json.dump(doc_lengths, f)
    save_index_with_json(inverted_index, json_index_path)


# Example of what's in the index
# print(list(inverted_index.items())[:5])

# json_index_path = '../inverted_index.json'
# save_index_with_json(inverted_index, json_index_path)


# json_index_path = 'inverted_index_custom.json'
# save_index_with_custom_json(inverted_index, json_index_path)

# loaded_json_index = load_index_with_json(json_index_path)

def vectorize_nnn(index):
    """Create document vectors using raw term frequencies without normalization."""
    doc_vectors = defaultdict(dict)
    for term, postings in index.items():
        for doc_id, freq in postings:
            doc_vectors[doc_id][term] = freq
    return doc_vectors

def vectorize_ntn(index, doc_lengths):
    """Create document vectors with term frequency normalized by the maximum frequency in the document."""
    doc_vectors = defaultdict(dict)
    # Calculate max frequency per document first
    max_freq_per_doc = defaultdict(int)
    for term, postings in index.items():
        for doc_id, freq in postings:
            if freq > max_freq_per_doc[doc_id]:
                max_freq_per_doc[doc_id] = freq

    # Normalize term frequencies
    for term, postings in index.items():
        for doc_id, freq in postings:
            doc_vectors[doc_id][term] = freq / max_freq_per_doc[doc_id]

    return doc_vectors

def calculate_idf(index, total_documents):
    """Calculate inverse document frequency for each term in the index."""
    return {term: math.log(total_documents / len(postings)) for term, postings in index.items()}

def vectorize_ntc(index, doc_lengths, total_documents):
    """Create document vectors with normalized TF-IDF and cosine normalization."""
    idf = calculate_idf(index, total_documents)
    doc_vectors = defaultdict(dict)
    max_freq_per_doc = defaultdict(int)

    # Calculate maximum frequency for each document
    for term, postings in index.items():
        for doc_id, freq in postings:
            if freq > max_freq_per_doc[doc_id]:
                max_freq_per_doc[doc_id] = freq

    # Calculate TF-IDF for each term and document
    for term, postings in index.items():
        for doc_id, freq in postings:
            if max_freq_per_doc[doc_id] == 0:  # Avoid division by zero
                continue
            normalized_tf = freq / max_freq_per_doc[doc_id]
            tf_idf = normalized_tf * idf[term]
            doc_vectors[doc_id][term] = tf_idf

    # Normalize each document vector to unit length (cosine normalization)
    for doc_id in doc_vectors:
        norm = math.sqrt(sum([value ** 2 for value in doc_vectors[doc_id].values()]))
        if norm == 0:  # Avoid division by zero in case of empty documents
            continue
        for term in doc_vectors[doc_id]:
            doc_vectors[doc_id][term] /= norm

    return doc_vectors



# Calculate the total number of documents (needed for IDF in the ntc model)
doc_documents = read_document(doc_dump_path)
total_documents = len(doc_documents)

# Generate document vectors using nnn model
nnn_vectors = vectorize_nnn(inverted_index)

# Generate document vectors using ntn model
ntn_vectors = vectorize_ntn(inverted_index, doc_lengths)

# Generate document vectors using ntc model
ntc_vectors = vectorize_ntc(inverted_index, doc_lengths, total_documents)

# Printing example vectors to see the format
print("Example nnn vector for one document:")
example_doc_id = next(iter(nnn_vectors))  # Get the first doc ID from the dictionary
print(f"Document ID: {example_doc_id}, Vector: {nnn_vectors[example_doc_id]}")

print("\nExample ntn vector for the same document:")
print(f"Document ID: {example_doc_id}, Vector: {ntn_vectors[example_doc_id]}")

print("\nExample ntc vector for the same document:")
print(f"Document ID: {example_doc_id}, Vector: {ntc_vectors[example_doc_id]}")

