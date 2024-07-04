
import shutil
import os
import json
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")

import cProfile
import pstats
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
import matplotlib.pyplot as plt
import json
import time

import json
import sys

nltk.download('punkt')  # For tokenization
nltk.download('wordnet')  # For lemmatization
nltk.download('stopwords')  # For stop word removal

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

from utility import index
import time
# from memory_profiler import profile
import cProfile
import pstats

# Define a no-op decorator to use when memory profiling is not enabled
def noop_decorator(f):
    return f

# Check if '--profile-memory' is in the command-line arguments
if '--profile-memory' in sys.argv:
    print("Memory profiling enabled")
    from memory_profiler import profile
else:
    profile = noop_decorator

time_profile = '--time-profile' in sys.argv


def load_index_in_memory(dir, lemm_stemm=False):
    if lemm_stemm:
        f = open(dir + "intermediate/lemm_stemm_postings.tsv", encoding="utf-8")
    else:
        f = open(dir + "intermediate/cleaned_postings.tsv", encoding="utf-8")

    postings = {}
    doc_freq = {}

    for line in f:
        splitline = line.split("\t")

        token = splitline[0]
        freq = int(splitline[1])

        doc_freq[token] = freq

        item_list = []

        for item in range(2, len(splitline)):
            item_list.append(splitline[item].strip())
        postings[token] = item_list

    return postings, doc_freq


def intersection(l1, l2):
    count1 = 0
    count2 = 0
    intersection_list = []

    while count1 < len(l1) and count2 < len(l2):
        if l1[count1] == l2[count2]:
            intersection_list.append(l1[count1])
            count1 = count1 + 1
            count2 = count2 + 1
        elif l1[count1] < l2[count2]:
            count1 = count1 + 1
        elif l1[count1] > l2[count2]:
            count2 = count2 + 1

    return intersection_list


def and_query(query_terms, corpus, postings, doc_freq, lemm_stemm=False,):
   

    # postings for only the query terms
    postings_for_keywords = {}
    doc_freq_for_keywords = {}

    for q in query_terms:
        if q in postings:
            postings_for_keywords[q] = postings[q]
        else:
            postings_for_keywords[q] = []

    # store doc frequency for query token in
    # dictionary

    for q in query_terms:
        if q in doc_freq:
            doc_freq_for_keywords[q] = doc_freq[q]
        else:
            doc_freq_for_keywords[q] = 0


    # sort tokens in increasing order of their
    # frequencies

    sorted_tokens = sorted(doc_freq_for_keywords.items(), key=lambda x: x[1])

    # initialize result to postings list of the
    # token with minimum doc frequency

    if not sorted_tokens:  # Check if sorted_tokens is empty
        return []  # No documents found for query terms
    # result = postings_for_keywords[sorted_tokens[0][0]]
    result = set(postings_for_keywords[sorted_tokens[0][0]])
    # iterate over the remaining postings list and
    # intersect them with result, and updating it
    # in every step
    
    for i in range(1, len(postings_for_keywords)):
        #result = intersection(result,postings_for_keywords[sorted_tokens[i][0]])
        result = result.intersection(set(postings_for_keywords[sorted_tokens[i][0]]))
        if len(result) == 0:
            return result

    return result

def search_in_json(query, documents):
    doc_match_count = 0
    total_matches = 0
    for doc in documents:
        # Check if the title is a list and join it into a single string if so
        title = " ".join(doc['title']) if isinstance(doc['title'], list) else doc['title']
        # Similar check for the abstract
        abstract = " ".join(doc['paperAbstract']) if isinstance(doc['paperAbstract'], list) else doc['paperAbstract']
        
        in_title = query.lower() in title.lower()
        in_abstract = query.lower() in abstract.lower()
        
        if in_title or in_abstract:
            doc_match_count += 1
            if in_title:
                total_matches += title.lower().count(query.lower())
            if in_abstract:
                total_matches += abstract.lower().count(query.lower())
    return doc_match_count, total_matches

@profile
def grep():
    doc_path = "s2/s2_doc.json"
    query_path = "s2/s2_query.json"

    # Load documents and queries
    with open(doc_path, 'r') as file:
        documents = json.load(file)['all_papers']
    with open(query_path, 'r') as file:
        queries = json.load(file)['queries']

    # Initialize DataFrame to store results
    results_df = pd.DataFrame(columns=['Query', 'Document Matches', 'Total Matches', 'Execution Time (microseconds)'])

    times = []

    for query in queries:
        query_text = query['query']
        start_time = time.perf_counter()
        doc_match_count, total_matches = search_in_json(query_text, documents)
        exec_time = (time.perf_counter() - start_time) * 1_000_000 
        times.append(exec_time)
        
        # Prepare a new row to add
        new_row = pd.DataFrame({
            'Query': [query_text],
            'Document Matches': [doc_match_count],
            'Total Matches': [total_matches],
            'Execution Time (microseconds)': [exec_time]
        })

        # Use pd.concat to append the new row
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Save results to CSV
    results_df.to_csv('grep_query_results.csv', index=False)

    # Plot execution times
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Query'], results_df['Execution Time (microseconds)'], marker='o', linestyle='-', color='blue')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Query')
    plt.ylabel('Execution Time (microseconds)')
    plt.title('Query Execution Times')
    plt.tight_layout()
    plt.savefig('grep_query_execution_times.png')
    plt.show()

    # Print average, minimum, and maximum times
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"\nAverage execution time: {avg_time:.6f} microseconds")
        print(f"Minimum execution time: {min_time:.6f} microseconds")
        print(f"Maximum execution time: {max_time:.6f} microseconds")
    else:
        print("No queries were processed.")

@profile
def bool_ret_lemm_Stemm(query_file):
    f = open(query_file, encoding="utf-8")
    json_query_file_object = json.load(f)
    
    # Initialize lemmatizer, stemmer, and stop words
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    postings, doc_freq = load_index_in_memory("s2/", lemm_stemm=True)

    # DataFrame to store results
    results_df = pd.DataFrame(columns=['Query', 'Document Count', 'Processing Time (microseconds)'])

    # Dictionary to store matching documents
    matching_documents = {}

    total_time = 0
    min_time = float('inf')
    max_time = 0
    query_count = 0

    for query in json_query_file_object["queries"]:
        start_time = time.perf_counter()

        query_terms = nltk.word_tokenize(query["query"].lower())
        query_terms = [stemmer.stem(lemmatizer.lemmatize(token)) for token in query_terms if token not in stop_words and token.isalpha()]

        docs_cont_all_query_terms = and_query(query_terms, "s2/", postings, doc_freq, lemm_stemm=True)

        end_time = time.perf_counter()
        query_time = (end_time - start_time) * 1_000_000  # Convert to microseconds for precision

        # Add row to DataFrame
        new_row = pd.DataFrame({
            'Query': [' '.join(query_terms)],
            'Document Count': [len(docs_cont_all_query_terms)],
            'Processing Time (microseconds)': [query_time]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Update matching documents dictionary
        matching_documents[' '.join(query_terms)] = list(docs_cont_all_query_terms)

        total_time += query_time
        min_time = min(min_time, query_time)
        max_time = max(max_time, query_time)
        query_count += 1

    # Save results to CSV
    results_df.to_csv('lemm_stemm_query_results.csv', index=False)

    # Plot processing time
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Query'], results_df['Processing Time (microseconds)'], marker='o')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Query')
    plt.ylabel('Processing Time (microseconds)')
    plt.title('Query Processing Times')
    plt.tight_layout()
    plt.savefig('lemm_stemm_query_processing_times.png')
    plt.show()

    # Save matching documents to JSON
    with open('lemm_stemm_matching_documents.json', 'w') as json_file:
        json.dump(matching_documents, json_file, indent=4)

    if query_count > 0:
        avg_time = total_time / query_count
        print(f"\nAverage time per query: {avg_time:.6f} microseconds.")
        print(f"Minimum time for a query: {min_time:.6f} microseconds.")
        print(f"Maximum time for a query: {max_time:.6f} microseconds.")
    else:
        print("No queries were processed.")

@profile
def bool_ret(query_file):
    f = open(query_file, encoding="utf-8")
    json_query_file_object = json.load(f)
    postings, doc_freq = load_index_in_memory("s2/", lemm_stemm=False)

    # Initialize a DataFrame to store results
    results_df = pd.DataFrame(columns=['Query', 'Document Count', 'Processing Time'])

    # Dictionary to store matching documents for each query
    matching_documents = {}

    for query in json_query_file_object["queries"]:
        start_time = time.perf_counter()

        query_terms = query["query"].split(" ")
        docs_cont_all_query_terms = and_query(query_terms, "s2/", postings, doc_freq, lemm_stemm=False)

        end_time = time.perf_counter()
        query_time = (end_time - start_time)*1_000_000

        # Prepare a new row to add
        new_row = pd.DataFrame({
            'Query': [' '.join(query_terms)],
            'Document Count': [len(docs_cont_all_query_terms)],
            'Processing Time': [query_time]
        })

        # Add new row to DataFrame
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Update matching documents dictionary
        matching_documents[' '.join(query_terms)] = list(docs_cont_all_query_terms)

    # Save results to CSV
    results_df.to_csv('bool_ret_query_results.csv', index=False)

    # Plot processing time
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Query'], results_df['Processing Time'], marker='o')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Query')
    plt.ylabel('Processing Time (microseconds)')
    plt.title('Query Processing Times')
    plt.tight_layout()
    plt.savefig('bool_ret_query_processing_times.png')
    plt.show()

    # Save matching documents to JSON
    with open('bool_ret_matching_documents.json', 'w') as json_file:
        json.dump(matching_documents, json_file, indent=4)
    
    # Print average, minimum, and maximum times
    if results_df.shape[0] > 0:
        avg_time = results_df['Processing Time'].mean()
        min_time = results_df['Processing Time'].min()
        max_time = results_df['Processing Time'].max()
        print(f"\nAverage time per query: {avg_time:.6f} microseconds.")
        print(f"Minimum time for a query: {min_time:.6f} microseconds.")
        print(f"Maximum time for a query: {max_time:.6f} microseconds.")

###### Trie based dictionary ######
class TrieNodeForDict:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.posting_list = []  # To store document IDs
        self.doc_frequency = 0  # To store document frequency
class TrieDict:
    def __init__(self):
        self.root = TrieNodeForDict()

    def insert(self, word, doc_id):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNodeForDict()
            node = node.children[char]

        # If the word is already present, we just add the doc_id to the posting list
        if not node.is_end_of_word:
            node.is_end_of_word = True
            node.doc_frequency = 1  # Starting the doc frequency count
        else:
            node.doc_frequency += 1  # Incrementing doc frequency if word already exists
        
        if doc_id not in node.posting_list:
            node.posting_list.append(doc_id)  # Adding document ID to posting list

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None, 0  # Word not found
            node = node.children[char]

        if node.is_end_of_word:
            return node.posting_list, node.doc_frequency
        return None, 0  # Word not found but is a prefix to another word

    def insert_document(self, doc_id, tokens):
        for token in tokens:
            self.insert(token, doc_id)

def load_index_in_memory_trie(dir):
    f = open(dir + "intermediate/cleaned_postings.tsv", encoding="utf-8")
    postings_Trie=TrieDict()

    for line in f:
        splitline = line.split("\t")
        token = splitline[0]
        for item in range(2, len(splitline)):
            postings_Trie.insert(token,splitline[item].strip())
    return postings_Trie

def and_queryTrieDict(query_terms, corpus,postings_Trie):
   

    # postings for only the query terms
    postings_for_keywords = {}
    doc_freq_for_keywords = {}

    for q in query_terms:
        random_var, val=postings_Trie.search(q)
        if random_var is not None:
            postings_for_keywords[q] = random_var
            doc_freq_for_keywords[q] = val
        else:
            postings_for_keywords[q] = []
            doc_freq_for_keywords[q] = 0

    sorted_tokens = sorted(doc_freq_for_keywords.items(), key=lambda x: x[1])
    if not sorted_tokens:  # Check if sorted_tokens is empty
        return []

    result = set(postings_for_keywords[sorted_tokens[0][0]])
    for i in range(1, len(postings_for_keywords)):
        # result = intersection(result, postings_for_keywords[sorted_tokens[i][0]])
        result = result.intersection(set(postings_for_keywords[sorted_tokens[i][0]]))
        if len(result) == 0:
            return result
    return result
@profile
def bool_ret_using_trie(query_file):
    f = open(query_file, encoding="utf-8")
    json_query_file_object = json.load(f)
    postings_Trie = load_index_in_memory_trie("s2/")

    # DataFrame to store results
    results_df = pd.DataFrame(columns=['Query', 'Document Count', 'Processing Time (microseconds)'])

    # Dictionary to store matching documents
    matching_documents = {}

    total_time = 0
    min_time = float('inf')
    max_time = 0
    query_count = 0

    for query in json_query_file_object["queries"]:
        start_time = time.perf_counter()

        query_terms = query["query"].split(" ")
        docs_cont_all_query_terms = and_queryTrieDict(query_terms, "s2/", postings_Trie)

        end_time = time.perf_counter()
        query_time = (end_time - start_time) * 1_000_000  # Convert to microseconds for precision

        # Prepare a new row to add
        new_row = pd.DataFrame({
            'Query': [' '.join(query_terms)],
            'Document Count': [len(docs_cont_all_query_terms)],
            'Processing Time (microseconds)': [query_time]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Update matching documents dictionary
        matching_documents[' '.join(query_terms)] = list(docs_cont_all_query_terms)

        total_time += query_time
        min_time = min(min_time, query_time)
        max_time = max(max_time, query_time)
        query_count += 1

    # Save results to CSV
    results_df.to_csv('trie_query_results.csv', index=False)

    # Plot processing time
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Query'], results_df['Processing Time (microseconds)'], marker='o')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Query')
    plt.ylabel('Processing Time (microseconds)')
    plt.title('Query Processing Times')
    plt.tight_layout()
    plt.savefig('trie_query_processing_times.png')
    plt.show()

    # Save matching documents to JSON
    with open('trie_matching_documents.json', 'w') as json_file:
        json.dump(matching_documents, json_file, indent=4)

    if query_count > 0:
        avg_time = total_time / query_count
        print(f"\nAverage time per query: {avg_time:.6f} microseconds.")
        print(f"Minimum time for a query: {min_time:.6f} microseconds.")
        print(f"Maximum time for a query: {max_time:.6f} microseconds.")
    else:
        print("No queries were processed.")

##### PERMUTERM INDEXING #####
        
def rotate(str, n):
    return str[n:] + str[:n]

def load_permuterm_index_in_memory(index_file):
    permuterm_index = {}
    with open(index_file, 'r', encoding='utf-8') as file:
        for line in file:
            permuterm, original_term = line.strip().split("\t")
            if permuterm in permuterm_index:
                permuterm_index[permuterm].append(original_term)
            else:
                permuterm_index[permuterm] = [original_term]
    return permuterm_index

def createPermutermIndex():
    postings,doc_freq=load_index_in_memory("s2/")
    p = open("s2/intermediate/PermutermIndex.tsv", "w", encoding="utf-8")
    for token in postings.keys():
        token_end=token+"$"
        for i in range(len(token_end)):
            out = rotate(token_end,i)
            p.write(out+"\t"+token+"\n")
    p.close()

def rotate_query(query):
    if query.endswith('*'):
        return "$" + query
    elif query.startswith('*'):
        return query[1:] + '$*'
    else:
        # For cases like X*Y, find the first '*' and rotate around it
        star_index = query.find('*')
        return query[star_index+1:] + '$' + query[:star_index] + '*'   

def lookup_permuterm(query, permuterm_index):
    rotated_query = rotate_query(query)
    # print(rotated_query)
    matches = []

    # Search the permuterm index
    for permuterm, original_terms in permuterm_index.items():
        if permuterm.startswith(rotated_query.replace('*', '')):
            matches.extend(original_terms)            
    return matches

@profile
def wc_permuterm_main():
    # Load the permuterm index into memory
    permuterm_index = load_permuterm_index_in_memory("s2/intermediate/PermutermIndex.tsv")
    postings,doc_freq=load_index_in_memory("s2/")
    
    with open("s2/s2_wildcard.json", encoding="utf-8") as f:
        json_query_file_object = json.load(f)

    # DataFrame to store results
    results_df = pd.DataFrame(columns=['Query', 'Term Count', 'Processing Time (microseconds)'])

    # Dictionary to store query terms and corresponding documents found
    query_terms_and_documents_found = {}

    total_time = 0
    min_time = float('inf')
    max_time = 0
    query_count = 0

    for query in json_query_file_object["queries"]:
        start_time = time.perf_counter()

        queryterms = lookup_permuterm(query["query"], permuterm_index)  # Use the in-memory index
        documents_for_terms = {}

        for term in queryterms:
            documents = postings.get(term, [])
            documents_for_terms[term] = documents

        end_time = time.perf_counter()
        query_time = (end_time - start_time) * 1_000_000  # Convert to microseconds for precision

        # Prepare a new row to add
        new_row = pd.DataFrame({
            'Query': [query['query']],
            'Term Count': [len(queryterms)],
            'Processing Time (microseconds)': [query_time]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Update query terms and documents found dictionary
        query_terms_and_documents_found[query['query']] = documents_for_terms

        total_time += query_time
        min_time = min(min_time, query_time)
        max_time = max(max_time, query_time)
        query_count += 1

    # Save results to CSV
    results_df.to_csv('permuterm_query_results.csv', index=False)

    # Plot processing time
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Query'], results_df['Processing Time (microseconds)'], marker='o')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Query')
    plt.ylabel('Processing Time (microseconds)')
    plt.title('Query Processing Times')
    plt.tight_layout()
    plt.savefig('permuterm_query_processing_times.png')
    plt.show()

    # Save query terms and documents found to JSON
    with open('permuterm_query_terms_and_documents_found.json', 'w') as json_file:
        json.dump(query_terms_and_documents_found, json_file, indent=4)

    if query_count > 0:
        avg_time = total_time / query_count
        print(f"\nAverage time per query: {avg_time:.6f} microseconds.")
        print(f"Minimum time for a query: {min_time:.6f} microseconds.")
        print(f"Maximum time for a query: {max_time:.6f} microseconds.")
    else:
        print("No queries were processed.")

   
#### FW AND BW INDEXING #####

class TrieNodeFWBW:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class TrieFWBW:
    def __init__(self):
        self.root = TrieNodeFWBW()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNodeFWBW()
            node = node.children[char]
        node.is_end_of_word = True

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        return self._words_with_prefix(node, prefix)

    def _words_with_prefix(self, node, prefix):
        words = []
        if node.is_end_of_word:
            words.append(prefix)
        for char, next_node in node.children.items():
            words.extend(self._words_with_prefix(next_node, prefix + char))
        return words

def wildcard_query(forward_trie, backward_trie, query):
    prefix, suffix = query.split('*')
    forward_results = set(forward_trie.search_prefix(prefix))
    backward_results = set([word[::-1] for word in backward_trie.search_prefix(suffix[::-1])])
    return forward_results.intersection(backward_results)

def create_FW_BW_In():
    postings,doc_freq=load_index_in_memory("s2/")
    forward_trie=TrieFWBW()
    backward_trie=TrieFWBW()
    for token in postings.keys():
        forward_trie.insert(token)
        backward_trie.insert(token[::-1])
    return forward_trie, backward_trie

@profile
def forward_backward_main():
    forward_trie, backward_trie = create_FW_BW_In()
    
    postings, doc_freq = load_index_in_memory("s2/")
    f = open("s2/s2_wildcard.json", encoding="utf-8")
    json_query_file_object = json.load(f)

    # DataFrame to store results
    results_df = pd.DataFrame(columns=['Query', 'Unique Documents Count', 'Processing Time (microseconds)'])

    # Dictionary to store unique documents found for each query
    unique_documents_found = {}

    total_time = 0
    min_time = float('inf')
    max_time = 0
    query_count = 0

    for query in json_query_file_object["queries"]:
        start_time = time.perf_counter()

        queryterms = wildcard_query(forward_trie, backward_trie, query["query"])
        uniq_docs = set()
        for term in queryterms:
            if term in postings:
                uniq_docs.update(postings[term])

        end_time = time.perf_counter()
        query_time = (end_time - start_time) * 1_000_000  # Convert to microseconds for precision

        # Add to DataFrame
        new_row = pd.DataFrame({
            'Query': [query['query']],
            'Unique Documents Count': [len(uniq_docs)],
            'Processing Time (microseconds)': [query_time]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Update unique documents found dictionary
        unique_documents_found[query['query']] = list(uniq_docs)

        total_time += query_time
        min_time = min(min_time, query_time)
        max_time = max(max_time, query_time)
        query_count += 1

    # Save results to CSV
    results_df.to_csv('fw_bw_query_results.csv', index=False)

    # Plot processing time
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Query'], results_df['Processing Time (microseconds)'], marker='o')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Query')
    plt.ylabel('Processing Time (microseconds)')
    plt.title('Query Processing Times')
    plt.tight_layout()
    plt.savefig('fw_bw_query_processing_times.png')
    plt.show()

    # Save unique documents found to JSON
    with open('fw_bw_unique_documents_found.json', 'w') as json_file:
        json.dump(unique_documents_found, json_file, indent=4)

    if query_count > 0:
        avg_time = total_time / query_count
        print(f"\nAverage time per query: {avg_time:.6f} microseconds.")
        print(f"Maximum time for a query: {max_time:.6f} microseconds.")
        print(f"Minimum time for a query: {min_time:.6f} microseconds.")
    else:
        print("No queries processed.")

###### Tolerant Retrieval ######
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 
            deletions = current_row[j] + 1       
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def dynamic_threshold(word):
    length = len(word)
    base_threshold = 0

    if length <= 4:
        return base_threshold
    elif length <= 7:
        return base_threshold + 1
    elif length <= 9:
        return base_threshold + 2
    else:
        return base_threshold + 3
    
def find_similar_words(query_word, postings, max_distance=2):

    similar_words = []
    for word in postings.keys():
        distance = levenshtein_distance(query_word, word)
        if distance <= max_distance:
            similar_words.append(word)
    return similar_words

def wildcard_query_Tolerant(forward_trie, backward_trie, query, postings):
    if query.__contains__('*'):
        prefix, suffix = query.split('*')
        forward_results = set(forward_trie.search_prefix(prefix))
        backward_results = set([word[::-1] for word in backward_trie.search_prefix(suffix[::-1])])
        return forward_results.intersection(backward_results)
    else:

        max_distance = dynamic_threshold(query)
        # return [query]
        # print(f"Query: {query}")
        # print(find_similar_words(query, postings, max_distance=max_distance))
        return find_similar_words(query, postings, max_distance=max_distance)
    
@profile
def TolerantRetrievalMain():
    forward_trie, backward_trie = create_FW_BW_In()
    
    postings, doc_freq = load_index_in_memory("s2/")
    f = open("s2/s2_wildcard_boolean.json", encoding="utf-8")
    json_query_file_object = json.load(f)

    # DataFrame to store results
    results_df = pd.DataFrame(columns=['Query', 'Results Count', 'Processing Time'])

    # Dictionary to store documents found for each query
    documents_found = {}

    total_time = 0
    min_time = float('inf')
    max_time = 0
    query_count = 0

    for query in json_query_file_object["queries"]:
        query_start_time = time.perf_counter()

        doc_cont_all_combn_ans = set()
        wildcards = query["query"].split()
        for wildcard in wildcards:
            queryterms = wildcard_query_Tolerant(forward_trie, backward_trie, wildcard, postings)
            union_of_docs = set()
            for queryterm in queryterms:
                if queryterm in postings:
                    union_of_docs.update(postings[queryterm])
            doc_cont_all_combn_ans = doc_cont_all_combn_ans.union(union_of_docs) if not doc_cont_all_combn_ans else doc_cont_all_combn_ans.intersection(union_of_docs)

        query_end_time = time.perf_counter()
        query_time = (query_end_time - query_start_time) * 1_000  # Convert to milliseconds

        # Add to DataFrame
        new_row = pd.DataFrame({
            'Query': [query['query']],
            'Results Count': [len(doc_cont_all_combn_ans)],
            'Processing Time': [query_time]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Update documents found dictionary
        documents_found[query['query']] = list(doc_cont_all_combn_ans)

        total_time += query_time
        min_time = min(min_time, query_time)
        max_time = max(max_time, query_time)
        query_count += 1

    # Save results to CSV
    results_df.to_csv('tolerant_retrieval_query_results.csv', index=False)

    # Plot processing time
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Query'], results_df['Processing Time'], marker='o')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel('Query')
    plt.ylabel('Processing Time (milliseconds)')
    plt.title('Query Processing Times')
    plt.tight_layout()
    plt.savefig('tolerant_retrieval_query_processing_times.png')
    plt.show()

    # Save documents found to JSON
    with open('tolerant_retrieval_documents_found.json', 'w') as json_file:
        json.dump(documents_found, json_file, indent=4)

    if query_count > 0:
        avg_time = total_time / query_count
        print(f"\nAverage time per query: {avg_time:.6f} milliseconds.")
        print(f"Minimum time for a query: {min_time:.6f} milliseconds.")
        print(f"Maximum time for a query: {max_time:.6f} milliseconds.")
    else:
        print("No queries were processed.")

def main_menu():
    print("\nMain Menu")
    print("1 - Run bool_ret")
    print("2 - Run bool_ret_lemm_Stemm")
    print("3 - Run bool_ret_using_trie")
    print("4 - Run wc_permuterm_main")
    print("5 - Run forward_backward")
    print("6 - Run TolerantRetrievalMain")
    print("7 - Run grep")
    print("0 - Exit")

    choice = input("Enter your choice: ")
    return choice

# Code starts here
if __name__ == '__main__':
    index("s2/", lemm_stemm=False)
    index("s2/", lemm_stemm=True)
    createPermutermIndex()
    print("Indexing done")
    while True:
        user_choice = main_menu()
        
        if user_choice == "1":
            # profiler = cProfile.Profile()
            # profiler.enable()            
            bool_ret("./s2/s2_query.json")
            # profiler.disable()
            # stats = pstats.Stats(profiler).sort_stats('cumulative')
            # stats.print_stats()
        elif user_choice == "2":
            bool_ret_lemm_Stemm("./s2/s2_query.json")
        elif user_choice == "3":
            bool_ret_using_trie("s2/s2_query.json")
        elif user_choice == "4":
            wc_permuterm_main()
        elif user_choice == "5":
            # profiler = cProfile.Profile()
            # profiler.enable()
            forward_backward_main()
            # profiler.disable()
            # profiler.dump_stats('profile_stats.prof')
            # stats = pstats.Stats('profile_stats.prof')
            # stats.strip_dirs().sort_stats('time').print_stats()
        elif user_choice == "6":
            TolerantRetrievalMain()
        elif user_choice == "7":
            grep()
        elif user_choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")