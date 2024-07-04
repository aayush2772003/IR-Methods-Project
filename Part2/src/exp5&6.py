import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

print("Start load")
nlp = spacy.load("../en_core_sci_md-0.5.4/en_core_sci_md/en_core_sci_md-0.5.4/")
print("End load")


# Load Documents
def load_documents(file_path):
    try:
        with open(file_path, 'r') as file:
            return {
                parts[0]: {'url': parts[1], 'title': parts[2].lower(), 'abstract': parts[3].lower()}
                for parts in (line.strip().split('\t') for line in file)
                if len(parts) >= 4
            }
    except Exception as e:
        print(f"Failed to load documents: {e}")
        return {}

# Load GENA Knowledge Graph
def load_gena_knowledge_graph(file_path):
    try:
        df = pd.read_csv(file_path).applymap(lambda x: x.lower() if isinstance(x, str) else x)
        graph = {}
        for subject, group in df.groupby('Subject'):
            graph[subject] = [(row['Relation'], row['Object']) for _, row in group.iterrows()]
            # graph[subject] = [(row['Relation'], row['Object']) for _, row in group.iterrows() if row['Relation'] in ['synonyms_1', 'synonyms_2', 'full_e1', 'mesh_e1'] and row['Object'] != 'no synonyms']
        return graph
    except Exception as e:
        print(f"Failed to load GENA knowledge graph: {e}")
        return {}


# Entity extraction using SpaCy
# Process texts in batches
def extract_entities(texts):
    texts = [str(text).lower() for text in texts]  # Ensure texts are lowercased
    docs = nlp.pipe(texts, batch_size=20)
    return [[ent.text.lower() for ent in doc.ents] for doc in docs]



# Entity-based Retrieval Model
def entity_based_retrieval(query, documents):
    query_entities = extract_entities([query.lower()])  # Lowercase the query
    flat_query_entities = [entity for sublist in query_entities for entity in sublist]

    query_vec = vectorizer.transform([' '.join(flat_query_entities)])
    results = {}
    for doc_id, doc_content in documents.items():
        doc_entities = extract_entities([doc_content['title'].lower()])  # Ensure title is lowercased
        flat_doc_entities = [entity for sublist in doc_entities for entity in sublist]

        doc_vec = vectorizer.transform([' '.join(flat_doc_entities)])
        similarity = cosine_similarity(query_vec, doc_vec)[0][0]
        results[doc_id] = similarity

    return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))



# Query Expansion using Knowledge Graph
def expand_query_with_kg(query_entities, gena_graph):
    expanded_entities = set(query_entities)
    for entity in query_entities:
        if entity in gena_graph:
            expanded_entities.update(str(obj) for _, obj in gena_graph[entity])  # Ensure objects are strings
    return list(expanded_entities)


def query_expansion_retrieval(query, documents, gena_graph):
    # Extract entities from the query and flatten the list
    query_entities = extract_entities([query])  
    flat_query_entities = [str(entity) for sublist in query_entities for entity in sublist]  # Convert entities to strings

    print("Original Query Entities:", flat_query_entities)  

    # Expand query using the knowledge graph
    expanded_query_entities = expand_query_with_kg(flat_query_entities, gena_graph)

    print("Expanded Query Entities:", expanded_query_entities)  

    # Re-extract entities from the expanded query
    re_extracted_entities = extract_entities([' '.join(expanded_query_entities)])  # Join expanded entities and re-extract
    re_extracted_flat = [entity for sublist in re_extracted_entities for entity in sublist]  # Flatten the list again
    print("Re-extracted Entities:", re_extracted_flat)  # Print re-extracted entities
    expanded_query_entities = re_extracted_flat
    # Transform the expanded query entities into a vector
    query_vec = vectorizer.transform([' '.join(expanded_query_entities)])

    results = {}
    for doc_id, doc_content in documents.items():
        # Extract and flatten document title entities
        doc_entities = extract_entities([doc_content['title']])
        flat_doc_entities = [str(entity) for sublist in doc_entities for entity in sublist]  # Convert entities to strings

        # Transform the flat list of document title entities into a vector
        doc_vec = vectorizer.transform([' '.join(flat_doc_entities)])

        # Calculate cosine similarity
        similarity = cosine_similarity(query_vec, doc_vec)[0][0]
        results[doc_id] = similarity

    return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

# Initialize Data and Vectorizer
documents = load_documents('../nfcorpus/raw/doc_dump.txt')
gena_graph = load_gena_knowledge_graph('../gena_data_final_triples.csv')

print("Extracting entities...")
all_entities = []
for content in documents.values():
    for entity_list in extract_entities([content['title']] + [content['abstract']]):
        all_entities.extend(entity_list)

# Print some entities for demonstration
print("Some entities:", all_entities[:10])

print("Fitting vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000).fit([' '.join(all_entities)])

# Example Query
query = "preventing brain loss with b vitamins ?"
results_entity_based = entity_based_retrieval(query, documents)
results_query_expansion = query_expansion_retrieval(query, documents, gena_graph)

print("done entity-based retrieval")
print("done query expansion retrieval")

# Compare results
# print("Entity-based Retrieval Results:", results_entity_based)
# print("Query Expansion Retrieval Results:", results_query_expansion)

print("Sample of results_entity_based:", next(iter(results_entity_based.items())))
for i, (doc_id, score) in enumerate(results_entity_based.items()):
    print(f"Document ID: {doc_id}, Score: {score}")
    if i == 4:
        break
print("Sample of results_query_expansion:", next(iter(results_query_expansion.items())))
for i, (doc_id, score) in enumerate(results_query_expansion.items()):
    print(f"Document ID: {doc_id}, Score: {score}")
    if i == 4:
        break


