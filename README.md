# Information Retrieval Techniques

## Introduction
This repository showcases various information retrieval techniques implemented using Python 3.10. The primary focus is on comparing different retrieval methodologies, linguistic post-processing, dictionary structures, wildcard querying, and tolerant retrieval. The experiments are conducted on NFCorpus and GENA datasets, analyzing trade-offs in terms of efficiency, complexity, and applicability to different search requirements.

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
    1. [Experiment 1: Comparing Grep with Inverted Index-based Boolean Retrieval](#experiment-1-comparing-grep-with-inverted-index-based-boolean-retrieval)
    2. [Experiment 2: Linguistic Post-processing](#experiment-2-linguistic-post-processing)
    3. [Experiment 3: Hash-based vs. Tree-based Dictionaries](#experiment-3-hash-based-vs-tree-based-dictionaries)
    4. [Experiment 4: Wildcard Querying](#experiment-4-wildcard-querying)
    5. [Experiment 5: Tolerant Retrieval](#experiment-5-tolerant-retrieval)
3. [Additional Experiments](#additional-experiments)
    1. [Dataset](#dataset)
    2. [Vector-based Models](#vector-based-models)
    3. [Rocchio Feedback Algorithm](#rocchio-feedback-algorithm)
    4. [Probabilistic Retrieval](#probabilistic-retrieval)
    5. [Entity-based Retrieval Models](#entity-based-retrieval-models)
    6. [Query Expansion using Knowledge Graphs](#query-expansion-using-knowledge-graphs)
    7. [Learning to Rank Models](#learning-to-rank-models)
    8. [Enhancements to Improve NDCG Scores](#enhancements-to-improve-ndcg-scores)
4. [Conclusion](#conclusion)

## Methodology

### Experiment 1: Comparing Grep with Inverted Index-based Boolean Retrieval

**Implementation Details:**

- **Grep Command:** Requires query terms to appear consecutively.
- **Boolean Retrieval (Inverted Index):** Handles non-consecutive occurrences of query terms.

**Time Complexity:**

- **Grep Command:** O(N×m)
- **Boolean Retrieval:** Index creation O(nlogn+m), Query O(ΣP)

**Memory Complexity:**

- **Grep Command:** O(k)
- **Boolean Retrieval:** Worst case O(t*d)

**Results:**

- **Grep:**
  - Average: 26969.27 µs
  - Minimum: 24952.01 µs
  - Maximum: 37849.59 µs

- **Inverted Index Boolean Retrieval:**
  - Average: 83.30 µs
  - Minimum: 6.76 µs
  - Maximum: 1039.45 µs

### Experiment 2: Linguistic Post-processing

**Techniques Applied:**

- Stemming
- Lemmatization
- Stop word removal

**Results:**

- Average time per query: 189.84 µs
- Minimum time: 63.94 µs
- Maximum time: 442.08 µs

### Experiment 3: Hash-based vs. Tree-based Dictionaries

**Implementation Details:**

- **Hash-based Retrieval:** O(1) lookup time
- **Trie-based Retrieval:** O(m) lookup time

**Memory Complexity:**

- **Hash Table:** O(T × (Lavg + Davg))
- **Tries:** O(N+T×Davg)

**Results:**

- **Trie Based:**
  - Average: 93.84 µs
  - Minimum: 8.37 µs
  - Maximum: 1338.13 µs
  - Memory: 288.25 MiB

- **Hash Based:**
  - Memory: 264.66 MiB

### Experiment 4: Wildcard Querying

**Implementation Details:**

- **Permuterm Index:** Token rotations for prefix searches
- **Forward and Backward Index:** Two tries for prefix and postfix searches

**Results:**

- **Permuterm Index:**
  - Average: 37664.63 µs
  - Minimum: 36748.28 µs
  - Maximum: 39471.74 µs

- **Tree-based Indexes:**
  - Average: 13226.69 µs
  - Minimum: 160.98 µs
  - Maximum: 41552.38 µs

### Experiment 5: Tolerant Retrieval

**Approach:**

- Forward and backward index tries
- Levenshtein Distance for spelling mistakes

**Results:**

- Average time per query: 380.46 ms
- Minimum time: 0.94 ms
- Maximum time: 1075.51 ms

## Advanced Ranking Experiments

### Dataset

**NFCorpus:** Documents, queries, query-document relevance values.
**GENA Knowledge Graph:** Connections between nutrition and mental well-being.

### Vector-based Models

Implemented vector space models (nnn, ntn, ntc).

### Rocchio Feedback Algorithm

**Results:**

- NDCG@3 before Rocchio: 0.2746
- NDCG@3 after Rocchio: 0.2645
- NDCG@10 before Rocchio: 0.1660
- NDCG@10 after Rocchio: 0.1607

### Probabilistic Retrieval

**Models Used:**

- **Language Model with Jelinek-Mercer Smoothing:** λ=0.1
- **BM25:** k1=1.5, b=0.75, epsilon=0.25

**Evaluation:**

- Precision and recall metrics were used.

### Entity-based Retrieval Models

Integrated GENA knowledge graph for entity-based search.

### Query Expansion using Knowledge Graphs

Developed a query expansion technique using Spacy's NLP model and re-extraction.

### Learning to Rank Models

Implemented pointwise, pairwise, and listwise learning to rank models using RandomForestClassifier, SVM, and a custom PyTorch model.

**Results:**

- **Pointwise:** NDCG@10: 0.6732
- **Pairwise:** NDCG@10 with 10 pairs per query: 0.3589
- **Listwise:** NDCG@10: 0.875

### Enhancements to Improve NDCG Scores

Used stemming, lemmatization, and bio-specific vocabulary from spaCy.

## Conclusion

This project explored several retrieval methods, highlighting the trade-offs between query time, indexing time, and memory usage. The experiments provide valuable insights into the efficiency and applicability of different information retrieval techniques.
