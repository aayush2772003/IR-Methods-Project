import os

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

import json

nltk.download('punkt')  # For tokenization
nltk.download('wordnet')  # For lemmatization
nltk.download('stopwords')  # For stop word removal

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    # Create an empty string to store the cleaned text
    cleaned_text = ""
    
    # Iterate over each character in the input text
    for char in text:
        # Check if the character is an alphanumeric character or a space
        # and append it to the cleaned_text if it is
        if (ord(char) >= 48 and ord(char) <= 57) or \
           (ord(char) >= 65 and ord(char) <= 90) or \
           (ord(char) >= 97 and ord(char) <= 122) or \
           ord(char) == 32:
            cleaned_text += char
        else:
            # Replace special characters with a space
            cleaned_text += " "
    
    return cleaned_text

def read_json_corpus(json_path):
    f = open(json_path + "/s2_doc.json", encoding="utf-8")
    json_file = json.load(f)
    if not os.path.exists(json_path + "/intermediate/"):
        os.mkdir(json_path + "/intermediate/")
    o = open(json_path + "/intermediate/cleaned_output.tsv", "w", encoding="utf-8")
    for json_object in json_file['all_papers']:
        doc_no = json_object['docno']
        title = json_object['title'][0]
        title=clean_text(title)
        paper_abstract = json_object['paperAbstract'][0]
        paper_abstract=clean_text(paper_abstract)
        tokens = title.split()
        for t in tokens:
            o.write(t.lower() + "\t" + str(doc_no) + "\n")
        tokens = paper_abstract.split()
        for t in tokens:
            o.write(t.lower() + "\t" + str(doc_no) + "\n")
    o.close()

def sort(dir, lemm_stemm=False):
    if lemm_stemm:
        f = open(dir + "/intermediate/lemm_stemm_output.tsv", encoding="utf-8")
        o = open(dir + "/intermediate/lemm_stemm_output_sorted.tsv", "w", encoding="utf-8")
    else:
        f = open(dir + "/intermediate/cleaned_output.tsv", encoding="utf-8")
        o = open(dir + "/intermediate/cleaned_output_sorted.tsv", "w", encoding="utf-8")

    # initialize an empty list of pairs of
    # tokens and their doc_ids
    pairs = []

    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        if len(split_line) == 2:
            pair = (split_line[0], split_line[1])
            pairs.append(pair)

    # sort (token, doc_id) pairs by token first and then doc_id
    sorted_pairs = sorted(pairs, key=lambda x: (x[0], x[1]))

    # write sorted pairs to file
    for sp in sorted_pairs:
        o.write(sp[0] + "\t" + sp[1] + "\n")
    o.close()

# converts (token, doc_id) pairs
# into a dictionary of tokens
# and an adjacency list of doc_id
def construct_postings(dir, lemm_stemm=False):
    if lemm_stemm:
        # open file to write postings
        o1 = open(dir + "/intermediate/lemm_stemm_postings.tsv", "w", encoding="utf-8")
    else:
        o1 = open(dir + "/intermediate/cleaned_postings.tsv", "w", encoding="utf-8")

    postings = {}  # initialize our dictionary of terms
    doc_freq = {}  # document frequency for each term

    # read the file containing the sorted pairs
    if lemm_stemm:
        f = open(dir + "/intermediate/lemm_stemm_output_sorted.tsv", encoding="utf-8")
    else:
        f = open(dir + "/intermediate/cleaned_output_sorted.tsv", encoding="utf-8")

    # initialize sorted pairs
    sorted_pairs = []

    # read sorted pairs
    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        pairs = (split_line[0], split_line[1])
        sorted_pairs.append(pairs)

    # construct postings from sorted pairs
    for pairs in sorted_pairs:
        if pairs[0] not in postings:
            postings[pairs[0]] = []
            postings[pairs[0]].append(pairs[1])
        else:
            len_postings = len(postings[pairs[0]])
            if len_postings >= 1:
                # check for duplicates
                # the same doc_ids will appear
                # one after another and detected by
                # checking the last element of the postings
                if pairs[1] != postings[pairs[0]][len_postings - 1]:
                    postings[pairs[0]].append(pairs[1])

    # update doc_freq which is the size of postings list
    for token in postings:
        doc_freq[token] = len(postings[token])

    # print("postings: " + str(postings))
    # print("doc freq: " + str(doc_freq))
    print("Dictionary size: " + str(len(postings)))

 # write postings and document frequency to file

    for token in postings:
        o1.write(token + "\t" + str(doc_freq[token]))
        for l in postings[token]:
            o1.write("\t" + l)
        o1.write("\n")
    o1.close()

def read_json_corpus_lemm_Stemm(json_path):
    f = open(json_path + "/s2_doc.json", encoding="utf-8")
    json_file = json.load(f)
    if not os.path.exists(json_path + "/intermediate/"):
        os.mkdir(json_path + "/intermediate/")
    o = open(json_path + "/intermediate/lemm_stemm_output.tsv", "w", encoding="utf-8")
    for json_object in json_file['all_papers']:
        doc_no = json_object['docno']
        title = json_object['title'][0]
        title = clean_text(title)
        paper_abstract = json_object['paperAbstract'][0]
        paper_abstract = clean_text(paper_abstract)
        for field in [title, paper_abstract]:
            tokens = nltk.word_tokenize(field.lower())
            filtered_tokens = [stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens if token not in stop_words and token.isalpha()]
            for t in filtered_tokens:
                o.write(t.lower() + "\t" + str(doc_no) + "\n")
    o.close()

# starting the indexing process
def index(dir, lemm_stemm=False):
    # reads the corpus and
    # creates an intermediary file
    # containing token and doc_id pairs.
    # read_corpus(dir)
    if lemm_stemm:
        read_json_corpus_lemm_Stemm(dir)
    else:
        read_json_corpus(dir)

    # sorts (token, doc_id) pairs
    # by token first and then doc_id
    sort(dir, lemm_stemm=lemm_stemm)

    # converts (token, doc_id) pairs
    # into a dictionary of tokens
    # and an adjacency list of doc_id
    construct_postings(dir, lemm_stemm=lemm_stemm)