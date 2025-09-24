import os
import math
import re
from collections import defaultdict, Counter
from flask import Flask, render_template, request

# Text processing: clean input text by converting to lowercase, removing punctuation, and splitting into words
def clean_text(text):
    text = text.lower()  # make all letters lowercase
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # remove non-alphanumeric characters
    return text.split()  # split text into a list of words

# Build an inverted index for the corpus
def create_index(folder_path):
    """
    Create the data structures needed for retrieval:
      - index: maps each term to document frequency and postings list
      - doc_lengths: length of each document vector for normalization
      - doc_map: maps numeric docID to filename
    """
    index = {}
    doc_lengths = {}
    doc_map = {}
    
    files = sorted(os.listdir(folder_path))
    N = len(files)  # total number of documents

    for i, fname in enumerate(files):
        doc_id = i + 1  # assign a numeric ID to the document
        doc_map[doc_id] = fname

        # Read document content and tokenize
        with open(os.path.join(folder_path, fname), "r", encoding="utf-8", errors="ignore") as f:
            words = clean_text(f.read())

        tf_counts = Counter(words)  # count occurrences of each term in the document

        # Populate the inverted index
        for term, tf in tf_counts.items():
            weight = 1 + math.log10(tf)  # log-weighted term frequency
            if term not in index:
                index[term] = [0, []]  # initialize document frequency and postings
            index[term][0] += 1  # increment document frequency for this term
            index[term][1].append((doc_id, weight))  # add posting for this term

    # Compute the length of each document vector for cosine similarity
    for postings in index.values():
        for doc_id, w in postings[1]:
            doc_lengths[doc_id] = doc_lengths.get(doc_id, 0) + w ** 2

    for doc_id in doc_lengths:
        doc_lengths[doc_id] = math.sqrt(doc_lengths[doc_id])  # final vector length

    return index, doc_lengths, doc_map, N

# Rank documents for a query using cosine similarity with tf-idf
def rank_query(query, index, doc_lengths, doc_map, N, top_k=10):
    """
    Compute the relevance score of each document for the query and return top_k results.
    """
    tokens = clean_text(query)  # tokenize query
    q_tf = Counter(tokens)      # term frequencies in the query

    # Build weighted query vector
    q_weights = {}
    for term, tf in q_tf.items():
        if term in index:
            weight = 1 + math.log10(tf)  # log-weighted term frequency
            df = index[term][0]          # document frequency of term
            idf = math.log10(N / df)     # inverse document frequency
            q_weights[term] = weight * idf  # tf-idf weight for query term

    # Normalize query vector to unit length
    norm = math.sqrt(sum(w**2 for w in q_weights.values()))
    if norm > 0:
        q_weights = {t: w / norm for t, w in q_weights.items()}

    # Compute cosine similarity between query and each document
    scores = defaultdict(float)
    for term, q_w in q_weights.items():
        for doc_id, d_w in index[term][1]:
            scores[doc_id] += q_w * (d_w / doc_lengths[doc_id])

    # Sort documents by decreasing score, tie-breaker is docID
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    # Return top_k documents with their scores
    return [(doc_map[doc_id], score) for doc_id, score in ranked[:top_k]]

# Initialize Flask web app
app = Flask(__name__)

# Path to corpus folder
corpus_folder = r"C:\Users\kousi\OneDrive\Desktop\study\new IR project\Corpus"

# Preprocess and index the corpus once when server starts
index, doc_lengths, doc_map, N = create_index(corpus_folder)

@app.route("/", methods=["GET", "POST"])
def search_page():
    """
    Display search page and handle query submissions.
    GET: show search box
    POST: compute and display ranked results
    """
    query = ""
    results = []

    if request.method == "POST":
        query = request.form.get("query", "")
        results = rank_query(query, index, doc_lengths, doc_map, N)

    return render_template("index.html", query=query, results=results)

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)






