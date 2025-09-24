import os
import math
import re
from collections import defaultdict, Counter
from flask import Flask, render_template, request

# -----------------------------
# Backend functions (same as your code)
# -----------------------------
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()

def build_index(corpus_dir):
    dictionary = {}
    doc_lengths = {}
    docIDs = {}
    id_to_doc = {}

    files = sorted(os.listdir(corpus_dir))
    N = len(files)

    for i, fname in enumerate(files):
        docID = i + 1
        docIDs[fname] = docID
        id_to_doc[docID] = fname

        with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8", errors="ignore") as f:
            tokens = tokenize(f.read())

        tf_counts = Counter(tokens)

        for term, tf in tf_counts.items():
            log_tf = 1 + math.log10(tf)
            if term not in dictionary:
                dictionary[term] = [0, []]
            dictionary[term][0] += 1
            dictionary[term][1].append((docID, log_tf))

    for term, (df, postings) in dictionary.items():
        for docID, weight in postings:
            doc_lengths[docID] = doc_lengths.get(docID, 0) + weight**2

    for docID in doc_lengths:
        doc_lengths[docID] = math.sqrt(doc_lengths[docID])

    return dictionary, doc_lengths, docIDs, id_to_doc, N

def process_query(query, dictionary, doc_lengths, id_to_doc, N, top_k=10):
    tokens = tokenize(query)
    tf_counts = Counter(tokens)

    query_weights = {}
    for term, tf in tf_counts.items():
        if term in dictionary:
            log_tf = 1 + math.log10(tf)
            df = dictionary[term][0]
            idf = math.log10(N / df)
            query_weights[term] = log_tf * idf

    norm = math.sqrt(sum(w**2 for w in query_weights.values()))
    if norm > 0:
        for term in query_weights:
            query_weights[term] /= norm

    scores = defaultdict(float)
    for term, q_weight in query_weights.items():
        if term in dictionary:
            for docID, d_weight in dictionary[term][1]:
                scores[docID] += q_weight * (d_weight / doc_lengths[docID])

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [(id_to_doc[docID], score) for docID, score in ranked[:top_k]]

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# Change this path to your corpus folder
corpus_dir = r"C:\Users\kousi\OneDrive\Desktop\study\new IR project\Corpus"
dictionary, doc_lengths, docIDs, id_to_doc, N = build_index(corpus_dir)

@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form["query"]
        results = process_query(query, dictionary, doc_lengths, id_to_doc, N)
    return render_template("index.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)




