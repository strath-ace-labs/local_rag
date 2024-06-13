

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from rank_bm25 import BM25Okapi
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text, stop_words):
    #print(text)
    sentences = sent_tokenize(text)
    tokenized_sentences = []
    for sentence in sentences:

        word_tokens = word_tokenize(sentence)
        filtered_sentence = [w.lower() for w in word_tokens if w.lower() not in stop_words and w.isalnum()]
        tokenized_sentences.append(filtered_sentence)

    return sentences, tokenized_sentences

def find_snippet(docs, queries, num_top_returned=1):
    """
    Find snippets from docs relating to the queries using BM25.
    
    Args:
    - docs (list of str): List of documents, each document is a string.
    - queries (list of str): List of queries, each query is a string.
    - num_top_returned (int): Number of top snippets to return for each query.
    
    Returns:
    - top_snippets (list of list of str): List of lists containing the top snippets for each query.
    - top_scores (list of list of float): List of lists containing the BM25 scores for the top snippets.
    """
    stop_words = set(stopwords.words('english'))
    
    # Preprocess all documents
    all_sentences = []
    tokenized_docs = []
    for doc in docs:
        sentences, tokenized_sentences = preprocess_text(doc, stop_words)
        all_sentences.extend(sentences)
        tokenized_docs.extend(tokenized_sentences)
        # print(len(all_sentences))
    # Initialize BM25
    bm25 = BM25Okapi(tokenized_docs)
    #print(tokenized_docs[0])
    # Process each query

    results = []
    for query in queries:

        query_sentences, tokenized_queries = preprocess_text(query, stop_words)
        for i, tokenized_query in enumerate(tokenized_queries):
            scores = bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[-num_top_returned:][::-1]
            top_snippets = [all_sentences[idx] for idx in top_indices]
            top_scores = [scores[idx] for idx in top_indices]
            results.append({
                'query_sentence': query_sentences[i],
                'top_snippets': top_snippets,
                'top_scores': top_scores,
                'top_indices': top_indices,
                'scores': scores
            })
    
    return results

def generate_interactive_html(docs, results):
    highlighted_html = """
    <style>
        .query-sentence:hover { background-color: blue; cursor: pointer; }
        .highlighted { background-color: green; }
    </style>
    <script>
        function highlightSentences(queryIndex) {
            var sentences = document.querySelectorAll('.doc-sentence');
            sentences.forEach(function(sentence) {
                sentence.style.backgroundColor = ''; // Reset background
            });
            var relevantSentences = document.querySelectorAll('.doc-sentence.query-' + queryIndex);
            relevantSentences.forEach(function(sentence) {
                sentence.style.backgroundColor = sentence.getAttribute('data-score-color');
            });
        }
    </script>
    """

    highlighted_html += "<h3>Answer:</h3><ul>"
    for idx, result in enumerate(results):
        highlighted_html += f"<li class='query-sentence' onmouseover='highlightSentences({idx})'>{result['query_sentence']}</li>"
    highlighted_html += "</ul>"

    highlighted_html += "<h3>Documents:</h3>"

    # all_sentences = []
    # for doc in docs:
        
    #     sentences = sent_tokenize(doc['text'])
    #     all_sentences.extend(sentences)

    global_idx = -1

    for doc_index, doc in enumerate(docs):
        sentences = sent_tokenize(doc['text'])
        highlighted_doc = f"<br> Document Title: {doc['document']}</br>"
        
        for idx, sentence in enumerate(sentences):
            global_idx += 1
            idx = global_idx
            sentence_classes = " ".join([f"query-{result_idx}" for result_idx, result in enumerate(results) if idx in result['top_indices']])
            score_colors = " ".join([f"rgba(50,205,50,{result['scores'][idx] / max(result['scores'])})" for result in results if idx in result['top_indices']])
            highlighted_doc += f"<span class='doc-sentence {sentence_classes}' data-score-color='{score_colors}'>{sentence} </span>"

        highlighted_html += f"<p>{highlighted_doc}</p>"
    
    return highlighted_html


