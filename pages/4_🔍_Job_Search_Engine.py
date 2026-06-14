import streamlit as st
import json
import os
import re
from text_utils import preprocess, identity_tokenizer
from rank_bm25 import BM25Okapi
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="Academic Search Engine", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .search-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .algo-badge {
        font-size: 12px;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 4px;
        color: white;
        margin-bottom: 10px;
        display: inline-block;
    }
    .tfidf-badge { background-color: #3B82F6; }
    .bm25-badge { background-color: #8B5CF6; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Academic Job Search Engine")
st.markdown("Evaluate core Information Retrieval algorithms (TF-IDF vs BM25) on the real-world dataset.")

@st.cache_data
def load_data():
    index_path = "data/inverted_index.json"
    corpus_path = "data/real_jobs_corpus.csv"
    
    if not os.path.exists(index_path) or not os.path.exists(corpus_path):
        return None, None
        
    with open(index_path, "r") as f:
        inverted_index = json.load(f)
        
    df = pd.read_csv(corpus_path)
    df = df[df['description'].str.len() > 50]
    jobs_dict = {str(row['id']): row.to_dict() for _, row in df.iterrows()}
    
    return inverted_index, jobs_dict

inverted_index, jobs_dict = load_data()

if inverted_index is None:
    st.error("Inverted Index not found. Please run `build_index.py` first.")
    st.stop()

query = st.text_input("Enter search query (e.g., 'Python Backend Developer AWS')", value="")

if st.button("Search", type="primary") and query.strip():
    start_time = time.time()
    
    # 1. Tokenize & Stem Query using Unified Preprocessor
    query_stems = preprocess(query)
    
    # 2. Boolean OR Retrieval from Inverted Index
    candidate_doc_ids = set()
    for stem in query_stems:
        if stem in inverted_index:
            candidate_doc_ids.update(inverted_index[stem])
            
    candidate_docs = [jobs_dict[doc_id] for doc_id in candidate_doc_ids if doc_id in jobs_dict]
    
    retrieval_time = time.time() - start_time
    
    if not candidate_docs:
        st.warning(f"No documents found containing the query terms. (Retrieved in {retrieval_time:.4f}s)")
    else:
        st.success(f"Retrieved {len(candidate_docs)} candidate documents in {retrieval_time:.4f}s")
        
        # 3. Ranking Comparisons
        docs_texts = [str(doc.get('description', '')) for doc in candidate_docs]
        
        # Preprocess query and corpus once
        processed_query = preprocess(query)
        processed_corpus = [preprocess(text) for text in docs_texts]
        
        # --- TF-IDF ---
        tfidf_start = time.time()
        vectorizer = TfidfVectorizer(
            tokenizer=identity_tokenizer,
            preprocessor=identity_tokenizer,
            token_pattern=None,
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform(processed_corpus + [processed_query])
        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]
        tfidf_scores = cosine_similarity(query_vec, doc_vecs).flatten()
        tfidf_time = time.time() - tfidf_start
        
        # --- BM25 ---
        bm25_start = time.time()
        bm25 = BM25Okapi(processed_corpus)
        bm25_scores = bm25.get_scores(processed_query)
        bm25_time = time.time() - bm25_start
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### TF-IDF Ranking (Took {tfidf_time:.4f}s)")
            tfidf_top_indices = tfidf_scores.argsort()[::-1][:5]
            for rank, idx in enumerate(tfidf_top_indices):
                doc = candidate_docs[idx]
                score = tfidf_scores[idx]
                st.markdown(f"""
                <div class='search-card'>
                    <div class='algo-badge tfidf-badge'>TF-IDF</div>
                    <span style='float:right; font-weight:bold; color:#3B82F6;'>Score: {score:.4f}</span>
                    <h4 style='margin:0;'>{rank+1}. {doc['title']}</h4>
                    <p style='margin:0; font-size:12px; color:#64748B;'>{doc['company']} • {doc['location']}</p>
                    <p style='font-size:14px; margin-top:10px;'>{str(doc['description'])[:200]}...</p>
                </div>
                """, unsafe_allow_html=True)
                
        with col2:
            st.markdown(f"### BM25 Ranking (Took {bm25_time:.4f}s)")
            bm25_top_indices = bm25_scores.argsort()[::-1][:5]
            for rank, idx in enumerate(bm25_top_indices):
                doc = candidate_docs[idx]
                score = bm25_scores[idx]
                st.markdown(f"""
                <div class='search-card'>
                    <div class='algo-badge bm25-badge'>BM25</div>
                    <span style='float:right; font-weight:bold; color:#8B5CF6;'>Score: {score:.4f}</span>
                    <h4 style='margin:0;'>{rank+1}. {doc['title']}</h4>
                    <p style='margin:0; font-size:12px; color:#64748B;'>{doc['company']} • {doc['location']}</p>
                    <p style='font-size:14px; margin-top:10px;'>{str(doc['description'])[:200]}...</p>
                </div>
                """, unsafe_allow_html=True)
