import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def load_jobs_corpus():
    path = "data/indian_jobs_corpus.csv"
    if os.path.exists(path):
        return pd.read_csv(path).to_dict('records')
    return []

def recommend_jobs(resume_text: str, top_n: int = 4) -> list:
    """
    Uses TF-IDF and Cosine Similarity to find the best matching jobs 
    from the massive Indian Jobs Corpus for a given resume.
    """
    if not resume_text.strip():
        return []

    jobs = load_jobs_corpus()
    if not jobs: return []

    # Prepare corpus: Job Descriptions + the Resume
    job_texts = [job["description"] for job in jobs]
    corpus = job_texts + [resume_text]

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # The resume is the last row, the jobs are the rows before it
    resume_vector = tfidf_matrix[-1]
    job_vectors = tfidf_matrix[:-1]

    # Compute similarity between resume and all jobs
    similarities = cosine_similarity(resume_vector, job_vectors).flatten()
    
    # Mathematical Scaling: Min-Max Normalization to naturally curve scores
    scaler = MinMaxScaler(feature_range=(40, 95))
    scaled_similarities = scaler.fit_transform(similarities.reshape(-1, 1)).flatten()

    # Rank jobs based on similarity
    ranked_indices = np.argsort(scaled_similarities)[::-1]
    
    results = []
    for idx in ranked_indices[:top_n]:
        job_match = jobs[idx].copy()
        # Use mathematical scaling instead of hardcoded spoofing
        match_score = int(scaled_similarities[idx])
        job_match["match_score"] = min(match_score, 99)
        results.append(job_match)
        
    return results
