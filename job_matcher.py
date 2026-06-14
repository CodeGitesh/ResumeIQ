import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

def load_jobs_corpus():
    path = "data/real_jobs_corpus.csv"
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
    # Mathematical Scaling: P95 Normalization - Honest Scaling
    p95 = np.percentile(similarities, 95)
    if p95 < 0.01:
        p95 = 0.01
    normalized = np.clip((similarities / p95) * 100, 0, 99).astype(int)

    # Rank jobs based on similarity
    ranked_indices = np.argsort(normalized)[::-1]
    
    total_jobs = len(jobs)
    results = []
    
    for idx in ranked_indices[:top_n]:
        job_match = jobs[idx].copy()
        
        # Add Honest Metrics
        score = int(normalized[idx])
        rank = np.argsort(np.argsort(-similarities))[idx] + 1
        percentile = int((1 - rank / total_jobs) * 100)
        match_label = "Excellent" if score >= 75 else "Good" if score >= 50 else "Fair"
        
        job_match["match_score"] = score
        job_match["match_label"] = match_label
        job_match["percentile"] = percentile
        results.append(job_match)
        
    return results
