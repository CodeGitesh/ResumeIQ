import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text_utils import preprocess, identity_tokenizer

def load_jobs_corpus():
    path = "data/real_jobs_corpus.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df['description'].str.len() > 50]
        return df.to_dict('records')
    return []

def recommend_jobs(resume_text: str, top_n: int = 10) -> list:
    if not resume_text.strip():
        return []

    jobs = load_jobs_corpus()
    if not jobs:
        return []

    job_texts = [job["description"] for job in jobs]
    corpus = job_texts + [resume_text]

    # Preprocess corpus with unified tokenizer
    processed_corpus = [preprocess(text) for text in corpus]

    vectorizer = TfidfVectorizer(
        tokenizer=identity_tokenizer,
        preprocessor=identity_tokenizer,
        token_pattern=None,
        max_features=5000,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(processed_corpus)

    resume_vector = tfidf_matrix[-1]
    job_vectors   = tfidf_matrix[:-1]

    similarities = cosine_similarity(resume_vector, job_vectors).flatten()

    # Percentile rank within non-zero matches only
    # Avoids P95 collapse when 80%+ jobs are unrelated to resume
    nonzero_sims = similarities[similarities > 0.001]

    # To prevent clustering at 99%, we use Baseline Max-Relative Scaling.
    # We find the top score. If the top score is garbage (<0.15), we use 0.15 as the baseline 
    # to prevent a terrible 0.04 match from inflating to 95%.
    max_raw = similarities.max()
    baseline = max(max_raw, 0.15)

    ranked_indices = np.argsort(similarities)[::-1]
    total_jobs     = len(jobs)

    results = []
    for rank_pos, idx in enumerate(ranked_indices[:top_n]):
        raw       = float(similarities[idx])
        job_match = jobs[idx].copy()

        # Scale relative to the baseline. Top job gets ~96%.
        scaled_score = int((raw / baseline) * 96)
        
        # Cap at 99 and floor at 0
        final_score = min(max(scaled_score, 0), 99)

        corpus_rank = rank_pos + 1
        corpus_pct  = int((1 - corpus_rank / total_jobs) * 100)

        match_label = (
            "Excellent" if final_score >= 80 else
            "Good"      if final_score >= 50 else
            "Fair"
        )
        
        # Explainability: Matched vs Missing Skills
        common_tech = ["python", "java", "c++", "c#", "sql", "aws", "docker", "kubernetes", 
                       "react", "angular", "node", "javascript", "html", "css", "machine learning",
                       "deep learning", "nlp", "django", "flask", "spring", "agile", "scrum", "git",
                       "linux", "bash", "rest", "api", "mongodb", "mysql", "postgresql", "ruby"]
        
        job_desc_lower = str(job_match.get("description", "")).lower()
        resume_lower = resume_text.lower()
        
        job_skills = [s for s in common_tech if s in job_desc_lower]
        matched_skills = [s for s in job_skills if s in resume_lower]
        missing_skills = [s for s in job_skills if s not in resume_lower]

        job_match["match_score"] = final_score
        job_match["match_label"] = match_label
        job_match["matched_skills"] = matched_skills
        job_match["missing_skills"] = missing_skills
        job_match["percentile"]  = corpus_pct
        job_match["raw_score"]   = round(raw, 5)
        job_match["pool_size"]   = len(nonzero_sims)
        job_match["corpus_p95"]  = round(float(np.percentile(similarities, 95)), 5)

        results.append(job_match)

    return results
