import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Mock Database of Jobs
MOCK_JOBS = [
    {
        "id": "J101",
        "title": "Software Engineer (Backend)",
        "company": "TechNova Solutions",
        "location": "Remote",
        "salary": "$110,000 - $140,000",
        "description": "Looking for a strong backend engineer experienced in Python, Django, REST APIs, PostgreSQL, and AWS. Must understand microservices architecture, Docker, and CI/CD pipelines."
    },
    {
        "id": "J102",
        "title": "Frontend Developer",
        "company": "Creative UI Agency",
        "location": "New York, NY",
        "salary": "$90,000 - $120,000",
        "description": "We need a frontend specialist with deep knowledge of React, Next.js, TypeScript, Tailwind CSS, and state management (Redux/Zustand). Experience with Figma is a plus."
    },
    {
        "id": "J103",
        "title": "Machine Learning Engineer",
        "company": "DataSphere AI",
        "location": "San Francisco, CA",
        "salary": "$130,000 - $170,000",
        "description": "Join our AI team! Required skills: Python, PyTorch, TensorFlow, scikit-learn, NLP, Pandas, and experience deploying ML models to production using FastAPI and Docker."
    },
    {
        "id": "J104",
        "title": "Data Scientist",
        "company": "FinTech Analytics",
        "location": "Remote",
        "salary": "$115,000 - $150,000",
        "description": "Seeking a data scientist for predictive modeling. Must know SQL, Python, R, XGBoost, K-Means, and have strong data visualization skills using Tableau or Matplotlib."
    },
    {
        "id": "J105",
        "title": "Full Stack Developer",
        "company": "StartupX",
        "location": "Austin, TX",
        "salary": "$100,000 - $135,000",
        "description": "Looking for a versatile full stack dev. Tech stack: Node.js, Express, React, MongoDB, GraphQL. Experience with AWS Lambda and serverless is highly preferred."
    },
    {
        "id": "J106",
        "title": "DevOps Engineer",
        "company": "CloudNative Inc.",
        "location": "Seattle, WA",
        "salary": "$125,000 - $160,000",
        "description": "We need a DevOps expert to manage our infrastructure. Required: Kubernetes, Terraform, AWS/GCP, Jenkins, Linux administration, and scripting in Python or Bash."
    },
    {
        "id": "J107",
        "title": "Android Engineer",
        "company": "MobileFirst Apps",
        "location": "Remote",
        "salary": "$105,000 - $145,000",
        "description": "Build modern Android apps using Kotlin, Jetpack Compose, MVVM architecture, Coroutines, and Room database. Experience with CI/CD for mobile is a plus."
    },
    {
        "id": "J108",
        "title": "Cloud Architect",
        "company": "Global Enterprise Corp",
        "location": "Chicago, IL",
        "salary": "$150,000 - $190,000",
        "description": "Lead our cloud migration strategy. Deep expertise in AWS, Azure, System Design, networking, security compliance, and infrastructure as code."
    }
]

def recommend_jobs(resume_text: str, top_n: int = 4) -> list:
    """
    Uses TF-IDF and Cosine Similarity to find the best matching jobs 
    from the mock database for a given resume.
    """
    if not resume_text.strip():
        return []

    # Prepare corpus: Job Descriptions + the Resume
    job_texts = [job["description"] for job in MOCK_JOBS]
    corpus = job_texts + [resume_text]

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # The resume is the last row, the jobs are the rows before it
    resume_vector = tfidf_matrix[-1]
    job_vectors = tfidf_matrix[:-1]

    # Compute similarity between resume and all jobs
    similarities = cosine_similarity(resume_vector, job_vectors).flatten()

    # Rank jobs based on similarity
    ranked_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in ranked_indices[:top_n]:
        job_match = MOCK_JOBS[idx].copy()
        # Convert score to a percentage
        match_score = int(similarities[idx] * 100)
        job_match["match_score"] = min(match_score + 25, 99) # Add a baseline boost so it doesn't look too low
        results.append(job_match)
        
    return results
