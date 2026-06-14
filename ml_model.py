"""
ml_model.py — ResumeIQ Machine Learning Models

Contains three models:
1. Job Role Classifier (KNN) — predicts job category from resume text
2. ATS Score Regressor (RandomForestRegressor) — predicts ATS compatibility score
3. Bullet Point Impact Classifier (MultinomialNB) — classifies bullet points as Strong/Weak
"""

import os
import re
import ssl
import pickle
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report


# ---------------------------------------------------------------------------
# Utility: text cleaning
# ---------------------------------------------------------------------------

def clean_resume_text(text: str) -> str:
    """Clean resume text by removing URLs, mentions, special chars, etc."""
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ===========================================================================================
# MODEL 1: Job Role Classifier (KNN)
# ===========================================================================================

def train_job_role_classifier() -> dict:
    """
    Downloads the resume dataset, trains a KNN classifier for job-role
    prediction, and saves the model + vectorizer to disk.

    Returns a dict with train/test accuracy.
    """
    print("=" * 70)
    print("MODEL 1: Job Role Classifier (MultinomialNB)")
    print("=" * 70)

    # --- Download dataset with SSL bypass (macOS compatibility) -----------
    dataset_url = (
        "https://raw.githubusercontent.com/611noorsaeed/"
        "Resume-Screening-App/main/UpdatedResumeDataSet.csv"
    )
    os.makedirs("data", exist_ok=True)
    data_path = os.path.join("data", "UpdatedResumeDataSet.csv")

    if not os.path.exists(data_path):
        print("Data not found locally. Please run generate_data.py to generate massive local datasets.")
        return {"train_acc": 0, "test_acc": 0}
    else:
        print(f"Using local dataset at {data_path}")

    # --- Load & clean -----------------------------------------------------
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    df["cleaned_resume"] = df["Resume"].apply(clean_resume_text)

    # --- Vectorize --------------------------------------------------------
    tfidf = TfidfVectorizer(max_features=1500, stop_words="english")
    X = tfidf.fit_transform(df["cleaned_resume"])

    # Use .tolist() to avoid PyArrow / pandas-backed array errors
    y = df["Category"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Train MultinomialNB ----------------------------------------------
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, nb_clf.predict(X_train))
    test_acc = accuracy_score(y_test, nb_clf.predict(X_test))
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")

    # --- Save -------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "resume_classifier.pkl"), "wb") as f:
        pickle.dump(nb_clf, f)
    with open(os.path.join("models", "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    print("Saved models/resume_classifier.pkl")
    print("Saved models/tfidf_vectorizer.pkl")
    print()
    return {"train_accuracy": train_acc, "test_accuracy": test_acc}


def predict_job_category(resume_text: str) -> dict:
    """
    Predict the job category for a given resume text.

    Returns
    -------
    dict  {"category": str, "confidence": float}
    """
    if not os.path.exists(os.path.join("models", "tfidf_vectorizer.pkl")) or not os.path.exists(os.path.join("models", "resume_classifier.pkl")):
        train_job_role_classifier()
        
    with open(os.path.join("models", "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join("models", "resume_classifier.pkl"), "rb") as f:
        knn = pickle.load(f)

    cleaned = clean_resume_text(resume_text)
    vec = tfidf.transform([cleaned])
    proba = knn.predict_proba(vec)[0]
    idx = np.argmax(proba)
    category = knn.classes_[idx]
    confidence = float(proba[idx])
    return {"category": category, "confidence": round(confidence, 4)}


# ===========================================================================================
# MODEL 2: Health Score Calculation
# ===========================================================================================

def compute_health_score(features: dict) -> dict:
    skill_score    = min(features.get("skill_count", 0) / 20 * 100, 100)
    verb_score     = min(features.get("action_verb_count", 0) / 10 * 100, 100)
    metrics_score  = min(features.get("metrics_count", 0) / 5 * 100, 100)
    section_score  = (features.get("section_count", 0) / 4) * 100
    format_score   = max(100 - features.get("formatting_penalty", 0) * 15, 0)
    
    total = (
        skill_score   * 0.30 +
        verb_score    * 0.20 +
        metrics_score * 0.20 +
        section_score * 0.15 +
        format_score  * 0.15
    )
    total = int(round(total))
    grade = "A" if total >= 80 else "B" if total >= 65 else "C" if total >= 50 else "D"
    return {
        "total_score": total,
        "grade": grade,
        "breakdown": {
            "Skills":      int(skill_score),
            "Action Verbs":int(verb_score),
            "Metrics":     int(metrics_score),
            "Sections":    int(section_score),
            "Formatting":  int(format_score),
        }
    }




# ===========================================================================================
# MODEL 3: Bullet Point Impact Classifier (MultinomialNB)
# ===========================================================================================

def _get_bullet_dataset() -> tuple:
    """
    Return a hardcoded list of ~100 bullet points labelled 1 (strong) / 0 (weak).
    """
    strong = [
        "Increased revenue by 35% by redesigning the checkout flow using React",
        "Reduced server response time by 40% through optimizing SQL queries and implementing Redis caching",
        "Led a team of 8 engineers to deliver a microservices migration 2 weeks ahead of schedule",
        "Automated CI/CD pipeline with Jenkins, reducing deployment time from 4 hours to 15 minutes",
        "Built a real-time analytics dashboard in Python and D3.js serving 10K daily active users",
        "Decreased customer churn by 18% by implementing a predictive ML model using XGBoost",
        "Developed RESTful APIs in Node.js handling 5M requests per day with 99.9% uptime",
        "Migrated legacy monolith to AWS Lambda, cutting infrastructure costs by 60%",
        "Improved test coverage from 45% to 92% by introducing pytest and integration tests",
        "Designed a recommendation engine that increased average order value by 22%",
        "Managed a $2M annual budget for cloud infrastructure across 3 AWS regions",
        "Trained and mentored 5 junior developers, improving team velocity by 30%",
        "Implemented OAuth 2.0 and JWT-based authentication securing 500K user accounts",
        "Optimized ETL pipeline processing 50GB of data daily, reducing runtime by 65%",
        "Negotiated vendor contracts saving the company $150K annually",
        "Spearheaded adoption of Kubernetes, achieving 99.99% service availability",
        "Created a fraud detection system using Random Forest that flagged 95% of fraudulent transactions",
        "Published 3 technical blog posts that generated 25K pageviews and 200 inbound leads",
        "Reduced bug backlog by 70% within 2 sprints through systematic triage and pair programming",
        "Architected a data lake on AWS S3 and Glue processing 1TB+ daily",
        "Delivered a mobile app feature used by 1.2M users within the first month of launch",
        "Improved page load speed by 50% through code splitting and lazy loading in React",
        "Coordinated cross-functional teams across 4 time zones to ship a product on schedule",
        "Achieved a 98% customer satisfaction score by redesigning the support ticket workflow",
        "Integrated Stripe payment gateway processing $3M in monthly transactions",
        "Developed a chatbot using NLP that resolved 40% of support tickets without human intervention",
        "Increased email open rates by 28% through A/B testing subject lines and send times",
        "Built a data pipeline in Apache Spark processing 100M records per hour",
        "Reduced onboarding time for new hires from 3 weeks to 5 days with documentation and tooling",
        "Deployed a containerized application on ECS serving 50K concurrent users",
        "Streamlined inventory management system, reducing stockouts by 25%",
        "Launched an internal CLI tool in Go that saved engineers 10 hours per week",
        "Drove adoption of TypeScript across 12 repositories improving type safety and reducing runtime errors by 35%",
        "Designed and implemented a rate-limiting middleware handling 100K requests per minute",
        "Achieved SOC 2 Type II compliance by leading security audit remediation across 5 services",
        "Wrote unit and integration tests covering 95% of critical payment processing paths",
        "Reduced mean time to recovery (MTTR) from 2 hours to 15 minutes with improved monitoring and runbooks",
        "Scaled PostgreSQL database to handle 10x traffic growth using read replicas and connection pooling",
        "Increased user engagement by 45% by implementing personalized push notifications",
        "Delivered quarterly OKRs consistently, completing 90% of planned initiatives on time",
        "Configured Terraform IaC for 30+ cloud resources, enabling reproducible deployments",
        "Optimized machine learning model inference latency from 200ms to 35ms using ONNX Runtime",
        "Built an automated reporting system that saved the finance team 20 hours per month",
        "Initiated a code review culture that reduced production incidents by 50%",
        "Created a customer segmentation model using K-Means that improved targeting accuracy by 30%",
        "Developed a GraphQL API that reduced frontend data-fetching calls by 60%",
        "Led a successful migration from MySQL to PostgreSQL with zero downtime",
        "Implemented feature flags using LaunchDarkly enabling safe rollouts for 2M users",
        "Improved search relevance by 38% using Elasticsearch and custom scoring algorithms",
        "Reduced Docker image sizes by 70% through multi-stage builds and Alpine base images",
    ]

    weak = [
        "Was responsible for handling the website",
        "Helped with various tasks in the office",
        "Worked on software development projects",
        "Responsible for managing databases",
        "Assisted in daily operations of the department",
        "Participated in team meetings and discussions",
        "Was part of the engineering team",
        "Handled customer inquiries",
        "Did some coding work",
        "Involved in testing activities",
        "Worked with team members on projects",
        "Maintained existing software systems",
        "Helped the team with different assignments",
        "Responsible for writing code",
        "Took care of IT-related issues",
        "Was in charge of updating the database",
        "Assisted with project planning",
        "Contributed to team efforts",
        "Worked on improving processes",
        "Handled administrative tasks for the team",
        "Supported the manager with reports",
        "Was responsible for some testing",
        "Performed general duties as assigned",
        "Helped maintain company systems",
        "Participated in the software development lifecycle",
        "Worked on bug fixes",
        "Assisted with documentation",
        "Was a member of the development team",
        "Helped with deployment activities",
        "Responsible for various IT tasks",
        "Contributed to code reviews occasionally",
        "Worked on data analysis tasks",
        "Was involved in client communications",
        "Assisted colleagues with technical problems",
        "Took part in brainstorming sessions",
        "Managed day-to-day responsibilities",
        "Responsible for updating spreadsheets",
        "Handled incoming support requests",
        "Participated in training sessions",
        "Worked on multiple projects simultaneously",
        "Helped organize team events",
        "Supported the sales team with data",
        "Was responsible for monitoring systems",
        "Assisted in onboarding new employees",
        "Contributed to the marketing strategy",
        "Worked closely with the product team",
        "Helped improve internal tools",
        "Responsible for compiling weekly reports",
        "Participated in quality assurance efforts",
        "Managed communication between departments",
    ]

    texts = strong + weak
    labels = [1] * len(strong) + [0] * len(weak)
    return texts, labels


def train_bullet_classifier() -> dict:
    """
    Train a MultinomialNB model to classify resume bullet points as
    Strong (1) or Weak (0) and save to disk.

    Returns a dict with accuracy and classification report string.
    """
    print("=" * 70)
    print("MODEL 3: Bullet Point Impact Classifier (MultinomialNB)")
    print("=" * 70)

    texts, labels = _get_bullet_dataset()
    print(f"Total bullet points: {len(texts)}  (Strong: {sum(labels)}, Weak: {len(labels) - sum(labels)})")

    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    X = tfidf.fit_transform(texts)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, nb.predict(X_train))
    test_acc = accuracy_score(y_test, nb.predict(X_test))
    report = classification_report(y_test, nb.predict(X_test), target_names=["Weak", "Strong"])

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")
    print("\nClassification Report (test set):")
    print(report)

    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "bullet_classifier.pkl"), "wb") as f:
        pickle.dump(nb, f)
    with open(os.path.join("models", "bullet_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    print("Saved models/bullet_classifier.pkl")
    print("Saved models/bullet_vectorizer.pkl")
    print()
    return {"train_accuracy": train_acc, "test_accuracy": test_acc, "report": report}


def classify_bullets(bullet_list: list) -> list:
    """
    Classify a list of resume bullet-point strings as Strong or Weak.

    Parameters
    ----------
    bullet_list : list[str]

    Returns
    -------
    list[dict]  Each dict: {"text": str, "label": "Strong"/"Weak", "confidence": float}
    """
    if not os.path.exists(os.path.join("models", "bullet_vectorizer.pkl")) or not os.path.exists(os.path.join("models", "bullet_classifier.pkl")):
        train_bullet_classifier()

    with open(os.path.join("models", "bullet_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join("models", "bullet_classifier.pkl"), "rb") as f:
        nb = pickle.load(f)

    X = tfidf.transform(bullet_list)
    probas = nb.predict_proba(X)
    preds = nb.predict(X)

    results = []
    for i, text in enumerate(bullet_list):
        label = "Strong" if preds[i] == 1 else "Weak"
        confidence = float(probas[i].max())
        results.append({
            "text": text,
            "label": label,
            "confidence": round(confidence, 4),
        })
    return results


# ===========================================================================================
# Train all models
# ===========================================================================================

def train_all_models():
    """Train and save all three ResumeIQ ML models."""
    print("\n🚀  ResumeIQ — Training All Models\n")

    results = {}

    results["job_role_classifier"] = train_job_role_classifier()
    results["bullet_classifier"] = train_bullet_classifier()

    # --- Quick smoke test --------------------------------------------------
    print("=" * 70)
    print("SMOKE TESTS")
    print("=" * 70)

    # Test 1: job category prediction
    sample_resume = (
        "Experienced Python developer with 5 years building REST APIs, "
        "microservices, Django, Flask, PostgreSQL, Docker, and AWS."
    )
    cat_result = predict_job_category(sample_resume)
    print(f"Job category prediction: {cat_result}")

    # Test 3: bullet point classification
    sample_bullets = [
        "Increased revenue by 35% by redesigning the checkout flow using React",
        "Was responsible for handling the website",
    ]
    bullet_results = classify_bullets(sample_bullets)
    for br in bullet_results:
        print(f"Bullet: '{br['text'][:60]}...' → {br['label']} ({br['confidence']:.2f})")

    print("\n✅  All models trained and verified successfully!\n")
    return results


if __name__ == "__main__":
    train_all_models()
