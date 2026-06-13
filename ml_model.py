import os
import re
import urllib.request
import ssl
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

DATA_URL = "https://raw.githubusercontent.com/611noorsaeed/Resume-Screening-App/main/UpdatedResumeDataSet.csv"
DATA_PATH = "data/UpdatedResumeDataSet.csv"
MODEL_PATH = "models/resume_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

def download_dataset():
    """Downloads the Kaggle Resume Dataset if it doesn't exist."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading Kaggle Resume Dataset...")
        
        # Bypass MacOS SSL certificate errors
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(DATA_URL, context=ctx) as response, open(DATA_PATH, 'wb') as out_file:
            out_file.write(response.read())
            
        print("Download complete.")

def clean_resume_text(text):
    """Basic text cleaning for the NLP model."""
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('#\S+', '', text)  # remove hashtags
    text = re.sub('@\S+', '  ', text)  # remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    return text.lower()

def train_and_save_model():
    """Trains the NLP model on the Kaggle dataset."""
    download_dataset()
    
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print("Cleaning text...")
    df['cleaned_resume'] = df['Resume'].apply(clean_resume_text)
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=1500)
    X = vectorizer.fit_transform(df['cleaned_resume'])
    
    y = df['Category'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
    
    print("Training KNeighborsClassifier...")
    model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model trained successfully! Validation Accuracy: {acc*100:.2f}%")
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
        
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
        
    return model, vectorizer

def load_models():
    """Loads the trained model and vectorizer."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return train_and_save_model()
    
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading models: {e}. Retraining...")
        return train_and_save_model()

def predict_job_category(resume_text: str) -> dict:
    """Predicts the job category of the given resume."""
    model, vectorizer = load_models()
    
    cleaned_text = clean_resume_text(resume_text)
    features = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(features)[0]
    
    # Get probabilities
    proba = model.predict_proba(features)[0]
    confidence = max(proba) * 100
    
    return {
        "category": prediction,
        "confidence": confidence
    }

if __name__ == "__main__":
    train_and_save_model()
