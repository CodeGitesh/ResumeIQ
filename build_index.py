import pandas as pd
import re
import json
from collections import defaultdict
from nltk.stem import PorterStemmer

def build_inverted_index(csv_path: str):
    """
    Reads the dataset and builds a classic Information Retrieval Inverted Index.
    Maps: Term -> List of Document IDs containing that term.
    """
    print(f"Loading corpus from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Run generate_data.py first.")
        return

    inverted_index = defaultdict(list)
    stemmer = PorterStemmer()
    
    # Expanded common stop words
    stop_words = {"and", "the", "to", "of", "in", "a", "is", "for", "with", "on", 
                  "are", "we", "you", "this", "our", "at", "as", "by", "or", "an"}

    for idx, row in df.iterrows():
        doc_id = row['id']
        text = str(row['description']).lower()
        
        # Tokenize (extract alphanumeric words)
        words = re.findall(r'\b[a-z0-9]+\b', text)
        
        # Remove duplicates per document so doc_id only appears once per term
        unique_words = set(words)
        
        for word in unique_words:
            if word not in stop_words and len(word) > 2:
                stemmed_word = stemmer.stem(word)
                inverted_index[stemmed_word].append(doc_id)

    print(f"\nSuccessfully indexed {len(df)} documents.")
    print(f"Total unique terms in index: {len(inverted_index)}\n")
    
    with open("data/inverted_index.json", "w") as f:
        json.dump(inverted_index, f)
    print("Saved inverted index to data/inverted_index.json")
    
    # Print a sample of the inverted index
    print("="*50)
    print("INVERTED INDEX SAMPLE (Term -> Document IDs)")
    print("="*50)
    
    # Show top 15 terms alphabetically
    sample_terms = sorted(list(inverted_index.keys()))[:15]
    for term in sample_terms:
        docs = inverted_index[term]
        # Show only first 5 doc IDs for readability if there are many
        docs_str = ", ".join(docs[:5]) + ("..." if len(docs) > 5 else "")
        print(f"Term: '{term:<15}' -> Docs: [{docs_str}] (Total: {len(docs)})")

if __name__ == "__main__":
    build_inverted_index("data/real_jobs_corpus.csv")
