import pandas as pd
import os
import random

def process_datasets():
    print("Starting massive data aggregation...")
    all_jobs = []
    
    base_path = "/Users/giteshgoyal/Desktop/Github"
    
    # 1. Process data job posts.csv (Armenian Jobs)
    try:
        p1 = os.path.join(base_path, "data job posts.csv")
        df1 = pd.read_csv(p1)
        for _, row in df1.dropna(subset=['Title', 'JobDescription']).iterrows():
            all_jobs.append({
                "id": f"ARM_{random.randint(100000, 999999)}",
                "title": str(row.get('Title', 'Unknown')),
                "company": str(row.get('Company', 'Confidential')),
                "location": str(row.get('Location', 'Remote')),
                "salary": str(row.get('Salary', 'Not Disclosed')),
                "description": str(row.get('JobDescription', '')) + " " + str(row.get('JobRequirment', ''))
            })
        print(f"Loaded {len(df1)} jobs from data job posts.csv")
    except Exception as e:
        print(f"Error reading dataset 1: {e}")

    # 2. Process Naukri Jobs
    try:
        p2 = os.path.join(base_path, "marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv")
        df2 = pd.read_csv(p2)
        for _, row in df2.dropna(subset=['Job Title', 'Key Skills']).iterrows():
            all_jobs.append({
                "id": f"NAU_{random.randint(100000, 999999)}",
                "title": str(row.get('Job Title', 'Unknown')),
                "company": "Naukri Listed Company", # Usually Company is omitted in this specific dump or in a weird column
                "location": str(row.get('Location', 'India')),
                "salary": str(row.get('Job Salary', 'Not Disclosed')),
                "description": f"Role: {row.get('Role', 'Unknown')}. Skills required: {row.get('Key Skills', '')}"
            })
        print(f"Loaded {len(df2)} jobs from Naukri sample")
    except Exception as e:
        print(f"Error reading dataset 2: {e}")

    # 3. Process LinkedIn Postings
    try:
        p3 = os.path.join(base_path, "postings.csv")
        # Reading only first 100k to avoid memory crash
        df3 = pd.read_csv(p3, nrows=100000)
        for _, row in df3.dropna(subset=['title', 'description']).iterrows():
            all_jobs.append({
                "id": f"LNK_{row.get('job_id', random.randint(100000, 999999))}",
                "title": str(row.get('title', 'Unknown')),
                "company": str(row.get('company_name', 'Confidential')),
                "location": str(row.get('location', 'Global')),
                "salary": str(row.get('med_salary', 'Not Disclosed')),
                "description": str(row.get('description', ''))
            })
        print(f"Loaded {len(df3)} jobs from LinkedIn postings.csv")
    except Exception as e:
        print(f"Error reading dataset 3: {e}")

    print(f"Total aggregated jobs: {len(all_jobs)}")
    
    # Shuffle and Sample 5000 jobs
    random.shuffle(all_jobs)
    sampled_jobs = all_jobs[:5000]
    
    # Save to clean corpus
    os.makedirs("data", exist_ok=True)
    out_path = "data/real_jobs_corpus.csv"
    pd.DataFrame(sampled_jobs).to_csv(out_path, index=False)
    print(f"Successfully saved {len(sampled_jobs)} sampled jobs to {out_path}")

if __name__ == "__main__":
    process_datasets()
