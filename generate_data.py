import pandas as pd
import random
import os

os.makedirs('data', exist_ok=True)

# 1. Generate Massive Indian Jobs Dataset (500 Jobs)
indian_cities = ["Bangalore", "Pune", "Hyderabad", "Mumbai", "Chennai", "Gurugram", "Noida", "Remote"]
companies = ["TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra", "Flipkart", "Zomato", "Swiggy", "Paytm", "Ola", "Freshworks", "Zoho"]
roles = ["Software Engineer", "Data Scientist", "Frontend Developer", "Backend Developer", "DevOps Engineer", "Machine Learning Engineer", "Product Manager"]
tech_stacks = [
    "Python, Django, PostgreSQL, AWS",
    "React, Node.js, Express, MongoDB",
    "Java, Spring Boot, Microservices, MySQL",
    "Python, TensorFlow, PyTorch, Scikit-learn",
    "Docker, Kubernetes, Terraform, AWS CI/CD",
    "Angular, TypeScript, HTML, CSS",
    "C++, Data Structures, Algorithms, Linux"
]

jobs_data = []
for i in range(1, 501):
    role = random.choice(roles)
    company = random.choice(companies)
    city = random.choice(indian_cities)
    tech = random.choice(tech_stacks)
    salary = f"₹{random.randint(5, 30)}LPA"
    desc = f"We are hiring a {role} at {company}. The candidate must have strong experience in {tech}. Responsibilities include building scalable systems, working in Agile, and strong communication skills. Location: {city}."
    
    jobs_data.append({
        "id": f"IND{i}",
        "title": role,
        "company": company,
        "location": city,
        "salary": salary,
        "description": desc
    })

pd.DataFrame(jobs_data).to_csv('data/indian_jobs_corpus.csv', index=False)
print("Created data/indian_jobs_corpus.csv (500 rows)")

# 2. Generate Massive Resume Dataset (1000 Resumes)
resume_data = []
for i in range(1, 1001):
    category = random.choice(roles)
    tech = random.choice(tech_stacks)
    exp = random.randint(1, 15)
    resume_text = f"Experienced {category} with {exp} years of experience in the tech industry. Highly skilled in {tech}. Delivered multiple projects on time. Strong background in software engineering, algorithms, and agile methodologies. Graduated with a B.Tech in Computer Science."
    resume_data.append({
        "Category": category,
        "Resume": resume_text
    })

pd.DataFrame(resume_data).to_csv('data/UpdatedResumeDataSet.csv', index=False)
print("Created data/UpdatedResumeDataSet.csv (1000 rows)")
