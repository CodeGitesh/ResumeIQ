import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "resumeiq.db")

def init_db():
    """Initialize the SQLite database with the candidates table."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT,
        phone TEXT,
        predicted_role TEXT,
        ats_score INTEGER,
        health_score INTEGER,
        strong_bullets INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company TEXT NOT NULL,
        role TEXT NOT NULL,
        status TEXT NOT NULL,
        url TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def insert_candidate(name, email, phone, role, ats, health, bullets):
    """Insert a new processed candidate into the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO candidates (name, email, phone, predicted_role, ats_score, health_score, strong_bullets)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, email, phone, role, ats, health, bullets))
    
    conn.commit()
    conn.close()

def get_all_candidates():
    """Retrieve all candidates for the HR Dashboard."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM candidates ORDER BY ats_score DESC')
    rows = cursor.fetchall()
    
    # Get column names
    col_names = [description[0] for description in cursor.description]
    
    conn.close()
    
    # Convert to list of dicts
    return [dict(zip(col_names, row)) for row in rows]

def insert_application(company, role, status, url):
    """Insert a new tracked job application."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO applications (company, role, status, url)
    VALUES (?, ?, ?, ?)
    ''', (company, role, status, url))
    
    conn.commit()
    conn.close()

def get_all_applications():
    """Retrieve all tracked applications."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM applications ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    col_names = [description[0] for description in cursor.description]
    conn.close()
    
    return [dict(zip(col_names, row)) for row in rows]

def update_application_status(app_id, new_status):
    """Update the status of a tracked application."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    UPDATE applications
    SET status = ?
    WHERE id = ?
    ''', (new_status, app_id))
    
    conn.commit()
    conn.close()
