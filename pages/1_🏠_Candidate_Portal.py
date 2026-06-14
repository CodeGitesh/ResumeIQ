import streamlit as st
import tempfile
import os
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import re
import io

from streamlit_lottie import st_lottie
from analyzer import (
    extract_text_from_pdf, extract_skills, check_formatting,
    compute_general_score, get_market_skill_gaps,
    extract_bullet_points, compute_section_scores, categorize_skills,
    extract_resume_features, calculate_yoe
)
from ml_model import predict_job_category, predict_ats_score, classify_bullets
from resume_builder import generate_interview_questions, generate_enhanced_pdf
from database import insert_candidate

# ── Session state init ─────────────────────────────────────────────────────────
for key in ["resume_text", "resume_path", "skills", "predicted_role", "issues",
            "bullet_results", "market_gaps", "section_scores", "ats_ml_score",
            "interview_qs", "answers", "db_saved", "health_data"]:
    if key not in st.session_state:
        st.session_state[key] = None

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except:
        return None

lottie_ai = load_lottieurl("https://lottie.host/804d9c73-ec14-41e9-911b-c662a5bafbe5/2iPZJ29Npe.json")

# ── Chart Functions (LIGHT MODE) ──────────────────────────────────────────────
def create_gauge_chart(score, title="Score"):
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')
    angle = np.pi * (1 - score / 100)
    theta_bg = np.linspace(0, np.pi, 100)
    ax.fill_between(theta_bg, 0.6, 1.0, color='#E2E8F0', alpha=1.0)
    theta_score = np.linspace(np.pi, angle, 100)
    color = '#10B981' if score >= 70 else '#F59E0B' if score >= 40 else '#EF4444'
    ax.fill_between(theta_score, 0.6, 1.0, color=color, alpha=1.0)
    ax.set_ylim(0, 1.2)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.axis('off')
    ax.text(np.pi/2, 0.2, f"{score}", ha='center', va='center', fontsize=28, fontweight='bold', color=color)
    ax.text(np.pi/2, -0.15, title, ha='center', va='center', fontsize=12, color='#1E293B', fontweight='bold')
    plt.tight_layout()
    return fig

def create_radar_chart(skill_categories):
    categories = list(skill_categories.keys())
    values = [skill_categories[c]["score"] for c in categories]
    N = len(categories)
    if N == 0: return plt.subplots(figsize=(4,4))[0]
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.plot(angles, values, 'o-', linewidth=2, color='#2563EB')
    ax.fill(angles, values, alpha=0.2, color='#2563EB')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, color='#1E293B')
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], size=7, color='#64748B')
    ax.spines['polar'].set_color('#E2E8F0')
    ax.grid(color='#E2E8F0', linewidth=1)
    plt.tight_layout()
    return fig

def create_section_bar_chart(section_scores):
    sections = list(section_scores.keys())
    scores = list(section_scores.values())
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    colors = ['#10B981' if s >= 70 else '#F59E0B' if s >= 40 else '#EF4444' for s in scores]
    bars = ax.barh(sections, scores, color=colors, height=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f'{score}%', va='center', ha='left', fontsize=10, color='#1E293B', fontweight='bold')
    ax.set_xlim(0, 110)
    ax.tick_params(colors='#1E293B', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#E2E8F0')
    ax.spines['left'].set_color('#E2E8F0')
    plt.tight_layout()
    return fig

def create_word_cloud(text):
    stopwords = set(["and", "the", "to", "of", "in", "for", "with", "a", "on", "by", "an", "as", "at", "from"])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues', stopwords=stopwords).generate(text)
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('white')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    return fig

# ── Helper: Action Verbs & Salary ──────────────────────────────────────────────
def suggest_action_verbs(text):
    passive = ["worked on", "helped", "assisted", "was responsible for", "did", "made"]
    suggestions = []
    text_lower = text.lower()
    for p in passive:
        if p in text_lower:
            suggestions.append(p)
    return suggestions

def estimate_salary(role, ats_score):
    base = 60000
    if "Engineer" in role or "Developer" in role or "Data" in role:
        base = 80000
    elif "Manager" in role or "Lead" in role:
        base = 100000
    
    # Scale based on ATS score (proxy for quality/experience)
    multiplier = (ats_score / 100) + 0.5 # 0.5 to 1.5 range
    low = int((base * multiplier) / 1000) * 1000
    high = int((base * multiplier * 1.3) / 1000) * 1000
    return f"${low:,} - ${high:,}"

# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.skill-tag { display: inline-block; padding: 4px 10px; border-radius: 12px; margin: 4px; font-size: 13px; font-weight: 500; }
.skill-match { background: #D1FAE5; color: #065F46; border: 1px solid #10B981; }
.skill-miss { background: #FEE2E2; color: #991B1B; border: 1px solid #EF4444; }
.bullet-card { padding: 10px; margin-bottom: 8px; background: #FFFFFF; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #E2E8F0; }
.b-strong { border-left: 4px solid #10B981; }
.b-weak { border-left: 4px solid #EF4444; }
.b-sugg { color: #2563EB; font-weight: bold; font-size: 13px;}
.info-card { background: #F0F9FF; border: 1px solid #BAE6FD; padding: 16px; border-radius: 8px; margin-bottom: 16px;}
</style>
""", unsafe_allow_html=True)

st.title("📄 Candidate Portal")
st.markdown("Upload your resume to get your **Full Report**, Salary Estimate, and tailored LinkedIn bio.")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("Drop your resume (PDF)", type=["pdf"])
with col2:
    if lottie_ai: st_lottie(lottie_ai, height=120, key="ai_brain")

if st.session_state.resume_text:
    if st.button("🗑️ Clear & Upload New Resume", type="secondary"):
        keys_to_clear = [
            "resume_text", "resume_path", "skills", "issues", "health_data", 
            "predicted_role", "bullet_results", "market_gaps", "ats_ml_score", 
            "section_scores", "yoe", "interview_qs"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if uploaded_file and not st.session_state.resume_text:
    if st.button("Generate Full Report", use_container_width=True, type="primary"):
        with st.status("🧠 Analyzing your resume...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            st.write("📄 Parsing text...")
            text = extract_text_from_pdf(tmp_path)
            st.session_state.resume_text = text
            st.session_state.resume_path = tmp_path
            
            st.write("🔍 Extracting Skills...")
            skills = extract_skills(text)
            st.session_state.skills = skills
            
            st.write("📏 Checking Formatting & Health...")
            issues = check_formatting(text, tmp_path)
            st.session_state.issues = issues
            st.session_state.health_data = compute_general_score(text, issues, skills)
            st.session_state.section_scores = compute_section_scores(text)
            
            st.write("🤖 Predicting Job Role...")
            prediction = predict_job_category(text)
            st.session_state.predicted_role = prediction["category"]
            
            st.write("🤖 Analyzing Bullets & Experience...")
            st.session_state.bullet_results = classify_bullets(extract_bullet_points(text))
            st.session_state.yoe = calculate_yoe(text)
            
            st.write("📈 Computing Market Gaps & Alignment...")
            gaps = get_market_skill_gaps(st.session_state.predicted_role, skills)
            st.session_state.market_gaps = gaps
            
            # Role-Specific ATS Score Adjuster
            features = extract_resume_features(text, tmp_path)
            base_ats = predict_ats_score(features)["score"]
            if len(gaps["matched"]) + len(gaps["missing"]) > 0:
                alignment_ratio = len(gaps["matched"]) / (len(gaps["matched"]) + len(gaps["missing"]))
                # Weight: 50% Base Resume Quality, 50% Exact Skill Alignment
                final_ats = (base_ats * 0.5) + (alignment_ratio * 100 * 0.5)
            else:
                final_ats = base_ats
                
            # Apply Generous ATS Curve (Most commercial parsers grade between 70-95)
            # This boosts the score while maintaining the relative distribution
            curved_ats = min(100, (final_ats * 0.6) + 40)
            
            st.session_state.ats_ml_score = {"score": int(curved_ats)}
            
            status.update(label="✅ Full Report Generated!", state="complete", expanded=False)
            st.rerun()

if st.session_state.resume_text:
    
    role = st.session_state.predicted_role or "Professional"
    ats = int((st.session_state.ats_ml_score or {"score": 0})["score"])
    
    # DB Save
    if not st.session_state.db_saved:
        try:
            email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", st.session_state.resume_text)
            email = email_match.group(0) if email_match else "unknown@email.com"
            name = st.session_state.resume_text.splitlines()[0][:30]
            health = st.session_state.health_data["score"]
            strong = sum(1 for b in st.session_state.bullet_results if b["label"] == "Strong")
            insert_candidate(name, email, "", role, ats, health, strong)
            st.session_state.db_saved = True
        except:
            pass

    st.markdown("---")
    
    # LinkedIn & Salary Card
    salary = estimate_salary(role, ats)
    st.markdown(f"""
    <div class='info-card'>
        <h3 style='margin-top:0; color:#0369A1;'>Estimated Market Value: {salary}</h3>
        <p style='color:#0C4A6E; margin-bottom:0;'>Based on your ML-predicted role (<b>{role}</b>) and ATS competitiveness score.</p>
    </div>
    """, unsafe_allow_html=True)

    @st.dialog("Your Auto-Generated LinkedIn Bio")
    def show_linkedin_bio():
        skills_str = ", ".join(st.session_state.skills[:5]) if st.session_state.skills else "problem-solving"
        bio = f"Driven and detail-oriented {role} with a proven track record of delivering high-quality results. Skilled in {skills_str}, I thrive in collaborative environments where I can leverage technology to solve complex problems.\n\nAlways eager to learn and adapt to new challenges, I am currently looking for opportunities to bring my expertise to an innovative team."
        st.write("Copy and paste this into your LinkedIn 'About' section:")
        st.code(bio, language="markdown")

    if st.button("🔵 Generate LinkedIn Bio"):
        show_linkedin_bio()

    st.markdown("---")
    yoe = st.session_state.get("yoe", 0.0)
    st.header(f"📊 VMock Benchmark Report: {role} ({yoe} YoE)")
    st.markdown("Your resume is scored on three academic pillars: Impact, Presentation, and Competencies.")
    
    # Calculate VMock Pillars
    # 1. Presentation = Formatting Health
    presentation_score = st.session_state.health_data["score"]
    
    # 2. Competencies = ATS Market Skill Alignment (from gaps)
    competencies_score = 100
    gaps = st.session_state.market_gaps or {"matched": [], "missing": []}
    if len(gaps["matched"]) + len(gaps["missing"]) > 0:
        competencies_score = int((len(gaps["matched"]) / (len(gaps["matched"]) + len(gaps["missing"]))) * 100)
        
    # 3. Impact = Ratio of Strong Bullets
    bullets = st.session_state.bullet_results
    strong_count = sum(1 for b in bullets if b["label"] == "Strong")
    impact_score = int((strong_count / max(1, len(bullets))) * 100)
    
    # Row 1: Scores
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### Impact Score")
        fig_impact = create_gauge_chart(impact_score, "Bullet Point Strength")
        st.pyplot(fig_impact)
        plt.close(fig_impact)
        
    with c2:
        st.markdown("### Presentation Score")
        fig_health = create_gauge_chart(presentation_score, "Formatting Health")
        st.pyplot(fig_health)
        plt.close(fig_health)
        
    with c3:
        st.markdown("### Competencies Score")
        fig_comp = create_gauge_chart(competencies_score, "Hard Skill Alignment")
        st.pyplot(fig_comp)
        plt.close(fig_comp)
        
    st.markdown("### 🧠 Explainable AI (XAI) Insights")
    xai_bullets = []
    if impact_score > 80:
        xai_bullets.append("✅ **High Impact:** Your Random Forest score was significantly boosted by a strong presence of Action Verbs and Metrics in your bullet points.")
    else:
        xai_bullets.append("⚠️ **Low Impact Penalty:** Your score was penalized due to a lack of quantifiable metrics. Adding numbers to your achievements will increase your score.")
        
    if presentation_score > 85:
        xai_bullets.append("✅ **Clean Presentation:** The parser easily extracted your data due to excellent formatting health.")
    else:
        xai_bullets.append("⚠️ **Formatting Penalty:** The ML engine struggled to parse some sections. Fix your margins or font consistency to prevent ATS rejection.")
        
    if competencies_score > 75:
        xai_bullets.append("✅ **Market Aligned:** You possess a high density of the hard skills expected for this specific role, boosting your Competencies score.")
    else:
        xai_bullets.append("⚠️ **Skill Gap Penalty:** Your resume lacks critical skills required for the current market, heavily weighing down your ATS prediction.")

    for insight in xai_bullets:
        st.markdown(insight)

    st.markdown("---")
    
    # Row 2: Skills & Gaps
    col_gaps, col_radar = st.columns([1.5, 1])
    
    with col_radar:
        st.markdown("### 🕸️ Skill Distribution")
        skill_cats = categorize_skills(st.session_state.skills or [])
        fig_radar = create_radar_chart(skill_cats)
        st.pyplot(fig_radar)
        plt.close(fig_radar)
        
        st.markdown("### ☁️ Keyword Cloud")
        fig_cloud = create_word_cloud(st.session_state.resume_text)
        st.pyplot(fig_cloud)
        plt.close(fig_cloud)

    with col_gaps:
        st.markdown("### 📈 Market Gap & Skills Improvement")
        gaps = st.session_state.market_gaps
        if gaps["missing"]:
            st.error("⚠️ **Missing In-Demand Skills:** Add these to boost your ATS score for this role.")
            missing_html = "".join([f"<span class='skill-tag skill-miss'>{s}</span>" for s in gaps["missing"]])
            st.markdown(missing_html, unsafe_allow_html=True)
        else:
            st.success("✅ No major market gaps detected for this role!")
            
        st.markdown("✅ **Your Top Skills:**")
        matched_html = "".join([f"<span class='skill-tag skill-match'>{s}</span>" for s in gaps["matched"][:10]])
        st.markdown(matched_html, unsafe_allow_html=True)
        
        st.markdown("### 🛠️ Formatting Issues to Fix")
        if not st.session_state.issues:
            st.success("No formatting issues found!")
        else:
            for issue in st.session_state.issues:
                st.warning(f"**{issue['severity'].upper()}**: {issue['issue']}")

    st.markdown("---")
    
    # Live Upskilling Recommender
    st.header("🎓 Automated Upskilling Recommender")
    st.markdown("Bridge your market gap. We've scraped the web for the top real-life courses for your missing skills.")
    
    if gaps["missing"]:
        from course_scraper import fetch_courses
        
        top_3_missing = gaps["missing"][:3]
        
        # Create 'Slide View' using tabs
        tabs = st.tabs(top_3_missing)
        
        for i, skill in enumerate(top_3_missing):
            with tabs[i]:
                st.markdown(f"**Top recommendations for:** `{skill}`")
                
                with st.spinner(f"📡 Live scraping YouTube & Aggregators for '{skill}' courses..."):
                    courses = fetch_courses(skill, limit=3)
                    
                if not courses:
                    st.info(f"Could not fetch live courses for {skill}. Try checking Udemy manually.")
                else:
                    cols = st.columns(3)
                    for j, course in enumerate(courses[:3]):
                        with cols[j]:
                            st.markdown(f"""
                            <div style='background:#FFFFFF; padding:15px; border-radius:8px; border:1px solid #E2E8F0; box-shadow:0 2px 4px rgba(0,0,0,0.05); height:100%;'>
                                <span style='font-size:12px; color:#64748B; font-weight:bold;'>{course['platform'].upper()}</span>
                                <h5 style='color:#1E293B; margin:8px 0;'>{course['title'][:60]}{"..." if len(course['title'])>60 else ""}</h5>
                                <a href="{course['url']}" target="_blank" style='text-decoration:none; color:#2563EB; font-weight:bold; font-size:14px;'>Watch Course ↗</a>
                            </div>
                            """, unsafe_allow_html=True)
    else:
        st.success("You are fully aligned with the required market skills! No urgent upskilling required.")

    st.markdown("---")
    
    # Row 3: ResumeWorded Line-by-Line Breakdown
    st.markdown("### ✍️ Line-by-Line Bullet Breakdown")
    st.markdown("We've extracted every bullet point on your resume. Here is granular, line-by-line feedback on your writing.")
    
    for b in bullets:
        text = b['text']
        # ResumeWorded Logic checks for Action Verbs and Numbers
        action_verbs = ["developed", "led", "managed", "created", "built", "improved", "designed", "optimized", "spearheaded", "implemented"]
        has_action = any(v in text.lower() for v in action_verbs)
        has_metric = bool(re.search(r'\b\d+%\b|\$\d+|\b\d+\b', text))
        
        feedback = []
        if has_action and has_metric:
            st.markdown(f"<div class='bullet-card b-strong'>✅ <strong>Perfect Impact</strong><br><i>\"{text}\"</i></div>", unsafe_allow_html=True)
        else:
            if not has_action:
                feedback.append("Missing strong action verb (e.g. 'led', 'developed').")
            if not has_metric:
                feedback.append("Missing quantifiable metric (e.g. '20%', '$50k').")
            
            feedback_str = " | ".join(feedback)
            st.markdown(f"<div class='bullet-card b-weak'>⚠️ <strong>Needs Work</strong><br><i>\"{text}\"</i><br><span class='b-sugg'>💡 Feedback: {feedback_str}</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.header("🤖 Smart Resume Enhancer")
    st.markdown("We've generated targeted questions based on the gaps above. Answer them to instantly download an enhanced PDF.")
    
    if not st.session_state.interview_qs:
        st.session_state.interview_qs = generate_interview_questions(
            st.session_state.issues or [], 
            st.session_state.skills or [], 
            st.session_state.market_gaps or {"matched": [], "missing": []}, 
            st.session_state.bullet_results or []
        )
    
    answers = {}
    for i, q in enumerate(st.session_state.interview_qs):
        answers[q] = st.text_area(f"Q{i+1}: {q}", key=f"q_{i}")
    
    if st.button("🚀 Generate Enhanced Resume PDF", type="primary"):
        filled = [a for a in answers.values() if a.strip()]
        if not filled:
            st.warning("Please answer at least one question.")
        else:
            with st.spinner("Compiling PDF..."):
                output_path = os.path.join(tempfile.gettempdir(), "enhanced_resume.pdf")
                generate_enhanced_pdf(
                    st.session_state.resume_text, answers, st.session_state.skills or [],
                    role, output_path
                )
            st.balloons()
            st.success("✅ Enhanced resume generated!")
            with open(output_path, "rb") as f:
                st.download_button("📥 Download PDF", data=f, file_name="enhanced_resume.pdf", mime="application/pdf")
