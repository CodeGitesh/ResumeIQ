import streamlit as st
import tempfile
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

from analyzer import (
    extract_text_from_pdf, extract_skills, check_formatting,
    compute_ats_score, compute_general_score, get_market_skill_gaps,
    extract_bullet_points, compute_section_scores, categorize_skills,
    extract_resume_features
)
from ml_model import predict_job_category, predict_ats_score, classify_bullets
from resume_builder import generate_interview_questions, generate_enhanced_pdf

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ v2.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: #1E2130;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #2A2D4A;
    text-align: center;
    margin-bottom: 16px;
}
.metric-value { font-size: 42px; font-weight: 700; }
.metric-label { font-size: 14px; color: #8B90A0; margin-top: 4px; }

.issue-high { border-left: 4px solid #E74C3C; padding: 12px 16px; background: rgba(231, 76, 60, 0.08); margin-bottom: 10px; border-radius: 6px; }
.issue-medium { border-left: 4px solid #F1C40F; padding: 12px 16px; background: rgba(241, 196, 15, 0.08); margin-bottom: 10px; border-radius: 6px; }

.skill-tag {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 14px;
    margin: 4px;
    font-size: 13px;
    font-weight: 500;
}
.skill-match { background: rgba(46, 204, 113, 0.15); color: #2ECC71; border: 1px solid rgba(46, 204, 113, 0.3); }
.skill-miss { background: rgba(231, 76, 60, 0.15); color: #E74C3C; border: 1px solid rgba(231, 76, 60, 0.3); }

.bullet-strong { border-left: 4px solid #2ECC71; padding: 10px 16px; background: rgba(46, 204, 113, 0.08); margin-bottom: 8px; border-radius: 6px; }
.bullet-weak { border-left: 4px solid #E74C3C; padding: 10px 16px; background: rgba(231, 76, 60, 0.08); margin-bottom: 8px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
for key in ["resume_text", "resume_path", "skills", "predicted_role", "issues",
            "bullet_results", "market_gaps", "section_scores", "ats_ml_score",
            "interview_qs", "answers"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ── Chart Helper Functions ─────────────────────────────────────────────────────

def create_gauge_chart(score, title="ATS Score"):
    """Creates a semi-circle gauge chart for a 0-100 score."""
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#0E1117')
    
    # Convert score to angle (0-180 degrees mapped to pi-0 radians)
    angle = np.pi * (1 - score / 100)
    
    # Background arc
    theta_bg = np.linspace(0, np.pi, 100)
    ax.fill_between(theta_bg, 0.6, 1.0, color='#2A2D4A', alpha=0.5)
    
    # Score arc
    theta_score = np.linspace(np.pi, angle, 100)
    color = '#2ECC71' if score >= 70 else '#F1C40F' if score >= 40 else '#E74C3C'
    ax.fill_between(theta_score, 0.6, 1.0, color=color, alpha=0.8)
    
    ax.set_ylim(0, 1.2)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.axis('off')
    
    # Score text
    ax.text(np.pi/2, 0.2, f"{score}", ha='center', va='center',
            fontsize=28, fontweight='bold', color=color)
    ax.text(np.pi/2, -0.15, title, ha='center', va='center',
            fontsize=10, color='#8B90A0')
    
    plt.tight_layout()
    return fig


def create_radar_chart(skill_categories):
    """Creates a radar/spider chart for skill categories."""
    categories = list(skill_categories.keys())
    values = [skill_categories[c]["score"] for c in categories]
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#6C63FF')
    ax.fill(angles, values, alpha=0.2, color='#6C63FF')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9, color='#CCC')
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], size=7, color='#666')
    ax.spines['polar'].set_color('#2A2D4A')
    ax.grid(color='#2A2D4A', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def create_section_bar_chart(section_scores):
    """Creates a horizontal bar chart for section-wise scores."""
    sections = list(section_scores.keys())
    scores = list(section_scores.values())
    
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    colors = ['#2ECC71' if s >= 70 else '#F1C40F' if s >= 40 else '#E74C3C' for s in scores]
    bars = ax.barh(sections, scores, color=colors, height=0.5, edgecolor='none')
    
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{score}%', va='center', ha='left', fontsize=10, color='#CCC', fontweight='bold')
    
    ax.set_xlim(0, 110)
    ax.tick_params(colors='#CCC', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#2A2D4A')
    ax.spines['left'].set_color('#2A2D4A')
    ax.set_xlabel('Score', color='#8B90A0', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_before_after_chart(before_score, after_score):
    """Creates a before/after comparison bar chart."""
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    bars = ax.bar(['Before', 'After'], [before_score, after_score],
                  color=['#E74C3C', '#2ECC71'], width=0.4, edgecolor='none')
    
    for bar, score in zip(bars, [before_score, after_score]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{score}', ha='center', va='bottom', fontsize=14, color='#CCC', fontweight='bold')
    
    ax.set_ylim(0, 110)
    ax.tick_params(colors='#CCC', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#2A2D4A')
    ax.spines['left'].set_color('#2A2D4A')
    ax.set_ylabel('Score', color='#8B90A0')
    
    plt.tight_layout()
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 ResumeIQ v2.0")
    st.caption("ML-Powered Resume Enhancer")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📄 Upload Resume",
        "❤️ Overall Health",
        "💪 Bullet Strength",
        "📊 Analytics",
        "🎯 ATS & Skill Gap",
        "🤖 Smart Enhancer"
    ])
    st.markdown("---")
    if st.session_state.resume_text:
        st.success("✅ Resume Loaded")
        if st.session_state.predicted_role:
            st.info(f"🏷️ {st.session_state.predicted_role}")
    else:
        st.warning("⚠️ Upload a Resume to begin")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1: Upload Resume
# ════════════════════════════════════════════════════════════════════════════════
if page == "📄 Upload Resume":
    st.header("📄 Upload Your Resume")
    st.markdown("Upload your PDF and our 3 ML models will instantly analyze it.")
    
    uploaded_file = st.file_uploader("Drop your PDF here", type=["pdf"])
    
    if uploaded_file:
        if st.button("🚀 Analyze Resume", use_container_width=True):
            with st.spinner("Extracting text from PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                try:
                    text = extract_text_from_pdf(tmp_path)
                    st.session_state.resume_text = text
                    st.session_state.resume_path = tmp_path
                    st.session_state.skills = extract_skills(text)
                    st.session_state.issues = check_formatting(text, tmp_path)
                    st.session_state.section_scores = compute_section_scores(text)
                except Exception as e:
                    st.error(f"Failed to parse resume: {e}")
                    st.stop()
            
            with st.spinner("🤖 Running ML Model 1: Job Role Prediction (KNN)..."):
                prediction = predict_job_category(st.session_state.resume_text)
                st.session_state.predicted_role = prediction["category"]
            
            with st.spinner("🤖 Running ML Model 2: ATS Score Prediction (Random Forest)..."):
                features = extract_resume_features(text, tmp_path)
                ats_result = predict_ats_score(features)
                st.session_state.ats_ml_score = ats_result
            
            with st.spinner("🤖 Running ML Model 3: Bullet Point Analysis (Naive Bayes)..."):
                bullets = extract_bullet_points(text)
                st.session_state.bullet_results = classify_bullets(bullets)
            
            with st.spinner("📈 Computing Market Skill Gaps..."):
                st.session_state.market_gaps = get_market_skill_gaps(
                    st.session_state.predicted_role, st.session_state.skills
                )
            
            st.success("✅ All 3 ML models completed analysis!")
                    
    if st.session_state.resume_text:
        st.markdown("---")
        
        # ML Results Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            role = st.session_state.predicted_role or "Unknown"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>ML Model 1: KNN Classifier</div>
                <div class='metric-value' style='color: #6C63FF; font-size: 24px;'>{role}</div>
                <div class='metric-label'>Predicted Job Role</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ats = st.session_state.ats_ml_score or {"score": 0}
            score_val = int(ats["score"])
            color = "#2ECC71" if score_val >= 70 else "#F1C40F" if score_val >= 40 else "#E74C3C"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>ML Model 2: Random Forest</div>
                <div class='metric-value' style='color: {color};'>{score_val}</div>
                <div class='metric-label'>Predicted ATS Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            bullets = st.session_state.bullet_results or []
            strong = sum(1 for b in bullets if b["label"] == "Strong")
            total = len(bullets)
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>ML Model 3: Naive Bayes</div>
                <div class='metric-value' style='color: #6C63FF;'>{strong}/{total}</div>
                <div class='metric-label'>Strong Bullet Points</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Market Gaps
        if st.session_state.market_gaps:
            gaps = st.session_state.market_gaps
            if gaps["missing"]:
                st.markdown("### 📈 Market Skill Gaps")
                st.warning(f"To be competitive as a **{st.session_state.predicted_role}**, add these in-demand skills:")
                missing_html = "".join([f"<span class='skill-tag skill-miss'>{s}</span>" for s in gaps["missing"]])
                st.markdown(missing_html, unsafe_allow_html=True)
            if gaps["matched"]:
                matched_html = "".join([f"<span class='skill-tag skill-match'>{s}</span>" for s in gaps["matched"]])
                st.markdown("**✅ Matched Market Skills:**")
                st.markdown(matched_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2: Overall Health
# ════════════════════════════════════════════════════════════════════════════════
elif page == "❤️ Overall Health":
    st.header("❤️ Overall Resume Health")
    
    if not st.session_state.resume_text:
        st.warning("Please upload a resume first.")
    else:
        issues = st.session_state.issues or check_formatting(st.session_state.resume_text, st.session_state.resume_path)
        health = compute_general_score(st.session_state.resume_text, issues, st.session_state.skills or [])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = create_gauge_chart(health["score"], "Health Score")
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.markdown("### Score Breakdown")
            for item in health["breakdown"]:
                impact = item["impact"]
                if str(impact).startswith("+"):
                    st.markdown(f"<div style='color: #2ECC71;'>✅ {item['item']} ({impact} pts)</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color: #E74C3C;'>❌ {item['item']} ({impact} pts)</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Formatting Issues")
        if not issues:
            st.success("🎉 No major formatting issues found!")
        else:
            for item in issues:
                css_class = "issue-high" if item['severity'] == "high" else "issue-medium"
                st.markdown(f"<div class='{css_class}'><strong>{item['severity'].upper()}:</strong> {item['issue']}</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3: Bullet Strength
# ════════════════════════════════════════════════════════════════════════════════
elif page == "💪 Bullet Strength":
    st.header("💪 Bullet Point Impact Analysis")
    st.markdown("Our **Naive Bayes ML model** classifies each bullet point as **Strong** or **Weak**.")
    
    if not st.session_state.resume_text:
        st.warning("Please upload a resume first.")
    elif not st.session_state.bullet_results:
        st.warning("Please re-upload and analyze your resume to generate bullet analysis.")
    else:
        results = st.session_state.bullet_results
        strong_count = sum(1 for b in results if b["label"] == "Strong")
        weak_count = len(results) - strong_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color: #2ECC71;'>{strong_count}</div>
                <div class='metric-label'>Strong Bullets</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color: #E74C3C;'>{weak_count}</div>
                <div class='metric-label'>Weak Bullets</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        for b in results:
            css = "bullet-strong" if b["label"] == "Strong" else "bullet-weak"
            icon = "💪" if b["label"] == "Strong" else "⚠️"
            conf = f"{b['confidence']*100:.0f}%"
            st.markdown(f"<div class='{css}'>{icon} <strong>{b['label']}</strong> ({conf}) — {b['text']}</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4: Analytics
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.header("📊 Visual Analytics Dashboard")
    
    if not st.session_state.resume_text:
        st.warning("Please upload a resume first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 ML-Predicted ATS Score")
            ats = st.session_state.ats_ml_score or {"score": 50}
            fig = create_gauge_chart(int(ats["score"]), "ATS Score")
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.markdown("### 🕸️ Skill Coverage Radar")
            skill_cats = categorize_skills(st.session_state.skills or [])
            fig = create_radar_chart(skill_cats)
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown("---")
        st.markdown("### 📊 Section-Wise Strength")
        scores = st.session_state.section_scores or compute_section_scores(st.session_state.resume_text)
        fig = create_section_bar_chart(scores)
        st.pyplot(fig)
        plt.close(fig)
        
        # Skill category details
        st.markdown("---")
        st.markdown("### 🏷️ Skill Breakdown by Category")
        skill_cats = categorize_skills(st.session_state.skills or [])
        for cat, data in skill_cats.items():
            if data["matched"]:
                tags = "".join([f"<span class='skill-tag skill-match'>{s}</span>" for s in data["matched"]])
                st.markdown(f"**{cat}** ({data['count']} found): {tags}", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5: ATS & Skill Gap (JD Comparison)
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🎯 ATS & Skill Gap":
    st.header("🎯 ATS Scoring vs Job Description")
    st.markdown("Paste a Job Description to see how well your resume matches.")
    
    if not st.session_state.resume_text:
        st.warning("Please upload a resume first.")
    else:
        jd_text = st.text_area("Paste Job Description", height=200, placeholder="Paste the JD here...")
        
        if st.button("Calculate Match", use_container_width=True) and jd_text:
            with st.spinner("Analyzing..."):
                results = compute_ats_score(st.session_state.resume_text, jd_text)
                
            score = results["score"]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                fig = create_gauge_chart(score, "JD Match")
                st.pyplot(fig)
                plt.close(fig)
                
            with col2:
                if score >= 70:
                    st.success("🔥 Excellent match! Highly qualified for this role.")
                elif score >= 40:
                    st.warning("⚠️ Moderate match. Add missing skills to boost score.")
                else:
                    st.error("❌ Low match. Major skill gaps detected.")
            
            st.markdown("### 🔍 Skill Gap")
            col_m, col_x = st.columns(2)
            with col_m:
                st.markdown("✅ **Matched:**")
                if results.get("matched_skills"):
                    html = "".join([f"<span class='skill-tag skill-match'>{s.title()}</span>" for s in results["matched_skills"]])
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.write("None.")
            with col_x:
                st.markdown("❌ **Missing:**")
                if results.get("missing_skills"):
                    html = "".join([f"<span class='skill-tag skill-miss'>{s.title()}</span>" for s in results["missing_skills"]])
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.write("None.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 6: Smart Enhancer
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Smart Enhancer":
    st.header("🤖 Smart Resume Enhancer")
    st.markdown("Our ML models identified gaps in your resume. Answer the targeted questions below and we'll generate an **enhanced PDF** for you.")
    
    if not st.session_state.resume_text:
        st.warning("Please upload a resume first.")
    else:
        # Generate questions if not already done
        if not st.session_state.interview_qs:
            issues = st.session_state.issues or []
            skills = st.session_state.skills or []
            market_gaps = st.session_state.market_gaps or {"matched": [], "missing": []}
            bullet_results = st.session_state.bullet_results or []
            
            st.session_state.interview_qs = generate_interview_questions(
                issues, skills, market_gaps, bullet_results
            )
        
        questions = st.session_state.interview_qs
        
        st.markdown(f"### 📝 Answer {len(questions)} Targeted Questions")
        st.caption("These questions were dynamically generated by our ML analysis of your resume.")
        
        answers = {}
        for i, q in enumerate(questions):
            answers[q] = st.text_area(f"Q{i+1}: {q}", height=80, key=f"q_{i}")
        
        st.markdown("---")
        
        if st.button("🚀 Generate Enhanced Resume PDF", use_container_width=True):
            # Check at least some answers are filled
            filled = [a for a in answers.values() if a.strip()]
            
            if not filled:
                st.warning("Please answer at least one question before generating.")
            else:
                with st.spinner("Generating your enhanced resume PDF..."):
                    output_path = os.path.join(tempfile.gettempdir(), "enhanced_resume.pdf")
                    generate_enhanced_pdf(
                        original_text=st.session_state.resume_text,
                        answers=answers,
                        skills=st.session_state.skills or [],
                        predicted_role=st.session_state.predicted_role or "Professional",
                        output_path=output_path
                    )
                
                st.success("✅ Enhanced resume generated!")
                
                # Before/After comparison
                old_score = int((st.session_state.ats_ml_score or {"score": 50})["score"])
                new_score = min(100, old_score + len(filled) * 5 + 10)
                
                st.markdown("### 📊 Before vs After")
                fig = create_before_after_chart(old_score, new_score)
                st.pyplot(fig)
                plt.close(fig)
                
                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Enhanced Resume PDF",
                        data=f,
                        file_name="enhanced_resume.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
