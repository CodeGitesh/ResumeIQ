import streamlit as st
import tempfile
import os
from analyzer import extract_text_from_pdf, extract_skills, check_formatting, compute_ats_score, compute_general_score, get_market_skill_gaps
from ml_model import predict_job_category

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ — Core",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: #1E2130;
    border-radius: 10px;
    padding: 20px;
    border: 1px solid #2A2D4A;
    text-align: center;
}
.metric-value { font-size: 36px; font-weight: 700; color: #6C63FF; }
.metric-label { font-size: 14px; color: #8B90A0; }

.issue-high { border-left: 4px solid #E74C3C; padding: 10px; background: rgba(231, 76, 60, 0.1); margin-bottom: 10px; border-radius: 4px; }
.issue-medium { border-left: 4px solid #F1C40F; padding: 10px; background: rgba(241, 196, 15, 0.1); margin-bottom: 10px; border-radius: 4px; }

.skill-tag {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    margin: 4px;
    font-size: 13px;
    font-weight: 500;
}
.skill-match { background: rgba(46, 204, 113, 0.2); color: #2ECC71; border: 1px solid rgba(46, 204, 113, 0.4); }
.skill-miss { background: rgba(231, 76, 60, 0.2); color: #E74C3C; border: 1px solid rgba(231, 76, 60, 0.4); }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "resume_path" not in st.session_state:
    st.session_state.resume_path = None
if "skills" not in st.session_state:
    st.session_state.skills = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 ResumeIQ Core")
    page = st.radio("Navigation", ["📄 Upload Resume", "❤️ Overall Health", "🎯 ATS & Skill Gap"])
    st.markdown("---")
    if st.session_state.resume_text:
        st.success("✅ Resume Loaded")
    else:
        st.error("⚠️ No Resume Loaded")


# ── Page 1: Upload Resume ──────────────────────────────────────────────────────
if page == "📄 Upload Resume":
    st.header("Upload Your Resume")
    st.markdown("Upload your PDF to extract text and analyze your baseline profile.")
    
    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])
    
    if uploaded_file:
        if st.button("Analyze Resume", use_container_width=True):
            with st.spinner("Parsing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                try:
                    text = extract_text_from_pdf(tmp_path)
                    st.session_state.resume_text = text
                    st.session_state.resume_path = tmp_path
                    st.session_state.skills = extract_skills(text)
                    st.success("Resume successfully parsed!")
                except Exception as e:
                    st.error(f"Failed to parse resume: {e}")
                    
    if st.session_state.resume_text:
        st.markdown("---")
        st.markdown("### AI Predicted Profile")
        
        with st.spinner("Running ML Model against Kaggle Dataset..."):
            prediction = predict_job_category(st.session_state.resume_text)
            
        role = prediction["category"]
        conf = prediction["confidence"]
        
        st.markdown(f"""
        <div class='metric-card' style='margin-bottom: 20px; border-color: #6C63FF;'>
            <div class='metric-label'>Based on our NLP model trained on thousands of Kaggle resumes, your profile matches:</div>
            <div class='metric-value' style='margin-top: 10px;'>{role}</div>
            <div style='color: #2ECC71; font-weight: 600; margin-top: 5px;'>{conf:.1f}% Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Extracted {len(st.session_state.skills)} Skills:**")
            st.write(", ".join(st.session_state.skills) if st.session_state.skills else "None found.")
            
            # Show Advanced Market Skill Gaps
            st.markdown("---")
            st.markdown("### 📈 Market Trends Recommender")
            market_gaps = get_market_skill_gaps(role, st.session_state.skills)
            
            if market_gaps["missing"]:
                st.warning(f"**Market Gap:** To be a competitive {role}, the market highly demands these skills which are missing from your resume:")
                missing_html = "".join([f"<span class='skill-tag skill-miss'>{s}</span>" for s in market_gaps["missing"]])
                st.markdown(missing_html, unsafe_allow_html=True)
            elif market_gaps["matched"]:
                st.success(f"**Great job!** You have all the core market skills expected for a {role}.")
                
            if market_gaps["matched"]:
                st.markdown("**Matched Core Market Skills:**")
                matched_html = "".join([f"<span class='skill-tag skill-match'>{s}</span>" for s in market_gaps["matched"]])
                st.markdown(matched_html, unsafe_allow_html=True)

        with col2:
            st.markdown("**Raw Text Preview:**")
            st.text_area("", st.session_state.resume_text[:1000] + "...", height=350, disabled=True)


# ── Page 2: Overall Health ──────────────────────────────────────────────────
elif page == "❤️ Overall Health":
    st.header("Overall Resume Health")
    st.markdown("A general score evaluating your resume's formatting, completeness, and impact independent of any specific job description.")
    
    if not st.session_state.resume_text:
        st.warning("Please upload a resume first.")
    else:
        with st.spinner("Running formatting checks and general scoring..."):
            issues = check_formatting(st.session_state.resume_text, st.session_state.resume_path)
            health = compute_general_score(st.session_state.resume_text, issues, st.session_state.skills)
            
        score = health["score"]
        
        # Display Score Circle
        st.markdown(f"""
        <div class='metric-card' style='margin-bottom: 20px;'>
            <div class='metric-value' style='color: {"#2ECC71" if score >= 80 else "#F1C40F" if score >= 60 else "#E74C3C"}; font-size: 48px;'>{score}/100</div>
            <div class='metric-label'>General Resume Score</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Breakdown
        st.markdown("### Score Breakdown")
        for item in health["breakdown"]:
            impact = item["impact"]
            if str(impact).startswith("+"):
                st.markdown(f"<div style='color: #2ECC71; font-weight: 500;'>✅ {item['item']} ({impact} pts)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color: #E74C3C; font-weight: 500;'>❌ {item['item']} ({impact} pts)</div>", unsafe_allow_html=True)
                
        st.markdown("---")
        st.markdown("### Formatting Issues & Suggestions")
        if not issues:
            st.success("🎉 Great job! No major formatting issues found.")
        else:
            for item in issues:
                css_class = "issue-high" if item['severity'] == "high" else "issue-medium"
                st.markdown(f"<div class='{css_class}'><strong>{item['severity'].upper()} Priority:</strong> {item['issue']}</div>", unsafe_allow_html=True)


# ── Page 3: ATS & Skill Gap ────────────────────────────────────────────────────
elif page == "🎯 ATS & Skill Gap":
    st.header("ATS Scoring & Skill Gap Analysis")
    st.markdown("Paste a Job Description below to see how well your resume matches the role.")
    
    if not st.session_state.resume_text:
        st.warning("Please upload a resume first.")
    else:
        jd_text = st.text_area("Paste Job Description", height=200, placeholder="Paste the responsibilities and requirements here...")
        
        if st.button("Calculate Match", use_container_width=True) and jd_text:
            with st.spinner("Analyzing match..."):
                results = compute_ats_score(st.session_state.resume_text, jd_text)
                
            score = results["score"]
            st.markdown("---")
            
            # ATS Score
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='color: {"#2ECC71" if score >= 70 else "#F1C40F" if score >= 40 else "#E74C3C"};'>{score}%</div>
                    <div class='metric-label'>ATS Match Score</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                if score >= 70:
                    st.success("🔥 Excellent match! You are highly qualified for this role.")
                elif score >= 40:
                    st.warning("⚠️ Moderate match. Consider adding the missing skills below to boost your chances.")
                else:
                    st.error("❌ Low match. Major skill gaps detected. Ensure you tailor your resume to the job description.")
            
            st.markdown("### 🔍 Skill Gap Analysis")
            matched = results["matched_skills"]
            missing = results["missing_skills"]
            
            col_match, col_miss = st.columns(2)
            with col_match:
                st.markdown("✅ **Matched Skills (You have these):**")
                if matched:
                    html = "".join([f"<span class='skill-tag skill-match'>{s.title()}</span>" for s in matched])
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.write("None.")
                    
            with col_miss:
                st.markdown("❌ **Missing Skills (Add these):**")
                if missing:
                    html = "".join([f"<span class='skill-tag skill-miss'>{s.title()}</span>" for s in missing])
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.write("None.")
