import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from analyzer import extract_skills

st.set_page_config(page_title="Custom Job Scanner", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .metric-card { background: #F0F9FF; border: 1px solid #BAE6FD; padding: 20px; border-radius: 8px; text-align: center; }
    .keyword-match { display: inline-block; padding: 4px 10px; border-radius: 12px; margin: 4px; font-size: 13px; font-weight: 500; background: #D1FAE5; color: #065F46; border: 1px solid #10B981; }
    .keyword-miss { display: inline-block; padding: 4px 10px; border-radius: 12px; margin: 4px; font-size: 13px; font-weight: 500; background: #FEE2E2; color: #991B1B; border: 1px solid #EF4444; }
</style>
""", unsafe_allow_html=True)

if "resume_text" not in st.session_state or not st.session_state.resume_text:
    st.warning("⚠️ Please upload your resume in the Candidate Portal first.")
    st.stop()

st.title("🎯 Custom Job Scanner")
st.markdown("Paste a Job Description (JD) below to see exactly how well your resume matches the required keywords.")

jd_text = st.text_area("Paste Job Description here:", height=250, placeholder="E.g. We are looking for a Software Engineer with experience in Python, AWS, and Docker...")

if st.button("🔍 Scan Resume vs JD", type="primary"):
    if len(jd_text.split()) < 20:
        st.error("Please paste a longer job description for an accurate scan.")
    else:
        with st.spinner("Analyzing keyword overlap..."):
            # 1. Extract Skills from JD
            jd_skills = extract_skills(jd_text)
            resume_skills = st.session_state.skills or []
            
            # 2. Find Missing vs Matched using Fuzzy Semantic Matching
            import difflib
            resume_skills_lower = [s.lower() for s in resume_skills]
            matched = []
            missing = []
            
            for skill in jd_skills:
                skill_lower = skill.lower()
                # Check for exact match first
                if skill_lower in resume_skills_lower:
                    matched.append(skill)
                else:
                    # Fuzzy match (threshold 0.75 matches "react" to "react.js")
                    close_matches = difflib.get_close_matches(skill_lower, resume_skills_lower, n=1, cutoff=0.75)
                    if close_matches:
                        matched.append(f"{skill} (matched as {close_matches[0]})")
                    else:
                        missing.append(skill)
                    
            # 3. Calculate Match Score based on Skill Overlap (Jobscan methodology)
            if len(jd_skills) > 0:
                skill_match_score = (len(matched) / len(jd_skills)) * 100
            else:
                skill_match_score = 0
                
            # Combine with baseline TF-IDF for general text similarity
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf = vectorizer.fit_transform([st.session_state.resume_text, jd_text])
            from sklearn.metrics.pairwise import cosine_similarity
            text_sim_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
            
            # 80% weight on exact skills, 20% on general text similarity
            if len(jd_skills) > 0:
                match_score = int((skill_match_score * 0.8) + (text_sim_score * 0.2))
            else:
                match_score = int(text_sim_score)
            
            match_score = min(100, match_score)
            
            # Display Results
            st.markdown("---")
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0; color:#0369A1;'>Resume Match Rate</h3>
                <h1 style='margin:10px 0; font-size:48px; color:{"#10B981" if match_score >= 70 else "#F59E0B" if match_score >= 40 else "#EF4444"};'>{match_score}%</h1>
                <p style='margin:0; color:#0C4A6E;'>Aim for 75%+ before submitting your application.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ❌ Missing Keywords")
                st.markdown("Add these exact words to your resume to beat the ATS:")
                if not missing:
                    st.success("You have all the required skills!")
                else:
                    missing_html = "".join([f"<span class='keyword-miss'>{s}</span>" for s in missing])
                    st.markdown(missing_html, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### ✅ Matched Keywords")
                st.markdown("You already have these required skills:")
                if not matched:
                    st.error("No keywords matched.")
                else:
                    matched_html = "".join([f"<span class='keyword-match'>{s}</span>" for s in matched])
                    st.markdown(matched_html, unsafe_allow_html=True)
