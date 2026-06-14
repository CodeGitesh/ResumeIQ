import streamlit as st
import os
import tempfile
from job_matcher import recommend_jobs
from resume_builder import generate_cover_letter_pdf

st.set_page_config(page_title="Job Matches", page_icon="💼", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .job-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .job-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
        border-color: #2563EB;
    }
    .match-score {
        float: right;
        font-size: 24px;
        font-weight: bold;
        color: #10B981;
    }
    .job-title {
        margin:0; 
        color: #1E293B;
    }
    .job-meta {
        margin:5px 0 10px 0; 
        color: #64748B;
    }
    .job-desc {
        color: #475569; 
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.title("💼 Live Job Recommendations")
st.markdown("Based on your uploaded resume, our TF-IDF matching engine has found the best active roles for you.")

if "resume_text" not in st.session_state or not st.session_state.resume_text:
    st.warning("⚠️ Please upload a resume in the Candidate Portal first to see job matches.")
else:
    ats = int((st.session_state.ats_ml_score or {"score": 0})["score"])
    st.markdown(f"""
    <div style='background:#F0F9FF; border:1px solid #BAE6FD; padding:20px; border-radius:8px; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center;'>
        <div>
            <h3 style='margin:0; color:#0369A1;'>Your Role-Specific ATS Score:</h3>
            <p style='margin:0; color:#0C4A6E;'>This dictates your competitiveness for the roles below.</p>
        </div>
        <h1 style='margin:0; color:{"#10B981" if ats >= 70 else "#F59E0B" if ats >= 40 else "#EF4444"};'>{ats}/100</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Top Matches")
    
    matches = recommend_jobs(st.session_state.resume_text, top_n=5)
    
    if not matches:
        st.info("No matching jobs found in the database.")
    else:
        for job in matches:
            st.markdown(f"""
            <div class='job-card'>
                <div class='match-score'>{job['match_score']}% Match</div>
                <h3 class='job-title'>{job['title']}</h3>
                <h5 class='job-meta'>{job['company']} • {job['location']} • {job['salary']}</h5>
                <p class='job-desc'>{job['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                # Generate Cover Letter
                if st.button(f"📄 Cover Letter", key=f"cl_{job['id']}", type="primary"):
                    with st.spinner("Generating PDF..."):
                        output_path = os.path.join(tempfile.gettempdir(), f"Cover_Letter_{job['company']}.pdf")
                        generate_cover_letter_pdf(
                            st.session_state.resume_text,
                            st.session_state.skills or [],
                            job['title'],
                            job['company'],
                            output_path
                        )
                        with open(output_path, "rb") as f:
                            st.download_button(
                                "📥 Download PDF",
                                data=f,
                                file_name=f"Cover_Letter_{job['company']}.pdf",
                                mime="application/pdf",
                                key=f"dl_{job['id']}"
                            )

            with col2:
                # Mailto cold email
                skills_str = ", ".join(st.session_state.skills[:3]) if st.session_state.skills else "software development"
                subject = f"Application for {job['title']} - [Your Name]"
                body = f"Hi Hiring Team at {job['company']},\n\nI hope this email finds you well. I'm reaching out to express my strong interest in the {job['title']} position.\n\nWith my background in {skills_str}, I believe my technical skills and proactive approach would make me a great fit for this role.\n\nI have attached my resume for your consideration. I'd love to chat briefly if you have a few minutes next week.\n\nBest regards,\n[Your Name]"
                
                import urllib.parse
                mail_link = f"mailto:hr@{job['company'].lower().replace(' ', '')}.com?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
                st.markdown(f"<a href='{mail_link}' target='_blank'><button style='width:100%; padding:8px; border-radius:8px; border:1px solid #E2E8F0; background:#FFFFFF; color:#1E293B; cursor:pointer;'>✉️ Draft Email</button></a>", unsafe_allow_html=True)
            
            with col3:
                # Track in Database
                if st.button("➕ Track Application", key=f"track_{job['id']}"):
                    from database import insert_application
                    insert_application(job['company'], job['title'], "Wishlist", "")
                    st.toast(f"Added {job['company']} to your Application Tracker!")
