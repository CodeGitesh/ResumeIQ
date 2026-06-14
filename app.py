import streamlit as st

st.set_page_config(
    page_title="ResumeIQ | AI Candidate OS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit elements and apply light mode css
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Define pages
pages = {
    "Candidate Tools": [
        st.Page("pages/1_🏠_Candidate_Portal.py", title="Upload & Enhance", icon="🏠"),
        st.Page("pages/2_💼_Job_Matches.py", title="Job Matches", icon="💼"),
        st.Page("pages/5_🎯_Jobscan_Matcher.py", title="Custom Job Scanner", icon="🎯"),
        st.Page("pages/6_📋_Application_Tracker.py", title="Application Tracker", icon="📋"),
        st.Page("pages/4_🎙️_Interview_Grader.py", title="Interview Grader", icon="🎙️"),
    ],
    "Enterprise Settings": [
        st.Page("pages/3_👔_HR_Dashboard.py", title="HR Admin Dashboard", icon="👔"),
    ]
}

# Run navigation
pg = st.navigation(pages)
pg.run()
