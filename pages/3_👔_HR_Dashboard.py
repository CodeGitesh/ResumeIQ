import streamlit as st
import pandas as pd
from database import get_all_candidates

st.set_page_config(page_title="HR Admin Dashboard", page_icon="👔", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .metric-card {
        background: rgba(30, 33, 48, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if not st.session_state.admin_logged_in:
    st.title("🔒 Admin Login")
    with st.form("login_form"):
        pwd = st.text_input("Enter Admin Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if pwd == "admin123":  # Hardcoded mock password
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("Incorrect password")
else:
    st.title("👔 Enterprise HR Dashboard")
    
    if st.button("Logout", key="logout"):
        st.session_state.admin_logged_in = False
        st.rerun()
        
    st.markdown("---")
    
    try:
        candidates = get_all_candidates()
    except Exception as e:
        candidates = []
        st.error(f"Database error: {e}")
        
    if not candidates:
        st.info("No candidates processed yet. Upload a resume in the Candidate Portal first.")
    else:
        df = pd.DataFrame(candidates)
        
        # High level metrics
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card'><h3>Total Candidates</h3><h1>{len(df)}</h1></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h3>Avg ATS Score</h3><h1 style='color:#2ECC71;'>{int(df['ats_score'].mean())}</h1></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h3>Top Role</h3><h1 style='color:#6C63FF;'>{df['predicted_role'].mode()[0]}</h1></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 Candidate Database")
        
        # Format the dataframe for display
        display_df = df[['id', 'name', 'predicted_role', 'ats_score', 'health_score', 'strong_bullets', 'timestamp']]
        display_df.columns = ['ID', 'Name', 'Role', 'ATS Score', 'Health', 'Strong Bullets', 'Date']
        
        st.markdown("---")
        st.markdown("### 📈 Business Intelligence Analytics")
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        col_pie, col_bar = st.columns(2)
        
        with col_pie:
            st.markdown("#### Role Distribution")
            role_counts = df['predicted_role'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            fig1.patch.set_facecolor('white')
            ax1.pie(role_counts, labels=role_counts.index, autopct='%1.1f%%', startangle=90, colors=['#2563EB', '#10B981', '#F59E0B', '#64748B'])
            ax1.axis('equal')
            st.pyplot(fig1)
            plt.close(fig1)
            
        with col_bar:
            st.markdown("#### Average ATS Score by Role")
            avg_ats = df.groupby('predicted_role')['ats_score'].mean()
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            fig2.patch.set_facecolor('white')
            ax2.bar(avg_ats.index, avg_ats.values, color='#2563EB')
            ax2.set_ylabel('Avg ATS Score')
            ax2.set_ylim(0, 100)
            plt.xticks(rotation=45, ha='right')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        
        st.markdown("---")
        st.markdown("### 📋 Candidate Database")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ATS Score": st.column_config.ProgressColumn(
                    "ATS Score",
                    help="Predicted ATS pass probability",
                    format="%f",
                    min_value=0,
                    max_value=100,
                ),
                "Health": st.column_config.NumberColumn(
                    "Health Score",
                    help="General formatting health",
                    format="%d",
                )
            }
        )
