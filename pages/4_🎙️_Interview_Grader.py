import streamlit as st
import random
import re

st.set_page_config(page_title="Interview Grader", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .grader-pass { background: #D1FAE5; color: #065F46; padding: 15px; border-radius: 8px; border-left: 4px solid #10B981; }
    .grader-fail { background: #FEE2E2; color: #991B1B; padding: 15px; border-radius: 8px; border-left: 4px solid #EF4444; }
</style>
""", unsafe_allow_html=True)

if "resume_text" not in st.session_state or not st.session_state.resume_text:
    st.warning("⚠️ Please upload a resume in the Candidate Portal first to activate the Interview Grader.")
    st.stop()

st.title("🎙️ Interactive Interview Grader")
st.markdown("Practice your interview skills. The NLP engine uses the **STAR Method** to verify your answer contains Context, Action Verbs, and Metrics/Results.")

if not st.session_state.interview_qs:
    st.info("Please generate a Full Report on the Candidate Portal to unlock interview questions.")
else:
    if "active_q" not in st.session_state:
        st.session_state.active_q = random.choice(st.session_state.interview_qs)
        
    if st.button("🔄 Get a new question"):
        st.session_state.active_q = random.choice(st.session_state.interview_qs)
        st.rerun()
        
    st.markdown(f"### Question: {st.session_state.active_q}")
    
    answer = st.text_area("Type your answer here...", height=200, placeholder="E.g., In my last role (Situation)... I was tasked with (Task)... I developed X using Python (Action)... resulting in a 20% increase in speed (Result).")
    
    if st.button("📝 Grade My Answer", type="primary"):
        if len(answer.split()) < 20:
            st.markdown("<div class='grader-fail'><strong>FAIL (Too Short)</strong><br>Your answer is too short. A good interview response should be at least 3-4 sentences long. Provide more context.</div>", unsafe_allow_html=True)
        else:
            ans_lower = answer.lower()
            
            # STAR DETECTOR LOGIC
            # 1. Action Verbs
            action_verbs = ["developed", "led", "managed", "created", "built", "improved", "designed", "optimized", "spearheaded", "implemented"]
            has_action = any(v in ans_lower for v in action_verbs)
            
            # 2. Results / Metrics (Checking for numbers, %, or $)
            has_metric = bool(re.search(r'\b\d+%\b|\$\d+|\b\d+\b', answer))
            
            # 3. "I" Statements (Ownership)
            has_i = " i " in answer.lower() or answer.lower().startswith("i ")
            
            score = 60
            feedback = []
            
            if has_action:
                score += 15
                feedback.append("✅ **Action:** Used strong action verbs.")
            else:
                feedback.append("❌ **Action:** Missing strong action verbs (e.g., 'led', 'developed'). You sound passive.")
                
            if has_metric:
                score += 15
                feedback.append("✅ **Result:** Included numbers/metrics to quantify your impact.")
            else:
                feedback.append("❌ **Result:** Missing quantifiable metrics. You MUST include numbers (e.g., 'increased by 20%') to prove your impact.")
                
            if has_i:
                score += 10
                feedback.append("✅ **Ownership:** Used 'I' statements.")
            else:
                feedback.append("❌ **Ownership:** Avoid saying 'we' too much. Use 'I' to show personal ownership.")
            
            if score >= 85:
                st.markdown(f"<div class='grader-pass'><strong>PASS ({score}/100)</strong><br><br>{'<br>'.join(feedback)}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='grader-fail'><strong>NEEDS WORK ({score}/100)</strong><br><br>{'<br>'.join(feedback)}<br><br><b>Tip:</b> Try rewriting your answer to include the missing elements above.</div>", unsafe_allow_html=True)
