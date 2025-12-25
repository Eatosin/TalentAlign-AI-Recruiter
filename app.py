import streamlit as st
import google.generativeai as genai
import os
import json
import pandas as pd
from pypdf import PdfReader
from dotenv import load_dotenv

# --- SETUP ---
st.set_page_config(page_title="TalentAlign AI", layout="wide", page_icon="👔")

# Load API Key (Cloud or Local)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

# --- FUNCTIONS ---
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text.strip()
    except Exception as e:
        return None

def analyze_resumes(jd, resumes, blind_mode=False):
    if not api_key:
        return {"error": "API Key missing. Please set GEMINI_API_KEY in Settings."}
    
    genai.configure(api_key=api_key)
    
    # Robust Model Selection
    try:
        # Try SOTA first
        model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
    except:
        # Fallback to Pro if 2.5 isn't available on key
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})

    # The Prompt
    prompt = f"""
    You are an Expert Technical Recruiter.
    
    JOB DESCRIPTION:
    {jd}
    
    TASK:
    Analyze the candidates.
    {'IMPORTANT: BLIND MODE ACTIVE. Ignore candidate names, age, gender, and university names to reduce bias.' if blind_mode else ''}
    
    OUTPUT FORMAT (JSON list of objects):
    [
        {{
            "name": "Candidate Name (or 'Candidate X' if blind)",
            "match_score": 85,
            "key_skills": ["Python", "AWS"],
            "missing_skills": ["Docker"],
            "summary": "Brief explanation of the score.",
            "status": "Interview"
        }}
    ]
    
    RESUMES:
    """
    
    for i, res in enumerate(resumes):
        prompt += f"\n--- CANDIDATE {i+1} ({res['name']}) ---\n{res['text']}\n"

    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"AI Processing Error: {str(e)}"}

# --- UI LAYOUT ---
with st.sidebar:
    st.title("⚙️ TalentAlign Controls")
    blind_mode = st.toggle("Blind Hiring Mode", value=True, help="Instructs AI to ignore demographic markers.")
    st.info("Powered by **Gemini AI**")
    
st.title("👔 TalentAlign: AI Recruitment Agent")
st.markdown("Upload resumes and let AI rank them against your Job Description.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Job Description")
    jd_input = st.text_area("Paste JD here:", height=300, placeholder="e.g. We are looking for a Senior Python Engineer...")

with col2:
    st.subheader("2. Resumes")
    uploaded_files = st.file_uploader("Upload PDF Resumes", type="pdf", accept_multiple_files=True)

if st.button("🚀 Screen Candidates", type="primary"):
    if not jd_input or not uploaded_files:
        st.warning("Please upload Resumes and paste a JD.")
    else:
        with st.spinner("Extracting text and analyzing profiles..."):
            valid_resumes = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                if text and len(text) > 50:
                    valid_resumes.append({"name": file.name, "text": text})
            
            if not valid_resumes:
                st.error("No valid text found. Are they scanned images?")
            else:
                results = analyze_resumes(jd_input, valid_resumes, blind_mode)
                
                if "error" in results:
                    st.error(results['error'])
                else:
                    st.success("Analysis Complete!")
                    
                    # Scoreboard
                    df = pd.DataFrame(results)
                    if not df.empty:
                        st.dataframe(
                            df[["name", "match_score", "status", "summary"]].sort_values("match_score", ascending=False),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Cards
                        st.markdown("---")
                        for cand in sorted(results, key=lambda x: x.get('match_score', 0), reverse=True):
                            with st.expander(f"{cand.get('match_score', 0)}% - {cand.get('name', 'Unknown')}"):
                                st.write(f"**Verdict:** {cand.get('status', 'Unknown')}")
                                st.write(f"**Summary:** {cand.get('summary', '')}")
                                c1, c2 = st.columns(2)
                                c1.success(f"**Skills:** {', '.join(cand.get('key_skills', []))}")
                                c2.error(f"**Missing:** {', '.join(cand.get('missing_skills', []))}")
