import streamlit as st
import google.generativeai as genai
import os
import json
import pandas as pd
from pypdf import PdfReader
from dotenv import load_dotenv
from supabase import create_client, Client

# --- SETUP ---
st.set_page_config(page_title="TalentAlign AI", layout="wide", page_icon="👔")

# Load Keys
api_key = os.getenv("GEMINI_API_KEY")
supa_url = os.getenv("SUPABASE_URL")
supa_key = os.getenv("SUPABASE_KEY")

# Fallback for local dev
if not api_key:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    supa_url = os.getenv("SUPABASE_URL")
    supa_key = os.getenv("SUPABASE_KEY")

# Initialize Supabase
supabase: Client = None
if supa_url and supa_key:
    try:
        supabase = create_client(supa_url, supa_key)
    except Exception as e:
        print(f"Db Error: {e}")

# --- FUNCTIONS ---
def save_to_db(job_title, candidate_name, score, summary):
    """Saves the scan result to Supabase."""
    if not supabase:
        return
    
    try:
        data = {
            "job_title": job_title,
            "candidate_name": candidate_name,
            "match_score": score,
            "summary": summary
        }
        supabase.table("resume_scans").insert(data).execute()
    except Exception as e:
        print(f"Failed to save to DB: {e}")

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        
        if len(text.strip()) < 50:
            return None
            
        return text.strip()
    except Exception as e:
        return None

def analyze_resumes(jd, resumes, blind_mode=False):
    if not api_key:
        return {"error": "API Key missing."}
    
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
    except:
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})

    prompt = f"""
    You are an Expert Technical Recruiter.
    JOB DESCRIPTION: {jd}
    TASK: Analyze candidates.
    {'BLIND MODE: Ignore names/gender.' if blind_mode else ''}
    OUTPUT FORMAT (JSON list):
    [{{ "name": "Name", "match_score": 85, "key_skills": [], "missing_skills": [], "summary": "...", "status": "Interview" }}]
    RESUMES:
    """
    
    for i, res in enumerate(resumes):
        prompt += f"\n--- CANDIDATE {i+1} ({res['name']}) ---\n{res['text']}\n"

    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"AI Error: {str(e)}"}

# --- UI LAYOUT ---
with st.sidebar:
    st.title("⚙️ Controls")
    blind_mode = st.toggle("Blind Hiring Mode", value=True)
    st.info("Powered by **Gemini 2.5**")
    
    # DB Status Indicator
    if supabase:
        st.success("🟢 Database Connected")
    else:
        st.warning("🔴 Database Offline")

st.title("👔 TalentAlign: AI Recruitment Agent")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Job Description")
    jd_input = st.text_area("Paste JD here:", height=300)

with col2:
    st.subheader("2. Resumes")
    uploaded_files = st.file_uploader("Upload PDF Resumes", type="pdf", accept_multiple_files=True)

if st.button("🚀 Screen Candidates", type="primary"):
    if not jd_input or not uploaded_files:
        st.warning("Please upload Resumes and paste a JD.")
    else:
        with st.spinner("Analyzing..."):
            valid_resumes = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                if text:
                    valid_resumes.append({"name": file.name, "text": text})
            
            if not valid_resumes:
                st.error("No valid text found.")
            else:
                results = analyze_resumes(jd_input, valid_resumes, blind_mode)
                
                if "error" in results:
                    st.error(results['error'])
                else:
                    st.success("Analysis Complete!")
                    
                    # Scoreboard
                    df = pd.DataFrame(results)
                    st.dataframe(df[["name", "match_score", "status", "summary"]], use_container_width=True)
                    
                    # Save to DB & Display Cards
                    for cand in results:
                        # Save each candidate to Supabase
                        save_to_db(
                            jd_input[:50], # Save first 50 chars of JD title
                            cand.get('name', 'Unknown'),
                            cand.get('match_score', 0),
                            cand.get('summary', '')
                        )
                        
                        with st.expander(f"{cand.get('match_score')}% - {cand.get('name')}"):
                            st.write(f"**Verdict:** {cand.get('status')}")
                            st.write(f"**Summary:** {cand.get('summary')}")
