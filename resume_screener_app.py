import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import torch
import json

# Load strong local model (Mistral-7B-Instruct – excellent for structured output)
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

generator = load_model()

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def screen_resumes(jd_text, resume_texts):
    prompt_template = """
    [INST] You are a fair, expert recruiter screening blindly (ignore names, gender, age, personal details).
    Job Description: {jd}
    
    For each resume, output ONLY valid JSON like:
    [{"resume_id": 1, "overall_score": 0-100, "explanation": "2-3 sentences on fit", "strengths": ["item1", "item2"], "gaps": ["item1", "item2"]}, ...]
    
    Resumes:
    {resumes}
    [/INST]
    """
    
    formatted_resumes = ""
    for i, text in enumerate(resume_texts):
        formatted_resumes += f"Resume {i+1}:\n{text[:3500]}\n\n"  # Safe truncation
    
    full_prompt = prompt_template.format(jd=jd_text, resumes=formatted_resumes)
    
    with st.spinner("AI screening in progress (Mistral-7B)..."):
        raw_output = generator(full_prompt, max_new_tokens=1000, temperature=0.3)[0]['generated_text']
    
    # Extract JSON
    try:
        start = raw_output.find("[")
        end = raw_output.rfind("]") + 1
        json_str = raw_output[start:end]
        results = json.loads(json_str)
    except Exception as e:
        results = [{"error": f"JSON parse failed: {str(e)}", "raw": raw_output}]
    
    return results

# UI
st.set_page_config(page_title="AI Resume Screener", layout="centered")
st.title("🚀 AI Resume Screener App")
st.markdown("Simple, fair, explainable screening powered by LLMs – by Owadokun Tosin Tobi")

jd_text = st.text_area("Job Description (required)", height=150, placeholder="Paste full JD here... e.g., AI Trainer role requiring prompt engineering, content creation...")

uploaded_resumes = st.file_uploader(
    "Upload Resume PDFs (multiple OK)",
    type="pdf",
    accept_multiple_files=True
)

if st.button("🔍 Screen Resumes", type="primary"):
    if jd_text and uploaded_resumes:
        with st.spinner("Extracting text..."):
            resume_texts = [extract_text_from_pdf(f) for f in uploaded_resumes]
        
        results = screen_resumes(jd_text, resume_texts)
        
        st.success("Screening Complete!")
        
        # Sort by score
        if "error" not in results[0]:
            results = sorted(results, key=lambda x: x.get("overall_score", 0), reverse=True)
        
        for res in results:
            if "error" in res:
                st.error("Issue with output:")
                st.code(res["raw"])
            else:
                st.markdown(f"### Resume {res['resume_id']} – **Score: {res['overall_score']}/100**")
                st.write("**Explanation**: " + res["explanation"])
                st.write("**Strengths**: " + ", ".join(res["strengths"]))
                st.write("**Gaps**: " + ", ".join(res["gaps"]))
                st.divider()
    else:
        st.warning("Add a job description and at least one resume.")

st.caption("Local Mistral-7B model – first run downloads model. Built for portfolio; ethical/fair use only.")