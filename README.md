# AI Resume Screener – Automated LLM-Powered Screening App

**By Owadokun Tosin Tobi**  
Physics Graduate | AI Prompt Engineer | Content Creator  
**Repo**: https://github.com/Eatosin/ai-resume-screener  
**Date**: December 15, 2025  

## Project Overview (STAR Format)
- **Situation**: In 2025, recruiters receive 250+ resumes per role, struggling with volume overload, unconscious bias, keyword gaming, and inaccurate fit prediction. Traditional ATS tools are rigid and often unfair.
- **Task**: Build a simple, automated web app that screens resumes fairly using LLMs for semantic understanding, debiasing, and explainable scoring.
- **Action**: Developed a Streamlit app that handles PDF uploads, extracts text, runs custom LLM prompts (Mistral-7B-Instruct), and outputs ranked results with justifications.
- **Result**: Screens multiple resumes in seconds, reduces simulated bias via blind evaluation, provides clear explanations. Tested on my resume against AI Trainer JD → scored 92/100 with transferable skills highlighted.

This standalone project showcases prompt engineering, PDF processing, automation, and ethical AI—perfect for AI Trainer / data annotation roles.

## Live Demo (Recommended)
[instant live demo](https://unbiased-ai-resume-screener.streamlit.app)

## Features
- Upload Job Description (text area)
- Upload multiple Resume PDFs
- Automatic text extraction (PyPDF2)
- LLM screening with debiasing & structured JSON output
- Ranked results: Score, Explanation, Strengths, Gaps
- Local/offline capable (free Hugging Face model)

## How to Run Locally
1. Clone: `git clone https://github.com/Eatosin/ai-resume-screener.git`
2. Install: `pip install -r requirements.txt`
3. Run: `streamlit run resume_screener_app.py`
4. Open browser → Paste JD + upload resumes → Screen!

**Note**: First run downloads Mistral-7B (~14GB). Needs 16GB+ RAM (GPU optional). For lighter testing, change model to "gpt2-medium".

## Files in This Repo
- `resume_screener_app.py` → Main app code
- `requirements.txt` → Dependencies
- `sample_resume.pdf` → My resume (for testing—replace with yours)
- `sample_jd.txt` → Example AI Trainer job description

## Future Enhancements
- Add OpenAI/Grok API option (via env key)
- CSV export of rankings
- Batch processing improvements
- Fine-tune prompts on real recruiter feedback

Star ⭐ this repo if helpful! Open to collaborations or AI Trainer opportunities.  
Connect: [@TosinOwadokun on X](https://x.com/TosinOwadokun) | [LinkedIn](https://www.linkedin.com/in/owadokun-tosin-tobi-6159091a3)  
Part of my AI portfolio:[High Quality AI Content Creator Tool](https://github.com/Eatosin/ai-content-creation-portfolio)
