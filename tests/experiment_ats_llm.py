"""
Experimental Script: Direct LLM ATS Scoring with Gradio & Streaming
Tests the idea of using the LLM to analyze Resume JSON vs Job Description directly for comprehensive scoring.
Updated for google-genai SDK.
"""

import json
import logging
import requests
import os
import gradio as gr
from dotenv import load_dotenv

from google import genai
from google.genai import types

# Load env vars
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SAMPLE DATA
SAMPLE_RESUME = {
  "name": "Praveen Kumar",
  "email": "praveenrepswal77@gmail.com",
  "phone": "+91 8000522943",
  "location": "Rajasthan, India",
  "linkedin": "",
  "github": "",
  "portfolio": "",
  "skills": [
    "Python", "SQL", "MySQL", "JavaScript", "HTML/CSS", "Git", "Docker",
    "VS Code", "Visual Studio", "GitHub", "Linux", "Pandas", "NumPy",
    "Matplotlib", "Selenium", "Flask", "Streamlit", "Requests", "LangChain",
    "Ollama", "FAISS", "Excel", "Power BI", "C++", "Quantitative Modeling",
    "Time Series Analysis", "Portfolio Optimization", "Backtesting",
    "Alpha Generation", "LSTM", "Feature Engineering", "Neural Networks",
    "FPGA Trading", "XGBoost"
  ],
  "education": [
    "Amity University Rajasthan, Rajasthan, India - Bachelor of Technology in Computer Science, Minor in Business, Aug. 2022 – July 2026",
    "Royal Academy, Rajasthan, India - 12th, Aug. 2021 – May 2022"
  ],
  "experience": [],
  "summary": "",
  "achievements": [],
  "certifications": [
    "Career Essentials in Data Analysis by Microsoft and LinkedIn...",
    "LinkedIn Certified Marketing Insider...",
    "Code in Place by Stanford University..."
  ],
  "projects": [
    "AI-Powered Opportunity Aggregator...",
    "Walmart Sales Data Analysis...",
    "LeetCard..."
  ],
  "publications": [],
  "languages": [],
  "volunteer": [],
  "awards": [],
  "interests": [],
  "ai_summary": "Praveen Kumar is an aspiring Computer Science professional...",
  "key_strengths": ["Proficient in Python...", "Expertise in data analysis..."]
}

SAMPLE_JD = """
Job Title: ML / AI Engineer
Experience Level: New Graduate to 2 Years
Location: Remote / Hybrid / Flexible

We’re seeking an entry-level ML/AI Engineer...
"""

def stream_ollama_response(prompt: str, model: str = "qwen3:4b"):
    """
    Generator that streams response from Ollama.
    Yields (thinking_content, answer_content)
    """
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': True,
        'think': True,
        'options': {'temperature': 0.2, 'num_predict': 8192}
    }
    
    try:
        response = requests.post(
            'http://localhost:11434/api/chat',
            json=payload,
            stream=True,
            timeout=300
        )
        response.raise_for_status()
        
        thinking_buffer = ""
        content_buffer = ""
        
        for line in response.iter_lines(decode_unicode=True):
            if not line: continue
            try:
                data = json.loads(line)
                message = data.get('message', {})
                if 'thinking' in message:
                    thinking_buffer += message['thinking']
                if 'content' in message:
                    content_buffer += message['content']
                yield thinking_buffer, content_buffer
                if data.get('done'):
                    break
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        yield f"Error: {str(e)}", ""

def stream_gemini_response(prompt: str, model: str = "gemini-2.5-flash"):
    """
    Generator that streams response from Gemini via google-genai.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        yield "", "❌ Error: GEMINI_API_KEY not found in environment variables."
        return

    try:
        client = genai.Client(api_key=api_key)
        
        # generate_content_stream
        response = client.models.generate_content_stream(
             model=model,
             contents=prompt,
             config=types.GenerateContentConfig(
                 temperature=0.2
             )
        )
        
        content_buffer = ""
        for chunk in response:
            if chunk.text:
                content_buffer += chunk.text
                yield "Gemini processes internally (no chain-of-thought exposed)...", content_buffer
                
    except Exception as e:
        yield "", f"❌ Gemini Error: {str(e)}"

def run_experiment(resume_json_str, jd_text, provider, model_name):
    """Grad-io handler."""
    try:
        resume_data = json.loads(resume_json_str)
    except Exception as e:
        yield "", f"❌ Invalid JSON in Resume Data: {e}"
        return

    prompt = f"""
    You are an expert ATS (Applicant Tracking System) Scorer.
    Analyze the following CANDIDATE RESUME against the JOB DESCRIPTION.
    Give your reply only in JSON format. Your response shouldn't contain anything other than a JSON object.
    
    JOB DESCRIPTION:
    {jd_text}
    
    CANDIDATE RESUME:
    {json.dumps(resume_data, indent=2)}
    
    TASK:
    1. Calculate a compatibility score (0-100).
    2. Identify Matching Key Skills.
    3. Identify Missing Critical Skills.
    4. Provide specific improvement suggestions.
    
    OUTPUT FORMAT (Valid JSON only):
    {{
        "match_score": x,
        "match_level": "High/Medium/Low",
        "reasoning": "Brief explanation...",
        "technical_skills_match": ["Skill1"],
        "missing_critical_skills": ["Missing1"],
        "improvements": ["Suggestion 1"]
    }}
    
    Strictly follow the output format. Your reply should follow the fomatting rules of the output format.
    Return your answer in ONLY JSON format.
    """
    
    if provider == "gemini":
        for thinking, answer in stream_gemini_response(prompt, model=model_name):
             yield thinking, answer
    else:
        for thinking, answer in stream_ollama_response(prompt, model=model_name):
             yield thinking, answer

# Create Gradio Interface
with gr.Blocks(title="Experimental ATS LLM Scoring") as demo:
    gr.Markdown("# 🧪 Experimental ATS LLM Scoring")
    gr.Markdown("Directly prompt the LLM to analyze Resume JSON vs Job Description.")
    
    with gr.Row():
        with gr.Column():
            resume_input = gr.Code(
                label="Resume Data (JSON)", 
                value=json.dumps(SAMPLE_RESUME, indent=2), 
                language="json",
                lines=15
            )
            jd_input = gr.TextArea(
                label="Job Description", 
                value=SAMPLE_JD, 
                lines=10
            )
            
            with gr.Row():
                provider_input = gr.Dropdown(
                    label="Provider", choices=["ollama", "gemini"], value="ollama"
                )
                model_input = gr.Textbox(
                    label="Model Name", value="qwen3:4b"
                )
            
            def update_default_model(p):
                return "gemini-2.5-flash" if p == "gemini" else "qwen3:4b"
            
            provider_input.change(update_default_model, inputs=provider_input, outputs=model_input)
            
            btn = gr.Button("🚀 Run Analysis", variant="primary")
            
        with gr.Column():
            thinking_output = gr.Textbox(
                label="💭 Thinking Process (Chain of Thought)", 
                lines=10,
                interactive=False
            )
            result_output = gr.Code(
                label="📄 Final Analysis result (JSON)", 
                language="json",
                lines=15
            )

    btn.click(
        fn=run_experiment,
        inputs=[resume_input, jd_input, provider_input, model_input],
        outputs=[thinking_output, result_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
