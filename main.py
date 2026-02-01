"""AI Job Assistant - Resume Parser & ATS Scorer & RAG Chat

Phase 3: Chat With Resume (RAG)
Includes Resume Parsing, ATS Scoring, and Interactive Chat.
Fixed for Gradio 6.0 and updated Google GenAI SDK.
"""

import logging
import json
import plotly.graph_objects as go
import gradio as gr
from dotenv import load_dotenv

from resume_parser import ResumeParser, ResumeData
from resume_parser.ats_scorer import ATSScorer, ATSResult
from resume_parser.ai_extractor import check_ollama_connection
from resume_parser.job_search import search_jobs
from resume_parser.job_matcher import JobMatcher
from resume_parser.interview_engine import InterviewManager
try:
    from resume_parser.rag_engine import RAGEngine
except ImportError:
    RAGEngine = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global storage
current_resume_data = None
rag_engine_instance = None

def get_rag_engine():
    """Singleton accessor for RAG Engine."""
    global rag_engine_instance
    if rag_engine_instance is None:
        if RAGEngine:
            try:
                rag_engine_instance = RAGEngine()
            except Exception as e:
                logger.error(f"Failed to init RAG Engine: {e}")
                return None
        else:
            logger.warning("RAGEngine class not imported (dependencies missing?)")
            return None
    return rag_engine_instance

def parse_resume(file, provider: str, model: str):
    """
    Parse uploaded resume and return structured JSON data.
    Also indexes text for RAG.
    """
    global current_resume_data
    
    if file is None:
        return json.dumps({"error": "Please upload a resume file"}, indent=2), "", ""
    
    try:
        selected_model = model
        if provider == "gemini":
            if not selected_model or selected_model.startswith("qwen"):
                selected_model = "gemini-2.5-flash"
        
        parser = ResumeParser(model=selected_model, provider=provider)
        logger.info(f"Processing file: {file.name}")
        
        result_tuple, raw_text = parser.parse(file.name)
        
        if isinstance(result_tuple, tuple):
            resume_data, debug_raw = result_tuple
        else:
            resume_data = result_tuple
            debug_raw = ""
            
        debug_info = parser.get_debug_info()
        
        if resume_data:
            current_resume_data = resume_data
            data_dict = resume_data.model_dump(exclude_none=False)
            
            rag = get_rag_engine()
            if rag and raw_text:
                rag.ingest_text(raw_text, metadata={"source": file.name})
                debug_info += "\n\n[RAG] Resume indexed for chat."
            
            # Auto-fill role for job search
            suggested_role = ""
            if resume_data.suggested_roles:
                suggested_role = resume_data.suggested_roles[0]
            
            return json.dumps(data_dict, indent=2, ensure_ascii=False), raw_text, debug_info, suggested_role
        else:
            return json.dumps({"error": "AI extraction failed"}, indent=2), raw_text, debug_info, ""
            
    except Exception as e:
        logger.error(f"Error parsing resume: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, indent=2), "", "", ""

def calculate_ats_score(job_description: str, provider: str, model: str):
    """Calculate ATS score and generate visualizations."""
    global current_resume_data
    
    if not current_resume_data:
        return (None, None, None, "⚠️ Please parse a resume first.", "")
    
    if not job_description or len(job_description) < 50:
        return (None, None, None, "⚠️ Please enter a valid Job Description.", "")
    
    try:
        selected_model = model
        if provider == "gemini": selected_model = "gemini-2.5-flash"
            
        scorer = ATSScorer(model=selected_model, provider=provider)
        result = scorer.calculate_score(current_resume_data, job_description)
        
        # 1. Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = result.score,
            title = {'text': "ATS Score"},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "darkblue"},
                     'steps': [{'range': [0, 50], 'color': "lightgray"}, {'range': [50, 80], 'color': "gray"}],
                     'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}}
        ))
        
        # 2. Radar Chart
        categories = list(result.breakdown.keys())
        max_values = {"Keywords": 40, "Skills": 30, "Formatting": 10, "Education": 10, "Experience": 10}
        normalized_values = [(result.breakdown[k] / max_values.get(k, 10)) * 100 for k in categories]
        
        fig_radar = go.Figure(data=go.Scatterpolar(r=normalized_values, theta=categories, fill='toself'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Category Performance (%)")
        
        # 3. Text Outputs
        missing_kw_md = "### ❌ Missing Keywords\n"
        missing_kw_md += " ".join([f"`{kw}`" for kw in result.missing_keywords]) if result.missing_keywords else "✅ None!"
            
        suggestions_md = "### 💡 Suggestions\n" + ("\n".join([f"- {s}" for s in result.suggestions]) if result.suggestions else "Looks good!")
            
        return fig_gauge, fig_radar, missing_kw_md, suggestions_md, json.dumps(result.to_dict(), indent=2)
        
    except Exception as e:
        logger.error(f"Error calculating ATS score: {e}", exc_info=True)
        return None, None, None, f"Error: {str(e)}", ""

def chat_response(message, history, provider, model):
    """
    Handle RAG Chat interaction.
    """
    rag = get_rag_engine()
    if not rag:
        yield "⚠️ Chat engine not initialized (Parsing resume might be required or deps missing).", "", "", ""
        return

    if not rag.vector_store:
         yield "⚠️ No resume content found. Please parse a resume first!", "", "", ""
         return

    try:
        selected_model = model
        if provider == "gemini":
             selected_model = "gemini-2.5-flash"
             
        # Prepare all chunks for display
        all_chunks_str = json.dumps(rag.all_chunks, indent=2) if rag.all_chunks else "[]"

        for chunk, context, prompt in rag.query(message, provider=provider, model=selected_model):
            yield chunk, context, prompt, all_chunks_str
            
    except Exception as e:
        yield f"Error: {str(e)}", "", "", ""

def check_system_status() -> str:
    status = "✅ System Ready\n" if check_ollama_connection() else "❌ Ollama Connection Failed\n"
    if get_rag_engine():
        status += "✅ RAG Engine Loaded"
    else:
        status += "⏳ RAG Engine Loading/Missing"
    return status

def create_job_cards(jobs):
    """Generate HTML cards for jobs."""
    if not jobs:
        return "<div style='padding: 20px; text-align: center; color: gray;'>No jobs found. Try different keywords.</div>"
        
    html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px;'>"
    
    for job in jobs:
        score_badge = ""
        if 'match_score' in job:
            score = job['match_score']
            color = "#48bb78" if score > 75 else "#ecc94b" if score > 50 else "#a0aec0"
            score_badge = f"<div style='position: absolute; top: 10px; right: 10px; background: {color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;'>{score}% Match</div>"

        card = f"""
        <div style="position: relative; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            {score_badge}
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                <h3 style="margin: 0; font-size: 16px; color: #2d3748; padding-right: 60px;">{job['title']}</h3>
            </div>
                <span style="font-size: 12px; background: #e2e8f0; padding: 2px 6px; border-radius: 4px;">{job['date']}</span>
            </div>
            <div style="font-weight: bold; color: #4a5568; margin-bottom: 8px;">{job['company']}</div>
            <div style="font-size: 14px; color: #718096; margin-bottom: 12px;">📍 {job['location']}</div>
            
            <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px;">
                {' '.join([f"<span style='background: #ebf8ff; color: #2b6cb0; font-size: 10px; padding: 2px 6px; border-radius: 10px;'>{tag}</span>" for tag in job['tags'][:3]])}
            </div>
            
            <div style="margin-top: auto;">
                <a href="{job['apply_url'] or job['url']}" target="_blank" style="display: block; text-align: center; background: #3182ce; color: white; padding: 8px; border-radius: 6px; text-decoration: none; font-weight: bold;">Apply Now 🚀</a>
            </div>
            <div style="font-size: 10px; color: #a0aec0; text-align: center; margin-top: 8px;">
                via Remote OK
            </div>
        </div>
        """
        html += card
        
    html += "</div>"
    return html

# State to store last fetched jobs for proper ranking
last_fetched_jobs = []

def search_jobs_wrapper(role, location):
    """Wrapper for job search to return HTML."""
    global last_fetched_jobs
    if not role:
        return "⚠️ Please enter a job role."
    
    try:
        jobs = search_jobs(role, location)
        last_fetched_jobs = jobs # Cache for ranking
        return create_job_cards(jobs)
    except Exception as e:
        return f"Error searching jobs: {str(e)}"

def rank_jobs_wrapper():
    """Rank the last fetched jobs against the current resume."""
    global last_fetched_jobs
    
    # 1. Check prerequisites
    rag = get_rag_engine()
    if not rag or not rag.embedding_model:
        return "⚠️ Matching Engine not ready. (RAG Engine missing)"
        
    if not current_resume_data or not rag.all_chunks:
        # We need resume text. We can get it from rag.all_chunks or we need to store raw text globally.
        # main.py does 'return ... raw_text' in parse_resume but doesn't store raw_text globally except for rag ingest.
        # Hack: Re-construct text from rag chunks or current_resume_data summary/skills?
        # Better: Use the raw text we ingested. rag_engine doesn't expose full text easily if chunked.
        # Let's use `current_resume_data` fields as the proxy for resume text if raw missing.
        pass

    # Re-construct resume text for matching
    resume_text = ""
    if current_resume_data:
        # Rich representation
        resume_text = f"{current_resume_data.ai_summary or ''} {' '.join(current_resume_data.skills)} {' '.join(current_resume_data.experience)}"
    
    if not resume_text:
        return "⚠️ No resume data found. Please parse a resume first."
        
    if not last_fetched_jobs:
        return "⚠️ No jobs to rank. Search for jobs first."

    try:
        matcher = JobMatcher(rag.embedding_model)
        ranked_jobs = matcher.match_jobs(resume_text, last_fetched_jobs)
        last_fetched_jobs = ranked_jobs # Update cache
        return create_job_cards(ranked_jobs)
    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        return f"Error ranking jobs: {str(e)}"

# State for Mock Interview
interview_manager = InterviewManager(output_dir="temp_audio")

def start_interview_wrapper():
    """Initialize interview session."""
    # Context from resume
    resume_context = ""
    if current_resume_data:
        resume_context = f"{current_resume_data.ai_summary} {' '.join(current_resume_data.skills)}"
    else:
        resume_context = "General Software Engineering candidate."
        
    q_text, audio_path, _ = interview_manager.start_interview(resume_context)
    
    # Return: History Text, Audio Output, Feedback Text
    history_html = f"<div class='chat-msg bot'><b>AI Interviewer:</b> {q_text}</div>"
    return history_html, audio_path, ""

def handle_interview_response(audio_path):
    """Process user audio response."""
    if not audio_path:
        return None, None, None
        
    resume_context = "General Context"
    if current_resume_data:
        resume_context = f"{current_resume_data.ai_summary} {' '.join(current_resume_data.skills)}"

    # Call engine
    user_text, next_q, audio_file, filler_audio = interview_manager.handle_turn(audio_path, resume_context)
    
    # Update Chat UI
    # We need to append to existing history but Gradio functions are usually stateless unless using State component.
    # For now, we reconstruct from manager.history
    
    chat_html = ""
    for msg in interview_manager.history:
        # Handle LangChain Message objects
        if hasattr(msg, 'type'):
            if msg.type == 'human':
                role = "User"
                cls = "user"
            elif msg.type == 'ai':
                role = "AI Interviewer"
                cls = "bot"
            else:
                # Skip system messages in UI
                continue
            content = msg.content
        else:
            # Fallback for dicts (backwards compatibility if needed)
            role = "User" if msg.get('role') == 'user' else "AI Interviewer"
            cls = "user" if msg.get('role') == 'user' else "bot"
            content = msg.get('content', '')

        chat_html += f"<div class='chat-msg {cls}'><b>{role}:</b> {content}</div>"
        
    feedback = interview_manager.get_latest_feedback()
    
    return chat_html, audio_file, feedback

def create_demo_interface():
    custom_css = """
    .gradio-container { max-width: 1200px !important; } 
    .output-json { font-family: monospace; font-size: 12px; }
    .chat-msg { padding: 10px; margin: 5px; border-radius: 8px; color: #1a202c !important; }
    .bot { background-color: #f0f4f8; border-left: 4px solid #4299e1; color: #2d3748 !important; }
    .user { background-color: #e2e8f0; border-right: 4px solid #48bb78; text-align: right; color: #2d3748 !important; }
    """
    
    # Removed theme and css from Blocks to avoid warning in Gradio 6.0
    with gr.Blocks(title="AI Job Assistant") as demo:
        gr.Markdown("# 🤖 AI Job Assistant")
        
        with gr.Row():
            status_btn = gr.Button("🔍 Check System Status", variant="secondary")
            status_output = gr.Textbox(label="Status", lines=2, interactive=False)
            status_btn.click(fn=check_system_status, outputs=status_output)
            
        with gr.Row():
            provider_input = gr.Dropdown(label="Provider", choices=["ollama", "gemini"], value="ollama")
            model_input = gr.Textbox(label="Model", value="qwen3:4b", placeholder="qwen3:4b or gemini-2.5-flash")
        
        def _update_model(provider):
            return "gemini-2.5-flash" if provider == "gemini" else "qwen3:4b"
        provider_input.change(_update_model, inputs=provider_input, outputs=model_input)
            
        with gr.Tabs():
            with gr.TabItem("📄 Resume Parser"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(label="Upload Resume", file_types=[".pdf", ".docx"])
                        parse_btn = gr.Button("🚀 Parse Resume", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("Structured Data"): json_output = gr.Code(label="JSON", language="json", lines=20)
                            with gr.TabItem("Raw Text"): text_output = gr.Textbox(label="Text", lines=20)
                            with gr.TabItem("Debug"): debug_output = gr.Textbox(label="Debug Info", lines=20)
                # parse_btn click is now handled at the end to include job_role_input output
                # parse_btn.click(parse_resume, inputs=[file_input, provider_input, model_input], outputs=[json_output, text_output, debug_output])
            
            with gr.TabItem("🎯 ATS Scorer"):
                gr.Markdown("### Calculate ATS Score based on Job Description")
                jd_input = gr.TextArea(label="Paste Job Description Here", lines=10)
                score_btn = gr.Button("📊 Calculate Score", variant="primary", size="lg")
                with gr.Row():
                    with gr.Column(): gauge_chart = gr.Plot(label="Overall Score")
                    with gr.Column(): radar_chart = gr.Plot(label="Breakdown")
                with gr.Row():
                    missing_kw_output = gr.Markdown(label="Missing Keywords")
                    suggestions_output = gr.Markdown(label="Suggestions")
                with gr.Accordion("Raw Scoring Data", open=False):
                    ats_raw_output = gr.Code(language="json")
                score_btn.click(calculate_ats_score, inputs=[jd_input, provider_input, model_input], 
                                outputs=[gauge_chart, radar_chart, missing_kw_output, suggestions_output, ats_raw_output])

            # TAB 3: CHAT (RAG)
            with gr.TabItem("💬 Chat with Resume"):
                gr.Markdown("### Ask questions about the uploaded resume")
                
                # Expanders for RAG Debugging
                with gr.Accordion("📚 Retrieved Context", open=False):
                    context_output = gr.Markdown(label="Chunks Used")
                
                with gr.Accordion("🤖 Full Prompt", open=False):
                    prompt_output = gr.Code(label="Prompt Sent to LLM", language="markdown")

                with gr.Accordion("📂 All Document Chunks", open=False):
                    chunks_output = gr.Code(label="All Text Splits", language="json")
                
                # Chat Interface with additional outputs
                chat_interface = gr.ChatInterface(
                    fn=chat_response,
                    additional_inputs=[provider_input, model_input],
                    additional_outputs=[context_output, prompt_output, chunks_output],
                    title="Resume Chat"
                )

            # TAB 4: JOB SEARCH
            with gr.TabItem("🌍 Job Search"):
                gr.Markdown("### Find Remote Jobs")
                gr.Markdown("Search for jobs via **Remote OK** API. Data provided by [Remote OK](https://remoteok.com/).")
                
                with gr.Row():
                    job_role_input = gr.Textbox(label="Job Role / Keywords", placeholder="e.g. Python Developer", scale=2)
                    job_location_input = gr.Textbox(label="Location (Optional)", placeholder="e.g. USA, Europe (Remote OK uses broad tags)", scale=1)
                    job_search_btn = gr.Button("🔍 Find Jobs", variant="primary", scale=1)
                
                # Match score button (Initially hidden or just visible)
                rank_btn = gr.Button("🧠 Rank by Resume Match", variant="secondary")
                
                jobs_output = gr.HTML(label="Job Results")
                
                job_search_btn.click(
                    search_jobs_wrapper,
                    inputs=[job_role_input, job_location_input],
                    outputs=jobs_output
                )
                
                rank_btn.click(
                    rank_jobs_wrapper,
                    inputs=[],
                    outputs=jobs_output
                )

            # TAB 5: MOCK INTERVIEW
            with gr.TabItem("🎤 Mock Interview"):
                gr.Markdown("### AI-Powered Mock Interview")
                gr.Markdown("Practice your interview skills with an AI that listens, speaks, and analyzes your answers in real-time.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        interview_chat = gr.HTML(label="Interview Transcript", value="<div style='color:gray'>Press Start to begin...</div>")
                        
                    with gr.Column(scale=1):
                        start_interview_btn = gr.Button("▶️ Start Interview", variant="primary")
                        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Your Answer")
                        audio_output = gr.Audio(label="AI Interviewer", autoplay=True)
                        feedback_display = gr.Markdown(label="Real-time Feedback")

                # Events
                start_interview_btn.click(
                    start_interview_wrapper,
                    inputs=[],
                    outputs=[interview_chat, audio_output, feedback_display]
                )
                
                audio_input.stop_recording(
                    handle_interview_response,
                    inputs=[audio_input],
                    outputs=[interview_chat, audio_output, feedback_display]
                )
                
                # Update parse_resume to auto-fill job role
                # Note: parse_btn is defined in Tab 1, but we can update its outputs here or modifying the original click
                # Since we are inside the same Blocks context, we can modify the previous click or add a new one? 
                # No, we must update the original click in Tab 1 to include job_role_input in outputs.
                # Let's re-define the click for Tab 1 here? No, better to do it where it was defined or update it.
                # I will update the Tab 1 definitions in the next chunk/replacement.
                
                # Update Tab 1 click to target job_role_input
                parse_btn.click(
                    parse_resume, 
                    inputs=[file_input, provider_input, model_input], 
                    outputs=[json_output, text_output, debug_output, job_role_input]
                )

    return demo, custom_css  # Return CSS to pass to launch()

def main():
    demo, css = create_demo_interface()
    # Pass css and theme to launch() as requested by Gradio 6.0 warning
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, css=css) # theme arg might not be in launch yet, checking docs... 
    # Actually, theme IS usually in Blocks in Gradio 4/5. 
    # If Gradio 6 moved it, I will just suppress it for now by not setting it in launch (using default) 
    # or pass it if I knew the API. Safe bet: define Blocks(..., theme=...) and ignore warning if it works, 
    # BUT warning said "theme, css" moved to launch. 
    # Let's try passing 'css' to launch. 'theme' object might be harder to pass if launch expects name string vs object.
    # Leaving theme default for safety.

if __name__ == "__main__":
    main()
