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

from datetime import datetime
from functools import partial

# Backend Imports
from backend.common.models import ResumeData
from backend.resume_parsing.parser import ResumeParser
from backend.resume_parsing.ai_extractor import check_ollama_connection
from backend.interview.engine import InterviewManager
from backend.job_portal.search import search_jobs
from backend.job_portal.matcher import JobMatcher
from backend.chat.rag_engine import RAGEngine
from backend.ats.scorer import calculate_ats_score_gemini, ATSScorer
from backend.tracker.tracker import ApplicationTracker, VALID_STATUSES, VALID_ROLE_TYPES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
# Load environment variables
load_dotenv()

# Cleanup on exit
import atexit
import shutil
import os

def cleanup_temp_files():
    """Delete temp_audio directory on exit."""
    temp_dir = "temp_audio"
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to cleanup temp directory: {e}")

atexit.register(cleanup_temp_files)

# Global storage
current_resume_data = None
rag_engine_instance = None
app_tracker = ApplicationTracker(storage_path="data/applications.json")

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

# ─── Application Tracker helpers ─────────────────────────────────────────────

STATUS_COLORS = {
    "Applied": ("#dbeafe", "#1d4ed8"),
    "Awaiting Interview": ("#fef3c7", "#92400e"),
    "Interviewed": ("#ede9fe", "#5b21b6"),
    "Offered": ("#d1fae5", "#065f46"),
    "Rejected": ("#fee2e2", "#991b1b"),
    "Withdrawn": ("#f3f4f6", "#374151"),
}


def _status_badge(status: str) -> str:
    bg, fg = STATUS_COLORS.get(status, ("#f3f4f6", "#374151"))
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:600;">'
        f"{status}</span>"
    )


def build_tracker_stats_html(stats: dict) -> str:
    cards = [
        ("📋", stats["total"], "Applications", "#3b82f6"),
        ("🤝", stats["interviews"], "Interviews", "#8b5cf6"),
        ("🎉", stats["offers"], "Offers", "#10b981"),
        ("✖", stats["rejections"], "Rejections", "#ef4444"),
    ]
    html = '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px;">'
    for icon, val, label, color in cards:
        html += (
            f'<div style="border:2px solid {color};border-radius:12px;padding:16px 24px;'
            f'min-width:130px;text-align:center;background:#fff;box-shadow:0 2px 6px rgba(0,0,0,.06);">'
            f'<div style="font-size:1.8em;">{icon}</div>'
            f'<div style="font-size:2em;font-weight:700;color:{color};">{val}</div>'
            f'<div style="color:#6b7280;font-size:0.9em;">{label}</div>'
            "</div>"
        )
    html += "</div>"
    return html


def build_tracker_table_html(rows: list) -> str:
    if not rows:
        return '<p style="color:#6b7280;padding:16px;">No applications yet. Add one below!</p>'

    header_cells = "".join(
        f'<th style="padding:10px 14px;background:#f8fafc;border-bottom:2px solid #e2e8f0;text-align:left;font-size:13px;color:#374151;">'
        f"{h}</th>"
        for h in ["#", "Job Title", "Company", "Role Type", "Status", "Date", "Link"]
    )
    rows_html = ""
    for row in rows:
        num, title, company, role_type, status, app_date, url, _ = row
        link_cell = (
            f'<a href="{url}" target="_blank" '
            f'style="color:#3b82f6;text-decoration:none;">🔗 Link</a>'
            if url
            else "—"
        )
        bg = "#fff" if num % 2 == 1 else "#f9fafb"
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:10px 14px;border-bottom:1px solid #f0f0f0;color:#6b7280;">{num}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid #f0f0f0;font-weight:600;color:#111827;">{title}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid #f0f0f0;color:#374151;">{company}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid #f0f0f0;color:#374151;">{role_type}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid #f0f0f0;">{_status_badge(status)}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid #f0f0f0;color:#374151;">{app_date}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid #f0f0f0;">{link_cell}</td>'
            "</tr>"
        )
    return (
        '<div style="overflow-x:auto;">'
        '<table style="width:100%;border-collapse:collapse;font-size:14px;">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
    )


def refresh_tracker():
    stats = app_tracker.get_stats()
    rows = app_tracker.to_dataframe_rows()
    return build_tracker_stats_html(stats), build_tracker_table_html(rows)


def add_application_wrapper(title, company, role_type, status, app_date, url, notes):
    if not title or not title.strip():
        stats_html, table_html = refresh_tracker()
        return stats_html, table_html, "⚠️ Job Title is required."
    if not company or not company.strip():
        stats_html, table_html = refresh_tracker()
        return stats_html, table_html, "⚠️ Company is required."
    try:
        app_tracker.add(
            job_title=title,
            company=company,
            role_type=role_type,
            status=status,
            application_date=app_date or None,
            job_url=url or None,
            notes=notes or None,
        )
        stats_html, table_html = refresh_tracker()
        return stats_html, table_html, f"✅ Added '{title}' at {company}."
    except Exception as e:
        stats_html, table_html = refresh_tracker()
        return stats_html, table_html, f"❌ Error: {e}"


def load_row_for_edit(row_num):
    """Load application data by 1-based row number into the edit form."""
    try:
        idx = int(row_num) - 1
    except (TypeError, ValueError):
        return "", "", "Full-time", "Applied", "", "", "", "", "⚠️ Enter a valid row number."
    rows = app_tracker.to_dataframe_rows()
    if idx < 0 or idx >= len(rows):
        return "", "", "Full-time", "Applied", "", "", "", "", "⚠️ Row number out of range."
    _, title, company, role_type, status, app_date, url, app_id = rows[idx]
    entry = app_tracker.get_by_id(app_id)
    return (
        title, company, role_type, status, app_date,
        url, entry.notes or "", app_id, f"🔄 Loaded row {row_num}: {title} @ {company}"
    )


def save_edit_wrapper(app_id, title, company, role_type, status, app_date, url, notes):
    if not app_id:
        stats_html, table_html = refresh_tracker()
        return stats_html, table_html, "⚠️ No application loaded. Use 'Load Row' first."
    updated = app_tracker.update(
        app_id,
        job_title=title, company=company, role_type=role_type, status=status,
        application_date=app_date, job_url=url, notes=notes,
    )
    stats_html, table_html = refresh_tracker()
    if updated:
        return stats_html, table_html, f"✅ Updated '{updated.job_title}' at {updated.company}."
    return stats_html, table_html, "❌ Update failed — entry not found."


def delete_application_wrapper(app_id):
    if not app_id:
        stats_html, table_html = refresh_tracker()
        return stats_html, table_html, "⚠️ No application loaded. Use 'Load Row' first.", ""
    entry = app_tracker.get_by_id(app_id)
    label = f"{entry.job_title} @ {entry.company}" if entry else app_id
    success = app_tracker.delete(app_id)
    stats_html, table_html = refresh_tracker()
    if success:
        return stats_html, table_html, f"🗑️ Deleted: {label}", ""
    return stats_html, table_html, "❌ Delete failed — entry not found.", app_id


# ── Analytics charts ──────────────────────────────────────────────────────────

ROLE_TYPE_COLORS = [
    "#3b82f6",  # Full-time      – blue
    "#8b5cf6",  # Part-time      – violet
    "#f59e0b",  # Internship     – amber
    "#10b981",  # Contract       – emerald
    "#ef4444",  # Freelance      – red
    "#6b7280",  # other fallback – gray
]


def build_donut_fig():
    """Plotly donut chart – applications by Role Type."""
    apps = app_tracker.get_all()
    counts: dict[str, int] = {}
    for a in apps:
        counts[a.role_type] = counts.get(a.role_type, 0) + 1

    if not counts:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#6b7280"),
            xref="paper", yref="paper",
        )
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    labels = list(counts.keys())
    values = list(counts.values())
    total = sum(values)
    colors = ROLE_TYPE_COLORS[: len(labels)]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
            textinfo="label+percent",
            textposition="outside",
            insidetextorientation="radial",
        )
    )
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:12px;color:#6b7280'>Total</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=22),
        xref="paper", yref="paper",
    )
    fig.update_layout(
        title=dict(text="Applications by Role Type", font=dict(size=15), x=0.5),
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5),
        height=400,
        margin=dict(t=50, b=20, l=20, r=140),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_area_fig():
    """Plotly cumulative area chart – applications added over time by application date."""
    from datetime import datetime as _dt

    apps = app_tracker.get_all()
    if not apps:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#6b7280"),
            xref="paper", yref="paper",
        )
        fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    # Sort by application_date
    dated = []
    for a in apps:
        try:
            d = _dt.fromisoformat(a.application_date).date()
        except Exception:
            import datetime
            d = datetime.date.today()
        dated.append(d)
    dated.sort()

    # Build cumulative series grouped by unique date
    from collections import Counter
    per_day = Counter(dated)
    sorted_days = sorted(per_day.keys())
    x_vals, y_vals, cumulative = [], [], 0
    for day in sorted_days:
        cumulative += per_day[day]
        x_vals.append(str(day))
        y_vals.append(cumulative)

    fig = go.Figure(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers+text",
            fill="tozeroy",
            fillcolor="rgba(74, 180, 130, 0.25)",
            line=dict(color="#4ab482", width=2.5, shape="spline"),
            marker=dict(color="#4ab482", size=7),
            text=[str(v) for v in y_vals],
            textposition="top center",
        )
    )
    fig.update_layout(
        title=dict(text="Cumulative Applications Over Time", font=dict(size=15), x=0.5),
        xaxis=dict(title="Application Date", showgrid=False),
        yaxis=dict(title="Count", gridcolor="#e5e7eb"),
        height=380,
        margin=dict(t=50, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def refresh_analytics():
    return build_donut_fig(), build_area_fig()


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
    .gradio-container .message { color: #1a202c !important; }
    .gradio-container .message.user { background-color: #e2e8f0 !important; color: #1a202c !important; }
    .gradio-container .message.bot,
    .gradio-container .message.assistant { background-color: #f0f4f8 !important; color: #1a202c !important; }
    .gradio-container .message .prose,
    .gradio-container .message .prose p,
    .gradio-container .message .prose li,
    .gradio-container .message .prose span,
    .gradio-container .message .prose strong,
    .gradio-container .message .prose code { color: #1a202c !important; }
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

            # TAB 5: APPLICATION TRACKER
            with gr.TabItem("📋 Application Tracker"):
                gr.Markdown("### Track your job applications in one place")

                with gr.Tabs():
                    # ── Sub-tab 1: Manage ──────────────────────────────────
                    with gr.TabItem("📋 Manage"):

                        # Stats row
                        initial_stats, initial_table = refresh_tracker()
                        tracker_stats = gr.HTML(value=initial_stats, label="Stats")
                        refresh_tracker_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")

                        # Applications table
                        tracker_table = gr.HTML(value=initial_table, label="Applications")

                        # Add new application
                        with gr.Accordion("➕ Add New Application", open=True):
                            with gr.Row():
                                add_title    = gr.Textbox(label="Job Title *", placeholder="e.g. Software Engineer", scale=2)
                                add_company  = gr.Textbox(label="Company *", placeholder="e.g. Google", scale=2)
                                add_roletype = gr.Dropdown(label="Role Type", choices=VALID_ROLE_TYPES, value="Full-time", scale=1)
                                add_status   = gr.Dropdown(label="Status", choices=VALID_STATUSES, value="Applied", scale=1)
                            with gr.Row():
                                add_date = gr.Textbox(label="Application Date (YYYY-MM-DD)", placeholder="Leave blank for today", scale=1)
                                add_url  = gr.Textbox(label="Job URL (optional)", placeholder="https://...", scale=3)
                            add_notes      = gr.Textbox(label="Notes (optional)", lines=2, placeholder="Any notes about the role...")
                            add_btn        = gr.Button("➕ Add Application", variant="primary")
                            add_status_msg = gr.Markdown("")

                        # Edit / Delete
                        with gr.Accordion("✏️ Edit / Delete Application", open=False):
                            gr.Markdown("Enter the **row number** from the table above, then click *Load Row* to populate the form.")
                            with gr.Row():
                                edit_row_num = gr.Number(label="Row #", value=1, minimum=1, step=1, scale=1)
                                load_row_btn = gr.Button("📂 Load Row", scale=1)
                            edit_load_msg = gr.Markdown("")
                            edit_id_state = gr.State("")
                            with gr.Row():
                                edit_title    = gr.Textbox(label="Job Title", scale=2)
                                edit_company  = gr.Textbox(label="Company", scale=2)
                                edit_roletype = gr.Dropdown(label="Role Type", choices=VALID_ROLE_TYPES, value="Full-time", scale=1)
                                edit_status   = gr.Dropdown(label="Status", choices=VALID_STATUSES, value="Applied", scale=1)
                            with gr.Row():
                                edit_date = gr.Textbox(label="Application Date (YYYY-MM-DD)", scale=1)
                                edit_url  = gr.Textbox(label="Job URL", scale=3)
                            edit_notes = gr.Textbox(label="Notes", lines=2)
                            with gr.Row():
                                save_edit_btn    = gr.Button("💾 Save Changes", variant="primary")
                                delete_entry_btn = gr.Button("🗑️ Delete Entry", variant="stop")
                            edit_action_msg = gr.Markdown("")

                        # Event wiring – Manage tab
                        refresh_tracker_btn.click(
                            refresh_tracker,
                            inputs=[],
                            outputs=[tracker_stats, tracker_table],
                        )
                        add_btn.click(
                            add_application_wrapper,
                            inputs=[add_title, add_company, add_roletype, add_status, add_date, add_url, add_notes],
                            outputs=[tracker_stats, tracker_table, add_status_msg],
                        )
                        load_row_btn.click(
                            load_row_for_edit,
                            inputs=[edit_row_num],
                            outputs=[
                                edit_title, edit_company, edit_roletype, edit_status,
                                edit_date, edit_url, edit_notes, edit_id_state, edit_load_msg,
                            ],
                        )
                        save_edit_btn.click(
                            save_edit_wrapper,
                            inputs=[edit_id_state, edit_title, edit_company, edit_roletype,
                                    edit_status, edit_date, edit_url, edit_notes],
                            outputs=[tracker_stats, tracker_table, edit_action_msg],
                        )
                        delete_entry_btn.click(
                            delete_application_wrapper,
                            inputs=[edit_id_state],
                            outputs=[tracker_stats, tracker_table, edit_action_msg, edit_id_state],
                        )

                    # ── Sub-tab 2: Analytics ───────────────────────────────
                    with gr.TabItem("📈 Analytics"):
                        gr.Markdown("Visual breakdown of your applications.")

                        analytics_refresh_btn = gr.Button("🔄 Refresh Charts", variant="secondary", size="sm")

                        with gr.Row():
                            donut_chart = gr.Plot(label="Applications by Role Type")
                            area_chart  = gr.Plot(label="Cumulative Applications Over Time")

                        # Render charts on load
                        _init_donut, _init_area = refresh_analytics()
                        donut_chart.value = _init_donut
                        area_chart.value  = _init_area

                        # Event wiring – Analytics tab
                        analytics_refresh_btn.click(
                            refresh_analytics,
                            inputs=[],
                            outputs=[donut_chart, area_chart],
                        )
                        # Also refresh charts whenever an application is added/edited/deleted
                        add_btn.click(
                            refresh_analytics,
                            inputs=[],
                            outputs=[donut_chart, area_chart],
                        )
                        save_edit_btn.click(
                            refresh_analytics,
                            inputs=[],
                            outputs=[donut_chart, area_chart],
                        )
                        delete_entry_btn.click(
                            refresh_analytics,
                            inputs=[],
                            outputs=[donut_chart, area_chart],
                        )

            # TAB 6: MOCK INTERVIEW
            with gr.TabItem("🎤 Mock Interview"):
                gr.Markdown("### AI-Powered Mock Interview")
                gr.Markdown("Practice your interview skills with an AI that listens, speaks, and analyzes your answers in real-time.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        interview_chat = gr.HTML(label="Interview Transcript", value="<div style='color:gray'>Press Start to begin...</div>")
                        
                    with gr.Column(scale=1):
                        with gr.Row():
                            start_interview_btn = gr.Button("▶️ Start", variant="primary")
                            end_interview_btn = gr.Button("⏹️ End & Report", variant="secondary")
                        
                        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Your Answer")
                        audio_output = gr.Audio(label="AI Interviewer", autoplay=True)
                        feedback_display = gr.Markdown(label="Interview Report")

                # Events
                def start_interview_wrapper():
                    q, audio, _ = interview_manager.start_interview(resume_context="Based on resume...")
                    chat_html = f"<div class='chat-msg bot'><b>AI Interviewer:</b> {q}</div>"
                    # Clear feedback on start
                    return chat_html, audio, ""

                def end_interview_wrapper():
                    report = interview_manager.end_interview()
                    return report

                def handle_interview_response_wrapper(audio_path):
                    chat_html, audio_file, _ = handle_interview_response(audio_path)
                    # Don't show feedback immediately, return empty string
                    return chat_html, audio_file, ""

                start_interview_btn.click(
                    start_interview_wrapper,
                    inputs=[],
                    outputs=[interview_chat, audio_output, feedback_display]
                )
                
                end_interview_btn.click(
                    end_interview_wrapper,
                    inputs=[],
                    outputs=[feedback_display]
                )
                
                audio_input.stop_recording(
                    handle_interview_response_wrapper,
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
