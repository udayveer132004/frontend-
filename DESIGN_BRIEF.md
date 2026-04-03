# AI Job Assistant - Design Brief

## 1. Product Summary
AI Job Assistant is a desktop-first web app (Gradio) that helps job seekers across the full application journey:
- Parse resumes into structured profile data
- Evaluate resume fit against a job description (ATS score)
- Chat with resume content (RAG)
- Search remote jobs and rank by resume match
- Track job applications and view analytics
- Practice AI-driven mock interviews (voice)

Core value proposition: one workflow from resume understanding to interview practice.

## 2. Primary Users
- Students and new graduates applying for internships or entry-level roles
- Early/mid-career professionals applying to remote software roles
- Users who need fast resume feedback and interview prep in one tool

User goals:
- Understand how strong their resume is for a target role
- Find relevant jobs quickly
- Organize and track applications
- Improve interview performance with practice and feedback

## 3. Product Scope (Current)
Single-page app with top-level controls and tabbed workspaces.

Global controls (always visible):
- System Status button
- Provider selector: Ollama or Gemini
- Model selector (depends on provider)
- Think toggle (reasoning mode)

Main tabs:
1. Resume Parser
2. ATS Scorer
3. Chat with Resume
4. Job Search
5. Application Tracker
6. Mock Interview

## 4. Information Architecture
High-level flow:
1. User uploads resume in Resume Parser
2. Parsed data is stored and indexed for chat/matching
3. User runs ATS Scorer with a job description
4. User asks follow-up questions in Chat with Resume
5. User searches jobs and optionally ranks them by match
6. User tracks application progress in Tracker
7. User practices in Mock Interview and ends with report

Critical dependency: Resume parsing should feel like the first milestone because ATS, chat, and ranking all depend on it.

## 5. Screen-by-Screen Requirements

### A) Resume Parser Tab
Purpose:
- Ingest resume and expose structured output + transparency/debug views

Main UI elements:
- File upload (.pdf, .docx)
- Primary CTA: Parse Resume
- Output sub-tabs:
  - Structured Data (JSON code view)
  - Raw Text
  - Debug Info

Design considerations:
- Emphasize successful parse state
- Show loading/progress feedback during extraction
- Make debug details collapsible/secondary for non-technical users

### B) ATS Scorer Tab
Purpose:
- Compare parsed resume against pasted job description

Main UI elements:
- Large text area for job description
- Primary CTA: Calculate Score
- Two visual outputs:
  - Gauge chart (overall score)
  - Radar chart (category performance)
- Text outputs:
  - Missing keywords
  - Suggestions
- Collapsible raw JSON scoring data

Design considerations:
- Score should be the immediate visual anchor
- Missing keywords should be scannable as chips/tags
- Suggestions should read as actionable checklist items

### C) Chat with Resume Tab
Purpose:
- Conversational Q&A over resume content

Main UI elements:
- Chat interface (user/assistant turns)
- Collapsible debug sections:
  - Retrieved Context
  - Full Prompt
  - All Document Chunks

Design considerations:
- Keep chat area roomy and readable
- Distinguish normal conversation vs debug artifacts
- Make it clear chat answers are grounded in uploaded resume

### D) Job Search Tab
Purpose:
- Find remote jobs and optionally rank by resume relevance

Main UI elements:
- Inputs:
  - Job Role / Keywords
  - Location (optional)
- CTA: Find Jobs
- Secondary CTA: Rank by Resume Match
- Results as responsive cards including:
  - Title, company, location, date
  - Tags/skills
  - Match score badge (when ranked)
  - Apply link
  - Source label

Design considerations:
- Card hierarchy: role and company first, metadata second
- Match score should be visible but not overpower core job info
- Ensure cards work well on smaller widths

### E) Application Tracker Tab
Sub-tab 1: Manage
- Stats cards: total, interviews, offers, rejections
- Application table with row numbers and status badge
- Add form (title, company, role type, status, date, URL, notes)
- Edit/Delete workflow using row number load

Sub-tab 2: Analytics
- Donut chart: applications by role type
- Area chart: cumulative applications over time
- Refresh controls

Design considerations:
- Treat this as lightweight CRM
- Status color coding must be consistent across cards/table/charts
- Editing should feel safe and explicit (prevent accidental delete)

### F) Mock Interview Tab
Purpose:
- Real-time voice interview practice with final report

Main UI elements:
- Interview transcript panel
- Start and End & Report controls
- Microphone audio input
- AI audio output playback
- Final report panel
- Debug trace panel

Design considerations:
- Strong session-state clarity (idle, recording, processing, speaking)
- Transcript should remain readable over longer sessions
- Final report should be easy to scan and compare over time

## 6. Key Data Objects for UI

Resume profile fields:
- name, email, phone, location, linkedin, github, portfolio
- skills, education, experience
- summary, achievements, certifications, projects, publications
- languages, volunteer, awards, interests
- ai_summary, key_strengths, suggested_roles

Application tracker fields:
- job_title, company, role_type, status, application_date
- job_url, notes, created_at, updated_at

Job card fields:
- title, company, location, date, tags
- apply_url/url, source, match_score (optional)

## 7. States and Edge Cases to Design
Global/system:
- Ollama unavailable
- Provider/model mismatch
- Long-running model response

Parser:
- No file uploaded
- Unsupported or unreadable file
- Parse failure with recoverable guidance

ATS:
- No parsed resume available
- Job description too short
- Scoring API/model failure

Chat:
- Resume not indexed yet
- No context found for a question
- Streaming response interruption

Job search:
- Empty search query
- No jobs found
- API source partial failure (some sources unavailable)

Tracker:
- Empty table state
- Invalid row selection for edit
- Update/delete target not found

Interview:
- Microphone permission denied
- Audio not captured
- Mid-session model failure
- End report when session not started

## 8. UX Priorities
- Progressive disclosure: keep advanced/debug info available but not primary
- Fast first success: guide user to parse resume first
- Explain dependencies: ATS/chat/rank require parsed resume
- Keep confidence high with clear success and error messages
- Reduce cognitive load with clean sectioning and visual hierarchy

## 9. Visual Direction (Suggested)
- Tone: professional, supportive, high-clarity career tool
- Personality: confident assistant, not playful chatbot
- Visual style: modern productivity dashboard
- Color behavior:
  - Neutral surfaces for readability
  - Semantic accents for status (success/warning/error/info)
  - Distinct but accessible chart palette

Typography goals:
- Strong hierarchy for tab content and result metrics
- Monospace treatment for JSON/debug data
- Comfortable body text for long descriptions and transcripts

## 10. Accessibility and Responsiveness
- WCAG-aware contrast for all status colors and chart labels
- Keyboard focus visibility on all controls
- Large click/tap targets for primary actions
- Responsive behavior for job cards and table overflow
- Captions/labels for audio actions and interview states

## 11. Deliverables Expected from Designer
- End-to-end user flow map
- Low-fidelity wireframes for all 6 tabs
- High-fidelity UI for key states (empty/loading/success/error)
- Component kit:
  - Status badges
  - Metric cards
  - Job cards
  - Form controls
  - Chart containers
  - Chat bubbles/transcript blocks
- Interaction notes for:
  - Parse -> ATS/chat/rank dependency guidance
  - Interview session lifecycle
  - Tracker edit/delete safety patterns

## 12. Handoff Notes for Development
- Current UI is Gradio-based, so component behavior should be practical for Python-rendered web UI
- Existing app already uses charts (gauge, radar, donut, area) and card-style HTML blocks
- Designs should prioritize clarity and implementation feasibility over heavy visual effects
