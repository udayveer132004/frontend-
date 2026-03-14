# AI Job Assistant - Resume Parser, ATS, Chat & Interview

AI-powered job assistance platform for resume analysis, job search, and AI mock interviews.

> **Note**: For detailed technical specifications, dependency versions, and model internals, please read [Project details.md](Project%20details.md).

## Features ✅

### 1. Resume Parsing
- 📄 **Upload PDF/DOCX resumes**
- 🤖 **AI-Powered Extraction** - Uses Ollama with **qwen3.5:2b** for intelligent parsing
- 📊 **Structured Data**: Extracts detailed Contact Info, Education, Experience, Projects, and Skills.
- 🛠️ **Debug Info**: View raw model output, thought process, and parsed JSON.

### 2. ATS Scoring
- 🎯 **Score Calculation**: Analytics against a Job Description.
- 🔍 **Keyword Analysis**: Identifies missing and matching keywords.
- 📈 **Visualizations**: Gauge and Radar charts for quick analysis.

### 3. Chat with Resume (RAG)
- 💬 **Interactive Chat**: Ask questions about the candidate based on the resume.
- 🧠 **RAG Engine**: Uses local vector store (Qdrant) and Embeddings (Sentence-Transformers).

### 4. Job Search & Matching
- 🔎 **Remote Job Search**: Fetches jobs from Remote OK, We Work Remotely, Jobicy, and Remotive.
- ⚡ **Semantic Matching**: Matches your resume profile against job descriptions.

### 5. AI Mock Interviewer
- 🗣️ **Voice Interaction**: Real-time voice-to-voice interview practice.
- 📝 **Deferred Feedback**: Receive a comprehensive performance report collecting all questions and scores at the end.
- ⚙️ **Configurable Models**: Supports Qwen3-ASR, Faster Whisper, and Wav2Vec2.

## Setup

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai/) with model `qwen3.5:2b`
- **NVIDIA GPU** (Recommended for Whisper STT)

### Installation

1. **Clone and navigate to project**:
   ```bash
   git clone <repo-url>
   cd ai-job-assistant
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you have an NVIDIA GPU, verify that `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` are installed for Faster Whisper support.*

4. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ollama pull qwen3.5:2b
   ```

5. **(Optional) Setup Gemini**:
   Create a `.env` file and add:
   ```env
   GEMINI_API_KEY=your_key_here
   ```

## Usage

### Run the Application

```bash
python main.py
```

The Gradio interface will open at: `http://127.0.0.1:7860`

### Using the Interface

1. **Resume Parser**: Upload -> Parse.
2. **ATS Scorer**: Paste JD -> View Score.
3. **Chat**: QA with your resume.
4. **Job Search**: Find matching roles.
5. **Mock Interview**: Click "Start Interview", allow microphone, answer questions, and click "End & Report" to see your score.

## Project Structure

```
x:\Project\
├── backend/
│   ├── ats/            # ATS Scoring Logic
│   ├── chat/           # RAG Engine & Chat Logic
│   ├── common/         # Shared Pydantic Models
│   ├── interview/      # Audio Processing & Interview Engine
│   ├── job_portal/     # Job Search APIs
│   └── resume_parsing/ # Resume Extraction
├── main.py             # Application Entry Point
├── Project details.md  # Deep technical documentation
└── requirements.txt    # Dependencies
```

## License

MIT
