# AI Job Assistant - Project Technical Report

## 1. Project Overview
This project is an AI-powered Job Assistant that provides:
- **Resume Parsing**: extracting structured data from PDF/DOCX.
- **RAG Chat**: chatting with your resume using local embeddings.
- **Job Search**: finding relevant remote jobs.
- **Mock Interview**: voice-based AI interviewer with feedback.

## 2. Technology Stack & Dependencies

### Core Frameworks
| Library | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.12+ | Core Language |
| **Gradio** | 6.5.1 | Web UI Framework |
| **FastAPI** | 0.128.0 | Backend API Framework |
| **Uvicorn** | 0.40.0 | ASGI Server |

### AI & Machine Learning
| Library | Version | Purpose |
| :--- | :--- | :--- |
| **LangChain** | 0.4.1+ | LLM Orchestration (Community/Core/Ollama) |
| **Transformers** | 5.0.0 | Hugging Face Models (STT) |
| **Torch** | 2.10.0 | Deep Learning Backend |
| **Faster-Whisper** | 1.2.1 | Optimized Speech-to-Text |
| **Sentence-Transformers** | 3.3.1 | Embedding Models |
| **Qdrant-Client** | 1.16.2 | Vector Database (In-Memory) |
| **Google-GenAI** | 0.6.15 | Google Gemini API Client |

### Utilities
| Library | Version | Purpose |
| :--- | :--- | :--- |
| **NVIDIA cuBLAS/cuDNN** | 12.x | GPU Acceleration libraries |
| **PyPDF / PDFPlumber** | 6.6.2 / 0.11.9 | PDF Extraction |
| **Python-Docx** | 1.2.0 | DOCX Extraction |
| **Requests** | 2.32.5 | HTTP Client (Job Search, Ollama API) |

## 3. Model Specifications

### Large Language Models (LLM)
1.  **Primary (Local)**: `qwen3:4b`
    *   **Provider**: Ollama
    *   **Usage**: Resume parsing, Chat, Interview logic.
2.  **Fallback/Cloud**: `gemini-2.5-flash`
    *   **Provider**: Google Gemini API
    *   **Usage**: Resume parsing (fallback), ATS Scoring.

### Speech-to-Text (STT) Models
The project supports multiple STT engines, configurable in `backend/interview/engine.py`.

1.  **Qwen3-ASR (Default)**
    *   **Model**: `Qwen/Qwen3-ASR-0.6B`
    *   **Source**: Hugging Face Transformers
    *   **Size**: ~0.6B params (~1.2GB)
    *   **Pros**: Balanced speed/accuracy, Multilingual.

2.  **Faster Whisper**
    *   **Model**: `small`
    *   **Source**: `faster-whisper` (CTranslate2)
    *   **Size**: ~500MB (Int8 quantization)
    *   **Pros**: High accuracy, optimized for CPU/GPU.
    *   **Note**: Includes automatic CPU fallback if GPU initialization fails.

3.  **Wav2Vec2**
    *   **Model**: `facebook/wav2vec2-base-960h`
    *   **Source**: Hugging Face Transformers
    *   **Size**: ~95M params (~378MB)
    *   **Pros**: Extremely fast on CPU, English only.

### Text-to-Speech (TTS)
*   **Engine**: `PocketTTS` (Lightweight wrapper)
*   **Voice**: `eponine` (Kokoro-style)
*   **Output**: WAV files (cached in `temp_audio/`)

### RAG & Embeddings
*   **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
    *   **Dimensions**: 384
    *   **Storage**: Qdrant (In-Memory Collection: `resume_chat`)

## 4. Setup Instructions

### Prerequisites
*   Python 3.10 or higher (3.12 recommended).
*   **Ollama** installed and running (`ollama serve`).
*   **NVIDIA GPU** (Optional but recommended for STT).

### Installation
1.  **Clone the Repository**
2.  **Create Virtual Environment**:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # Windows
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    # OR manually install key libs if requirements.txt is missing:
    pip install gradio langchain-ollama transformers faster-whisper qdrant-client sentence-transformers google-genai
    ```
4.  **Install GPU Support (If NVIDIA GPU present)**:
    ```bash
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
    ```
5.  **Pull Ollama Model**:
    ```bash
    ollama pull qwen3:4b
    ```

### Configuration (.env)
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_key
```

### Running the Application
```bash
python main.py
```
Access the UI at `http://127.0.0.1:7860`.

## 5. Project Structure
```
x:\Project\
├── backend/
│   ├── ats/            # ATS Scoring Logic
│   ├── chat/           # RAG Engine & Chat Logic
│   ├── common/         # Shared Models (Pydantic)
│   ├── interview/      # Audio Processing & Interview Engine
│   ├── job_portal/     # Job Search APIs
│   └── resume_parsing/ # Resume Extraction (Ollama/Gemini)
├── main.py             # Application Entry Point (Gradio UI)
├── requirements.txt    # (Recommended to generate)
└── .env                # Secrets
```
