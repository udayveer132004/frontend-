"""
Interview Engine for AI Mock Interview.
Refactored to use LangChain for robust chat handling and CLI-based EdgeTTS to avoid Windows asyncio conflicts.
"""

import os
import json
import logging
import subprocess
import threading
import time
import traceback

from datetime import datetime
from typing import List, Dict, Any, Tuple
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "-1"))

class InterviewManager:
    def __init__(self, output_dir="temp_audio", model: str = "qwen3.5:2b", think: bool = True):
        self._add_nvidia_paths()
        self.history = [] # LangChain Message History
        self.scores: List[Dict[str, Any]] = []
        self.current_question = ""
        self.output_dir = output_dir
        self.model = model
        self.think = think
        self.debug_events: List[str] = []

        # Initialize Chat Model via LangChain
        # Temperature 0.2 for creativity but stability
        self.llm = self._build_llm()
        self._debug(f"InterviewManager initialized with model={self.model}, think={self.think}")

        # Load Whisper (Tiny)
        logger.info("Loading Whisper Model (tiny)...")
        self.stt_model = None # Disable Whisper

        # Initialize Wav2Vec2 (CPU friendly) via Pipeline
        logger.info("Loading Wav2Vec2 Pipeline...")
        try:
            from transformers import pipeline
            self.asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
            self.asr_init_error = None
            logger.info("Wav2Vec2 Ready!")
        except Exception as e:
            self.asr_init_error = str(e)
            logger.error(f"Failed to load Wav2Vec2 Pipeline: {e}")
            self.asr_pipeline = None

        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Pocket TTS
        logger.info("Loading Pocket TTS Model...")
        try:
            from pocket_tts import TTSModel
            import scipy.io.wavfile
            
            self.tts_model = TTSModel.load_model()
            self.tts_voice_state = self.tts_model.get_state_for_audio_prompt("eponine") # Pre-made voice
            self.scipy_wavfile = scipy.io.wavfile
            logger.info("Pocket TTS Ready!")
        except Exception as e:
            logger.error(f"Failed to load Pocket TTS: {e}")
            self.tts_model = None
            self.tts_voice_state = None
        
        # Pre-generate filler audio
        self.filler_audio_path = os.path.join(output_dir, "filler.wav")
        if not os.path.exists(self.filler_audio_path):
             self._generate_tts("That's an interesting point, let me think about the next question.", self.filler_audio_path)

    def _debug(self, message: str) -> None:
        """Store structured debug events and mirror to logger for interview diagnostics."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] {message}"
        self.debug_events.append(line)
        if len(self.debug_events) > 500:
            self.debug_events = self.debug_events[-500:]
        logger.info(f"[INTERVIEW_DEBUG] {message}")

    def clear_debug_trace(self) -> None:
        self.debug_events = []

    def get_debug_trace(self) -> str:
        if not self.debug_events:
            return "No interview debug events yet."
        return "\n".join(self.debug_events)

    @staticmethod
    def _safe_file_size(path: str) -> int:
        try:
            return os.path.getsize(path) if path and os.path.exists(path) else 0
        except Exception:
            return 0

    def _generate_tts_with_timeout(self, text: str, output_path: str, timeout_sec: int = 20) -> str:
        """Generate TTS with timeout so UI callbacks don't hang indefinitely."""
        self._debug(f"TTS timeout wrapper start; timeout={timeout_sec}s, output={output_path}")
        result = {"path": "", "error": None}

        def _worker():
            try:
                result["path"] = self._generate_tts(text, output_path)
            except Exception as e:
                result["error"] = str(e)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout_sec)

        if t.is_alive():
            logger.warning(f"TTS generation timed out after {timeout_sec}s. Returning text response without audio.")
            self._debug("TTS timeout reached; returning without audio")
            return ""

        if result["error"]:
            logger.error(f"TTS generation failed in worker: {result['error']}")
            self._debug(f"TTS worker failed: {result['error']}")
            return ""

        self._debug(f"TTS timeout wrapper completed; audio_path={result['path']}")
        return result["path"] or ""

    def _invoke_llm_with_timeout(self, messages, timeout_sec: int, label: str, llm=None):
        """Invoke LLM in a worker thread with a hard timeout to avoid UI hangs."""
        result = {"response": None, "error": None}
        target_llm = llm or self.llm

        def _worker():
            try:
                result["response"] = target_llm.invoke(messages)
            except Exception as e:
                result["error"] = e

        self._debug(f"{label}: LLM invoke worker start; timeout={timeout_sec}s")
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout_sec)

        if t.is_alive():
            self._debug(f"{label}: LLM invoke timeout reached")
            raise TimeoutError(f"LLM invoke timed out after {timeout_sec}s")

        if result["error"] is not None:
            self._debug(f"{label}: LLM invoke raised error: {result['error']}")
            raise result["error"]

        self._debug(f"{label}: LLM invoke worker finished")
        return result["response"]

    def _build_llm(self) -> ChatOllama:
        """Create a ChatOllama instance with current model configuration."""
        return ChatOllama(
            model=self.model,
            temperature=0.2,
            options={"num_gpu": OLLAMA_NUM_GPU, "num_predict": 180},
            model_kwargs={"think": self.think},
        )

    def _build_scorer_llm(self) -> ChatOllama:
        """Create a lightweight LLM instance for background scoring only."""
        return ChatOllama(
            model=self.model,
            temperature=0.0,
            options={"num_gpu": OLLAMA_NUM_GPU, "num_predict": 80},
            model_kwargs={"think": False},
        )

    def configure_llm(self, model: str, think: bool = True) -> None:
        """Update interview LLM model/options at runtime from UI selection."""
        model = (model or "").strip() or self.model
        if model == self.model and think == self.think:
            return
        self.model = model
        self.think = think
        self.llm = self._build_llm()
        logger.info(f"Interview LLM configured: model={self.model}, think={self.think}")
        self._debug(f"LLM configured: model={self.model}, think={self.think}")

    def _add_nvidia_paths(self):
        """Add nvidia library paths to environment for faster-whisper."""
        try:
            import site
            import glob
            
            site_packages = site.getsitepackages()
            for sp in site_packages:
                # Look for nvidia/*/lib or bin
                nvidia_base = os.path.join(sp, 'nvidia')
                if os.path.exists(nvidia_base):
                    for root, dirs, files in os.walk(nvidia_base):
                        if 'bin' in dirs:
                            bin_path = os.path.join(root, 'bin')
                            if bin_path not in os.environ['PATH']:
                                os.environ['PATH'] += os.pathsep + bin_path
                        if 'lib' in dirs:
                            lib_path = os.path.join(root, 'lib')
                            if lib_path not in os.environ['PATH']:
                                os.environ['PATH'] += os.pathsep + lib_path
        except Exception as e:
            logger.warning(f"Failed to add NVIDIA paths: {e}")

    # ...

    def transcribe_audio_wav2vec2(self, audio_path: str) -> str:
        """Convert audio file to text using Wav2Vec2 Pipeline."""
        if not self.asr_pipeline or not audio_path:
            error_details = self.asr_init_error if hasattr(self, 'asr_init_error') else "Unknown Init Error"
            logger.error(f"ASR Model not loaded. Error: {error_details}")
            return "Error: ASR Model failed to load."
        
        logger.info(f"Attempting to transcribe (Wav2Vec2): {audio_path}")
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file does not exist: {audio_path}")
            return "Error: Audio file not found."
            
        try:
            # Pipeline is simpler, handles loading internally
            result = self.asr_pipeline(audio_path)
            text = result.get("text", "").strip().capitalize()
            
            logger.info(f"Transcribed (Wav2Vec2): {text}")
            return text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return "Error transcribing audio."

    def transcribe_audio_qwen(self, audio_path: str) -> str:
        """
        Convert audio file to text using Qwen/Qwen3-ASR-0.6B.
        Size: ~0.6 Billion parameters (approx 1.2GB RAM).
        """
        logger.info(f"Attempting to transcribe (Qwen3-0.6B): {audio_path}")
        
        if not os.path.exists(audio_path):
            return "Error: Audio file not found."

        try:
            # Lazy load Qwen model
            if not hasattr(self, 'qwen_pipeline'):
                logger.info("Loading Qwen3 ASR 0.6B Pipeline (first run)...")
                from transformers import pipeline
                # Qwen3 ASR should work with standard 'automatic-speech-recognition'
                # If not, we might need AutoModelForSpeechSeq2Seq
                self.qwen_pipeline = pipeline("automatic-speech-recognition", model="Qwen/Qwen3-ASR-0.6B", trust_remote_code=True)
            
            result = self.qwen_pipeline(audio_path)
            text = result.get("text", "").strip().capitalize()
            
            logger.info(f"Transcribed (Qwen): {text}")
            return text
        except Exception as e:
            logger.error(f"Qwen Transcription failed: {e}")
            return f"Error: {str(e)}"

    def transcribe_audio_faster_whisper(self, audio_path: str) -> str:
        """
        Convert audio file to text using Faster Whisper (Small).
        Requires: `pip install faster-whisper`
        Size: Small model is ~500 MB VRAM/RAM.
        """
        logger.info(f"Attempting to transcribe (Faster Whisper Small): {audio_path}")
        self._debug(f"STT start (faster-whisper): path={audio_path}, exists={os.path.exists(audio_path)}")
        
        if not os.path.exists(audio_path):
            return "Error: Audio file not found."

        try:
            # Lazy load Faster Whisper
            if not hasattr(self, 'whisper_model'):
                logger.info("Loading Faster Whisper Small (first run)...")
                try:
                    from faster_whisper import WhisperModel
                except ImportError:
                    return "Error: faster-whisper not installed. Run `pip install faster-whisper`."
                
                # Run on CPU with INT8 by default to be safe, or CUDA if available
                # 'small' model
                device = "cuda" if subprocess.call("nvidia-smi", shell=True) == 0 else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                try:
                    self.whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
                except Exception as e:
                    logger.warning(f"Faster Whisper CUDA init failed: {e}. Falling back to CPU.")
                    self.whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
            
            stt_start = time.time()
            segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
            text = " ".join([segment.text for segment in segments]).strip()
            self._debug(f"STT complete (faster-whisper) in {time.time() - stt_start:.2f}s; text_len={len(text)}")
            
            logger.info(f"Transcribed (Faster Whisper): {text}")
            return text
        except Exception as e:
            logger.error(f"Faster Whisper Transcription failed: {e}")
            self._debug(f"STT failure (faster-whisper): {e}")
            return f"Error: {str(e)}"

    def transcribe_audio(self, audio_path: str) -> str:
        """Wrapper to choose STT provider."""
        self._debug("STT dispatch: faster-whisper")
        # Un-comment the one you want to use:
        
        # 1. Wav2Vec2 (Fastest, Smallest, English only)
        # return self.transcribe_audio_wav2vec2(audio_path)
        
        # 2. Qwen3 ASR (Balanced, Multilingual)
        # return self.transcribe_audio_qwen(audio_path)
        
        # 3. Faster Whisper Medium (Most Accurate, Heavier)
        return self.transcribe_audio_faster_whisper(audio_path)

    # ...

    def handle_turn(self, user_audio_path: str, resume_context: str) -> Tuple[str, str, Any, Any]:
        """Process user answer and generate next question."""
        started_at = time.time()
        logger.info("Interview turn started")
        self._debug(
            f"Turn start: model={self.model}, think={self.think}, audio={user_audio_path}, size={self._safe_file_size(user_audio_path)}"
        )

        # 1. Transcribe
        user_text = self.transcribe_audio(user_audio_path)
        self._debug(f"Turn STT text_len={len(user_text) if user_text else 0}")
        if user_text and not user_text.startswith("Error"):
            self._debug("Turn STT succeeded")
        
        # GUARD: If STT failed
        if not user_text or user_text.startswith("Error"):
            error_msg = user_text if user_text else "Error: No speech detected."
            # Return None for audio paths to avoid PermissionError in Gradio
            return error_msg, "I couldn't hear you clearly.", None, None
        
        # 2. Add to history
        self.history.append(HumanMessage(content=user_text))
        self._debug(f"Turn history size after user append={len(self.history)}")
        
        # Keep a reference for scoring after we generate the next question.
        scoring_question = self.current_question
        
        # 4. Generate Next Question
        # Check history length to keep context window manageable
        if len(self.history) > 10:
            # Keep System + Last 8 messages
            self.history = [self.history[0]] + self.history[-8:]
            
        try:
            llm_start = time.time()
            self._debug("Turn LLM invoke started")
            response = self._invoke_llm_with_timeout(self.history, timeout_sec=45, label="Turn")
            next_q = response.content
            self._debug(f"Turn LLM invoke completed in {time.time() - llm_start:.2f}s; resp_len={len(next_q) if next_q else 0}")
            
            if "<think>" in next_q:
                next_q = next_q.split("</think>")[-1].strip()
            next_q = next_q.replace("Interviewer:", "").strip()
            
            self.current_question = next_q
            self.history.append(AIMessage(content=next_q))
            self._debug(f"Turn history size after ai append={len(self.history)}")

            # Start scoring AFTER next question generation to avoid model contention.
            sc_thread = threading.Thread(target=self._score_answer_background, args=(scoring_question, user_text), daemon=True)
            sc_thread.start()
            self._debug("Turn background scoring thread started (post-reply)")
            
            # 5. Generate Audio
            audio_file = os.path.join(self.output_dir, f"q_{int(datetime.now().timestamp())}.wav")
            generated_path = self._generate_tts_with_timeout(next_q, audio_file, timeout_sec=20)
            
            # If generation failed (returned ""), return None
            if not generated_path:
                generated_path = None
            
            elapsed = time.time() - started_at
            logger.info(f"Interview turn completed in {elapsed:.2f}s")
            self._debug(
                f"Turn complete in {elapsed:.2f}s; audio_generated={bool(generated_path)}; audio_size={self._safe_file_size(generated_path) if generated_path else 0}"
            )
            return user_text, next_q, generated_path, self.filler_audio_path
            
        except Exception as e:
            logger.error(f"Turn failed: {e}")
            self._debug(f"Turn failed: {e}\n{traceback.format_exc()}")
            fallback_q = "Let's continue. Can you walk me through one project you are most proud of and your exact contribution?"
            self.current_question = fallback_q
            self.history.append(AIMessage(content=fallback_q))
            self._debug("Turn fallback question returned after failure")
            return user_text, fallback_q, None, None



    def _generate_tts(self, text: str, output_path: str) -> str:
        """
        Generate audio using Pocket TTS (CPU-based, lightweight).
        """
        if not self.tts_model or not self.tts_voice_state:
            logger.error("Pocket TTS not initialized")
            return ""
            
        try:
            tts_start = time.time()
            self._debug(f"TTS generate start; text_len={len(text) if text else 0}, output={output_path}")
            audio = self.tts_model.generate_audio(self.tts_voice_state, text)
            # Audio is a 1D torch tensor containing PCM data.
            self.scipy_wavfile.write(output_path, self.tts_model.sample_rate, audio.numpy())
            
            # Ensure file is flushed to disk for Gradio
            import time
            time.sleep(0.1) 
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"TTS generated: {output_path} ({os.path.getsize(output_path)} bytes)")
                self._debug(f"TTS generate success in {time.time() - tts_start:.2f}s; size={os.path.getsize(output_path)}")
                return output_path
            else:
                logger.error("TTS generated empty file.")
                self._debug("TTS generated empty file")
                return ""
        except Exception as e:
            logger.error(f"Pocket TTS generation failed: {e}")
            self._debug(f"TTS generation exception: {e}")
            return ""



    def start_interview(self, resume_context: str) -> Tuple[str, str, str]:
        """Start a new interview session."""
        started_at = time.time()
        logger.info("Starting interview session")
        self.clear_debug_trace()
        self._debug(f"Start interview requested; model={self.model}, think={self.think}, resume_ctx_len={len(resume_context) if resume_context else 0}")

        self.history = []
        self.scores = []
        
        # 1. Prepare System Message
        system_prompt = f"""
        You are an expert technical interviewer conducting a mock interview.
        
        CANDIDATE CONTEXT:
        {resume_context[:2000]}
        
        INTERVIEW RULES:
        1. **Structure**: For every turn, provided brief feedback on the user's previous answer (if applicable), then IMMEDIATELY ask the NEXT technical question.
        2. **Consistency**: NEVER stop at just feedback. ALWAYS end your response with a clear question.
        3. **Variety**: Cover different topics based on the resume (e.g., Projects, work experience, skills, education, behavioural etc.). Do not stay on one topic for too long and ask questions related to resume mostly.
        4. **Brevity**: Keep your spoken response to 2-4 sentences max. Be conversational but professional.
        5. **Format**: Just speak the response. Do not use prefixes like "Interviewer:" or "*thinking*".
        
        GOAL: Assess the candidate's depth while keeping the conversation moving.
        """
        
        # 2. Invoke LLM
        messages = [SystemMessage(content=system_prompt), HumanMessage(content="Start the interview.")]
        try:
            llm_start = time.time()
            self._debug("Start interview LLM invoke started")
            response = self._invoke_llm_with_timeout(messages, timeout_sec=45, label="Start")
            q_text = response.content
            self._debug(f"Start interview LLM invoke completed in {time.time() - llm_start:.2f}s; resp_len={len(q_text) if q_text else 0}")
            
            # Clean strict leftovers
            if "<think>" in q_text:
                q_text = q_text.split("</think>")[-1].strip()
            q_text = q_text.replace("Interviewer:", "").strip()
            
            # Update State
            self.current_question = q_text
            self.history = [
                SystemMessage(content=system_prompt),
                AIMessage(content=q_text)
            ]
            
            # 3. Generate Audio
            audio_file = os.path.join(self.output_dir, f"q_{int(datetime.now().timestamp())}.wav")
            generated_path = self._generate_tts_with_timeout(q_text, audio_file, timeout_sec=20)
            
            if not generated_path:
                generated_path = None

            elapsed = time.time() - started_at
            logger.info(f"Interview start completed in {elapsed:.2f}s")
            self._debug(
                f"Start interview complete in {elapsed:.2f}s; audio_generated={bool(generated_path)}; audio_size={self._safe_file_size(generated_path) if generated_path else 0}"
            )
            return q_text, generated_path, None
            
        except Exception as e:
            logger.error(f"Start interview failed: {e}")
            self._debug(f"Start interview failed: {e}\n{traceback.format_exc()}")
            fallback_q = "Let's begin. Please introduce yourself in 60-90 seconds and highlight your strongest technical project."
            self.current_question = fallback_q
            self.history = [
                SystemMessage(content=system_prompt),
                AIMessage(content=fallback_q)
            ]
            self._debug("Start fallback question returned after failure")
            return fallback_q, None, None



    def _score_answer_background(self, question, answer):
        """Analyze answer quality using a separate lightweight chain or simple invoke."""
        try:
            self._debug("Background scoring started")
            prompt = f"""
            Analyze this interview response.
            Question: {question}
            Answer: {answer}
            
            Output JSON only:
            {{ "score": 1-10, "feedback": "1 sentence feedback" }}
            """
            # Use raw model or separate instance to not mess up main history
            # For speed, we can just use the same class instance method or a fresh invoke
            # We'll use a one-off messages list
            msgs = [HumanMessage(content=prompt)]
            
            # Ensure JSON mode if possible, or just parse text
            # Setting format='json' (Ollama specific) requires binding or specific method
            # LangChain for Ollama supports .bind(format="json")
            
            scorer_llm = self._build_scorer_llm().bind(format="json")
            resp = self._invoke_llm_with_timeout(msgs, timeout_sec=20, label="Score", llm=scorer_llm)
            
            content = resp.content
            data = json.loads(content)
            
            self.scores.append({
                "question": question,
                "answer": answer,
                "score": data.get("score", 5),
                "feedback": data.get("feedback", "Good effort.")
            })
            self._debug(f"Background scoring completed; score={data.get('score', 5)}")
             
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            self._debug(f"Background scoring failed: {e}")

    def get_latest_feedback(self) -> str:
        """Format scores."""
        if not self.scores:
            return "Waiting for feedback..."
        latest = self.scores[-1]
        return f"**Analysis**\n🎯 Score: {latest['score']}/10\n💡 {latest['feedback']}"

    def end_interview(self) -> str:
        """Generate a final aggregated report of the interview."""
        if not self.scores:
            return "## Interview Report\nNo questions were answered."
        
        total_score = sum(s['score'] for s in self.scores)
        avg_score = total_score / len(self.scores)
        
        report = f"# 📝 Interview Performance Report\n\n"
        report += f"**Overall Score:** {avg_score:.1f}/10\n"
        report += f"**Questions Answered:** {len(self.scores)}\n\n"
        report += "## Question Breakdown\n\n"
        
        for i, item in enumerate(self.scores, 1):
            report += f"### Q{i}: {item['question']}\n"
            report += f"**Your Answer:** *\"{item['answer'][:100]}...\"*\n"
            report += f"**Score:** {item['score']}/10\n"
            report += f"**Feedback:** {item['feedback']}\n"
            report += "---\n"
            
        return report
