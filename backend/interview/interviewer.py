"""
Robust Interview Manager for mock interviews.

Design goals:
- Never block UI callbacks indefinitely.
- Always return an interviewer text response, even on STT/LLM/TTS failures.
- Keep detailed internal debug trace for diagnostics.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests

logger = logging.getLogger(__name__)
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "-1"))
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")


class InterviewManager:
    def __init__(self, output_dir: str = "temp_audio", model: str = "qwen3.5:2b", think: bool = True):
        self.history: List[Dict[str, str]] = []
        self.scores: List[Dict[str, Any]] = []
        self.current_question = ""
        self.output_dir = output_dir
        self.model = model
        self.think = think
        self.debug_events: List[str] = []
        self._scores_lock = threading.Lock()

        self._whisper_model = None
        self._tts_model = None
        self._tts_voice_state = None
        self._tts_wav_writer = None

        os.makedirs(self.output_dir, exist_ok=True)
        self._debug(f"InterviewManager ready with model={self.model}, think={self.think}")

        # Optional TTS init. If unavailable, text mode still works.
        self._init_tts_optional()

    def _debug(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] {message}"
        self.debug_events.append(line)
        if len(self.debug_events) > 600:
            self.debug_events = self.debug_events[-600:]
        logger.info(f"[INTERVIEW_DEBUG] {message}")

    def clear_debug_trace(self) -> None:
        self.debug_events = []

    def get_debug_trace(self) -> str:
        return "\n".join(self.debug_events) if self.debug_events else "No interview debug events yet."

    def configure_llm(self, model: str, think: bool = True) -> None:
        model = (model or "").strip() or self.model
        if model == self.model and think == self.think:
            return
        self.model = model
        self.think = think
        self._debug(f"LLM configured: model={self.model}, think={self.think}")

    def _invoke_with_timeout(self, fn, timeout_sec: int, label: str):
        result = {"value": None, "error": None}

        def _worker():
            try:
                result["value"] = fn()
            except Exception as e:
                result["error"] = e

        self._debug(f"{label}: worker start; timeout={timeout_sec}s")
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout_sec)

        if t.is_alive():
            self._debug(f"{label}: timeout reached")
            raise TimeoutError(f"{label} timed out after {timeout_sec}s")
        if result["error"] is not None:
            self._debug(f"{label}: error={result['error']}")
            raise result["error"]

        self._debug(f"{label}: worker finished")
        return result["value"]

    def _ollama_chat(
        self,
        messages: List[Dict[str, str]],
        timeout_sec: int = 60,
        num_predict: int = 160,
        think_override: bool | None = None,
        temperature: float = 0.2,
    ) -> str:
        use_think = self.think if think_override is None else think_override
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": use_think,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_gpu": OLLAMA_NUM_GPU,
            },
        }

        def _request():
            resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")

        content = self._invoke_with_timeout(_request, timeout_sec=timeout_sec + 2, label="OllamaChat")
        return self._clean_response(content)

    @staticmethod
    def _clean_response(text: str) -> str:
        out = (text or "").strip()
        if "</think>" in out:
            out = out.split("</think>")[-1].strip()
        out = out.replace("Interviewer:", "").strip()
        return out

    def _init_tts_optional(self) -> None:
        try:
            from pocket_tts import TTSModel
            import scipy.io.wavfile

            self._tts_model = TTSModel.load_model()
            self._tts_voice_state = self._tts_model.get_state_for_audio_prompt("eponine")
            self._tts_wav_writer = scipy.io.wavfile
            self._debug("Pocket TTS initialized")
        except Exception as e:
            self._debug(f"Pocket TTS unavailable: {e}")

    def _generate_tts(self, text: str, output_path: str) -> str:
        if not self._tts_model or not self._tts_voice_state:
            return ""
        audio = self._tts_model.generate_audio(self._tts_voice_state, text)
        self._tts_wav_writer.write(output_path, self._tts_model.sample_rate, audio.numpy())
        return output_path if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else ""

    def _generate_tts_with_timeout(self, text: str, output_path: str, timeout_sec: int = 25) -> str:
        try:
            return self._invoke_with_timeout(
                lambda: self._generate_tts(text, output_path),
                timeout_sec=timeout_sec,
                label="TTS",
            ) or ""
        except Exception as e:
            self._debug(f"TTS skipped: {e}")
            return ""

    def _ensure_whisper_model(self):
        if self._whisper_model is not None:
            return
        from faster_whisper import WhisperModel

        self._whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        self._debug("Faster-Whisper tiny initialized")

    def _transcribe_audio_impl(self, audio_path: str) -> str:
        if not audio_path or not os.path.exists(audio_path):
            return ""
        self._ensure_whisper_model()
        segments, _ = self._whisper_model.transcribe(audio_path, beam_size=1, vad_filter=True)
        text = " ".join(seg.text for seg in segments).strip()
        return text

    def transcribe_audio(self, audio_path: str) -> str:
        self._debug(f"STT start: file={audio_path}")
        try:
            text = self._invoke_with_timeout(
                lambda: self._transcribe_audio_impl(audio_path),
                timeout_sec=40,
                label="STT",
            )
            self._debug(f"STT done: text_len={len(text) if text else 0}")
            return text
        except Exception as e:
            self._debug(f"STT failed: {e}")
            return ""

    def start_interview(self, resume_context: str) -> Tuple[str, str, str]:
        self.clear_debug_trace()
        self._debug(f"Start requested: model={self.model}, think={self.think}, resume_ctx_len={len(resume_context or '')}")

        self.history = []
        self.scores = []

        system_prompt = (
            "You are a technical interviewer. Keep replies concise (2-3 sentences) and always end with exactly one question. "
            "Focus on resume-relevant technical depth."
        )
        user_prompt = (
            f"Candidate context:\n{(resume_context or '')[:1200]}\n\n"
            "Start the interview with a friendly opener and one technical question."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            q_text = self._ollama_chat(messages, timeout_sec=70, num_predict=120)
            if not q_text:
                raise ValueError("Empty response from interviewer model")
            self._debug(f"Start LLM response len={len(q_text)}")
        except Exception as e:
            self._debug(f"Start LLM failed: {e}")
            q_text = "Let's begin. Please introduce yourself briefly and describe one technical project you built end-to-end."
            self._debug("Start fallback question used")

        self.current_question = q_text
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "ai", "content": q_text},
        ]

        audio_file = os.path.join(self.output_dir, f"q_{int(time.time())}.wav")
        audio_path = self._generate_tts_with_timeout(q_text, audio_file, timeout_sec=25)
        if not audio_path:
            audio_path = None
        self._debug(f"Start complete: audio_generated={bool(audio_path)}")
        return q_text, audio_path, None

    def _score_answer_heuristic(self, answer: str) -> Dict[str, Any]:
        words = [w for w in (answer or "").split() if w.strip()]
        word_count = len(words)
        score = 4
        if word_count >= 20:
            score += 2
        if word_count >= 40:
            score += 2
        if any(k in answer.lower() for k in ["because", "tradeoff", "optimized", "latency", "scalable", "cache"]):
            score += 2
        score = max(1, min(10, score))
        feedback = "Good attempt. Add more concrete metrics and trade-offs." if score < 7 else "Good depth. Keep explaining decisions and outcomes with metrics."
        return {"score": score, "feedback": feedback}

    def _score_answer_with_llm(self, question: str, answer: str) -> Dict[str, Any]:
        """Evaluate candidate answer via LLM. Returns {'score': int, 'feedback': str}."""
        eval_messages = [
            {
                "role": "system",
                "content": (
                    "You are an interview evaluator. Return ONLY valid JSON with keys: "
                    "score (integer 1-10) and feedback (one concise sentence)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Candidate Answer: {answer}\n"
                    "Evaluate technical depth, clarity, correctness, and practical reasoning."
                ),
            },
        ]

        raw = self._ollama_chat(
            eval_messages,
            timeout_sec=45,
            num_predict=100,
            think_override=False,
            temperature=0.0,
        )
        data = json.loads(raw)
        score = int(data.get("score", 5))
        score = max(1, min(10, score))
        feedback = str(data.get("feedback", "Good effort."))[:300]
        return {"score": score, "feedback": feedback}

    def _evaluate_answer_background(self, question: str, answer: str) -> None:
        """Run LLM answer evaluation in the background so UI stays responsive."""
        self._debug("Background LLM evaluation started")
        try:
            result = self._score_answer_with_llm(question, answer)
            self._debug(f"Background LLM evaluation completed; score={result['score']}")
        except Exception as e:
            self._debug(f"Background LLM evaluation failed: {e}; using heuristic fallback")
            result = self._score_answer_heuristic(answer)

        with self._scores_lock:
            self.scores.append(
                {
                    "question": question,
                    "answer": answer,
                    "score": result["score"],
                    "feedback": result["feedback"],
                }
            )

    def handle_turn(self, user_audio_path: str, resume_context: str) -> Tuple[str, str, Any, Any]:
        self._debug(f"Turn requested: model={self.model}, think={self.think}")

        user_text = self.transcribe_audio(user_audio_path)
        if not user_text:
            user_text = "I could not transcribe your audio clearly."
            self._debug("Turn using transcription fallback text")

        asked_question = self.current_question
        self.history.append({"role": "user", "content": user_text})
        self._debug(f"Turn history size after user={len(self.history)}")

        recent = self.history[-6:]
        messages = [
            {
                "role": "system",
                "content": "You are a technical interviewer. Give brief feedback in one sentence, then ask exactly one next technical question.",
            },
            {
                "role": "user",
                "content": (
                    f"Resume context:\n{(resume_context or '')[:1000]}\n\n"
                    f"Current question: {self.current_question}\n"
                    f"Candidate answer: {user_text}\n\n"
                    "Now give one-sentence feedback and ask the next question."
                ),
            },
        ]

        try:
            next_q = self._ollama_chat(messages, timeout_sec=70, num_predict=140)
            if not next_q:
                raise ValueError("Empty response from interviewer model")
            self._debug(f"Turn LLM response len={len(next_q)}")
        except Exception as e:
            self._debug(f"Turn LLM failed: {e}")
            next_q = "Thanks. Next question: describe a production issue you debugged, how you isolated root cause, and what fix you shipped."
            self._debug("Turn fallback question used")

        self.current_question = next_q
        self.history.append({"role": "ai", "content": next_q})

        # Evaluate the PREVIOUS answer asynchronously using LLM.
        # This keeps UI responsive and runs while the user is preparing the next response.
        threading.Thread(
            target=self._evaluate_answer_background,
            args=(asked_question or "Interview answer", user_text),
            daemon=True,
        ).start()
        self._debug("Background LLM evaluation thread launched")

        audio_file = os.path.join(self.output_dir, f"q_{int(time.time())}.wav")
        audio_path = self._generate_tts_with_timeout(next_q, audio_file, timeout_sec=25)
        if not audio_path:
            audio_path = None

        self._debug(f"Turn complete: audio_generated={bool(audio_path)}")
        return user_text, next_q, audio_path, None

    def get_latest_feedback(self) -> str:
        with self._scores_lock:
            if not self.scores:
                return "Evaluating answer with LLM..."
            latest = self.scores[-1]
        return f"**Analysis**\n🎯 Score: {latest['score']}/10\n💡 {latest['feedback']}"

    def end_interview(self) -> str:
        with self._scores_lock:
            snapshot = list(self.scores)

        if not snapshot:
            return "## Interview Report\nNo questions were answered."

        total_score = sum(s["score"] for s in snapshot)
        avg = total_score / len(snapshot)
        report = "# Interview Performance Report\n\n"
        report += f"**Overall Score:** {avg:.1f}/10\n"
        report += f"**Questions Answered:** {len(snapshot)}\n\n"
        for i, item in enumerate(snapshot, 1):
            report += f"### Q{i}\n"
            report += f"**Answer excerpt:** {item['answer'][:140]}\n\n"
            report += f"**Score:** {item['score']}/10\n"
            report += f"**Feedback:** {item['feedback']}\n\n"
        return report
