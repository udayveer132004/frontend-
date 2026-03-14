"""Main ResumeParser class - orchestrates text extraction and AI parsing."""

import logging
from pathlib import Path
from typing import Optional
from backend.common.models import ResumeData

from .text_extractor import extract_text
from .ai_extractor import extract_resume_data, check_ollama_connection, get_last_model_debug_output
from .ai_extractor_gemini import extract_resume_data_gemini


logger = logging.getLogger(__name__)


class ResumeParser:
    """
    Main resume parser class that coordinates text extraction and AI-powered parsing.
    
    Usage:
        parser = ResumeParser()
        resume_data = parser.parse("path/to/resume.pdf")
    """
    
    def __init__(self, model: str = "qwen3.5:2b", provider: str = "ollama", think: bool = True):
        """
        Initialize the ResumeParser.
        
        Args:
            model: Model name to use for extraction (default: qwen3.5:2b)
            provider: Model provider to use ("ollama" or "gemini")
            think: Whether to enable Ollama thinking mode
        """
        self.model = model
        self.provider = provider.lower().strip()
        self.think = think
        self._debug_info = ""  # Store debug info
        self._check_prerequisites()
    
    def _check_prerequisites(self) -> None:
        """Check if Ollama is running and model is available."""
        if self.provider == "ollama":
            if not check_ollama_connection(self.model):
                logger.warning(
                    f"Ollama model '{self.model}' not found. "
                    "Make sure Ollama is running and the model is pulled."
                )
        elif self.provider == "gemini":
            import os
            if not os.getenv("GEMINI_API_KEY"):
                logger.warning(
                    "GEMINI_API_KEY is not set. Gemini extraction will fail."
                )
    
    def get_debug_info(self) -> str:
        """Return debug info from last parse operation."""
        return self._debug_info
    
    def parse(self, file_path: str) -> tuple[Optional[ResumeData], str]:
        """
        Parse a resume file and extract structured data.
        
        Args:
            file_path: Path to the resume file (PDF or DOCX)
            
        Returns:
            Tuple of (ResumeData object, raw extracted text)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is unsupported or parsing fails
            ConnectionError: If unable to connect to Ollama
        """
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Resume file not found: {file_path}")
        
        logger.info(f"Parsing resume: {path.name}")
        
        # Step 1: Extract text from file
        text = ""
        try:
            text = extract_text(file_path)
            logger.info(f"Extracted {len(text)} characters from {path.name}")
            
            # Save extracted text to file for debugging
            debug_file = Path("extracted_text.txt")
            debug_file.write_text(text, encoding='utf-8')
            logger.info(f"Saved extracted text to {debug_file.absolute()}")
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
        
        # Step 2: Extract structured data using AI
        try:
            if self.provider == "gemini":
                resume_data, raw_response = extract_resume_data_gemini(
                    text, model=self.model, return_debug=True
                )
            else:
                resume_data, raw_response = extract_resume_data(
                    text, model=self.model, return_debug=True, think=self.think
                )
            if resume_data:
                logger.info(f"Successfully parsed resume: {resume_data.name if resume_data else 'Unknown'}")
            else:
                logger.warning("Model returned output, but it could not be parsed into valid ResumeData.")

            last_content, last_thinking = get_last_model_debug_output()
            shown_raw = raw_response or last_content or "N/A"
            shown_thinking = last_thinking or "N/A"
            
            # Store debug info - NO TRUNCATION
            self._debug_info = f"""=== MODEL DEBUG INFO ===

Provider: {self.provider}
Model: {self.model}

=== INPUT (Resume Text - {len(text)} chars) ===
{text}

=== RAW MODEL OUTPUT ({len(shown_raw) if shown_raw != 'N/A' else 0} chars) ===
{shown_raw}

=== MODEL THINKING ({len(shown_thinking) if shown_thinking != 'N/A' else 0} chars) ===
{shown_thinking}
"""
            return resume_data, text
        except Exception as e:
            logger.error(f"AI extraction failed: {e}")
            last_content, last_thinking = get_last_model_debug_output()
            shown_raw = last_content or "N/A"
            shown_thinking = last_thinking or "N/A"

            self._debug_info = f"""=== MODEL DEBUG INFO ===

Provider: {self.provider}
Model: {self.model}
Error: {str(e)}

=== INPUT (Resume Text - {len(text)} chars) ===
{text}

=== RAW MODEL OUTPUT ({len(shown_raw) if shown_raw != 'N/A' else 0} chars) ===
{shown_raw}

=== MODEL THINKING ({len(shown_thinking) if shown_thinking != 'N/A' else 0} chars) ===
{shown_thinking}
"""
            # Return text even if AI extraction fails
            return None, text
    
    def parse_text(self, text: str) -> Optional[ResumeData]:
        """
        Parse resume text directly (without file extraction).
        
        Useful for testing or when text is already extracted.
        
        Args:
            text: Resume text content
            
        Returns:
            ResumeData object with extracted information
        """
        if self.provider == "gemini":
            result, _ = extract_resume_data_gemini(text, model=self.model, return_debug=True)
            return result
        result, _ = extract_resume_data(text, model=self.model, return_debug=True, think=self.think)
        return result
