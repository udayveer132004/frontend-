"""ATS Scoring Module - Calculate resume compatibility scores using LLM with fallback."""

import json
import logging
import os
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from google import genai
from google.genai import types
from backend.common.models import ResumeData

logger = logging.getLogger(__name__)
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "-1"))


@dataclass
class ATSResult:
    """Result of ATS scoring."""
    score: int
    breakdown: Dict[str, float]
    missing_keywords: List[str]
    formatting_issues: List[str]
    suggestions: List[str]
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ATSScorer:
    """
    ATS Scorer that evaluates a resume against a job description.
    Uses LLM (Ollama/Gemini) primarily, with a heuristic fallback.
    """
    
    def __init__(self, model: str = "qwen3.5:2b", provider: str = "ollama", think: bool = True):
        self.model = model
        self.provider = provider
        self.think = think
    
    def calculate_score(self, resume_data: ResumeData, job_description: str) -> ATSResult:
        """
        Calculate ATS score using LLM, fallback to heuristic if needed.
        """
        try:
            logger.info(f"Attempting LLM scoring with {self.provider}/{self.model}")
            return self._score_with_llm(resume_data, job_description)
        except Exception as e:
            logger.error(f"LLM Scoring failed: {e}. Falling back to heuristic.")
            return self._score_heuristic(resume_data, job_description)

    def _score_with_llm(self, resume_data: ResumeData, job_description: str) -> ATSResult:
        """
        Score using LLM (Ollama or Gemini).
        """
        prompt = f"""
        You are an expert ATS (Applicant Tracking System) Scorer.
        Analyze the following CANDIDATE RESUME against the JOB DESCRIPTION.
        Your nature: Be a little brutal (no too much) in nature.
        
        JOB DESCRIPTION:
        {job_description}
        
        CANDIDATE RESUME:
        {resume_data.model_dump_json(exclude_none=True)}
        
        TASK:
        1. Calculate a compatibility score (0-100). Make sure to give the final score.
        2. Provide a breakdown of the score (Keywords, Skills, Formatting, Education, Experience).
        3. Identify Matching and Missing Critical Skills.
        4. Provide specific improvement suggestions.
        
        OUTPUT FORMAT (Valid JSON only):
        {{
            "score": x,
            "breakdown": {{
                "Keywords": x,
                "Skills": x,
                "Formatting": x,
                "Education": x,
                "Experience": x
            }},
            "missing_keywords": ["MissingSkill1", "MissingSkill2"],
            "formatting_issues": ["Issue1", "Issue2"],
            "suggestions": ["Suggestion 1", "Suggestion 2"],
            "reasoning": "Brief explanation of the score..."
        }}
        
        IMPORTANT: 
        - The 'breakdown' values should roughly sum up to the total 'score'.
        - 'Keywords' max 40, 'Skills' max 30, 'Formatting' max 10, 'Education' max 10, 'Experience' max 10.
        - Return ONLY valid JSON.
        """
        
        response_json_str = ""
        
        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            response_json_str = response.text
            
        else: # ollama
            payload = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'stream': False,
                'think': self.think,
                'options': {'temperature': 0.2, 'num_predict': 4096, 'num_gpu': OLLAMA_NUM_GPU},
                'format': 'json'
            }
            response = requests.post('http://localhost:11434/api/chat', json=payload)
            response.raise_for_status()
            response_json_str = response.json().get("message", {}).get("content", "")

        # Parse JSON
        if "```json" in response_json_str:
            response_json_str = response_json_str.split("```json")[1].split("```")[0]
        elif "```" in response_json_str:
             response_json_str = response_json_str.split("```")[1].split("```")[0]
             
        data = json.loads(response_json_str.strip())
        
        breakdown = data.get("breakdown", {})
        score = data.get("score", 0)
        
        if not breakdown:
             breakdown = {
                 "Keywords": score * 0.4,
                 "Skills": score * 0.3, 
                 "Formatting": 10,
                 "Education": 10,
                 "Experience": score * 0.1
             }

        return ATSResult(
            score=int(score),
            breakdown=breakdown,
            missing_keywords=data.get("missing_keywords", []),
            formatting_issues=data.get("formatting_issues", []),
            suggestions=data.get("suggestions", []),
            reasoning=data.get("reasoning", "")
        )

    def _score_heuristic(self, resume_data: ResumeData, job_description: str) -> ATSResult:
        """
        Original logic: Calculate ATS score based on weighted criteria (Fallback).
        """
        # (Heuristic logic preserved below)
        jd_keywords = self._extract_keywords_from_jd_heuristic(job_description)
        
        def safe_join(val):
            if isinstance(val, list):
                return " ".join([str(v) for v in val if v])
            return str(val) if val else ""
            
        resume_text_blob = " ".join([
            safe_join(resume_data.summary),
            safe_join(resume_data.experience),
            safe_join(resume_data.education),
            safe_join(resume_data.achievements),
            safe_join(resume_data.skills),
            safe_join(resume_data.projects)
        ]).lower()
        
        # A. Keyword Match (40 points)
        matched_keywords = []
        missing_keywords = []
        for kw in jd_keywords:
            if kw.lower() in resume_text_blob:
                matched_keywords.append(kw)
            else:
                missing_keywords.append(kw)
        
        keyword_score = 40
        if jd_keywords:
            keyword_score = (len(matched_keywords) / len(jd_keywords)) * 40

        # B. Skills Match (30 points)
        resume_skills_lower = {s.lower() for s in resume_data.skills}
        skill_matches = 0
        relevant_jd_skills = [k for k in jd_keywords if k.lower() not in ["communication", "teamwork"]]
        if not relevant_jd_skills: relevant_jd_skills = jd_keywords

        for k in relevant_jd_skills:
            if k.lower() in resume_skills_lower:
                skill_matches += 1
        
        skills_score = 30
        if relevant_jd_skills:
            skills_score = (skill_matches / len(relevant_jd_skills)) * 30

        # C. Formatting (10)
        formatting_score = 10
        formatting_issues = []
        if not resume_data.email: formatting_issues.append("Missing Email"); formatting_score -= 2
        if not resume_data.phone: formatting_issues.append("Missing Phone"); formatting_score -= 2
        if not resume_data.skills: formatting_issues.append("Skills missing"); formatting_score -= 3
        formatting_score = max(0, formatting_score)

        # D. Education (10)
        education_score = 10 if resume_data.education else 0
        
        # E. Experience (10)
        experience_score = 10 if safe_join(resume_data.experience) else 0
        
        total_score = int(keyword_score + skills_score + formatting_score + education_score + experience_score)
        total_score = min(100, max(0, total_score))
        
        suggestions = []
        if total_score < 70: suggestions.append("Include more keywords.")
        if missing_keywords: suggestions.append(f"Add keywords: {', '.join(missing_keywords[:3])}")
        
        return ATSResult(
            score=total_score,
            breakdown={
                "Keywords": round(keyword_score, 1),
                "Skills": round(skills_score, 1),
                "Formatting": formatting_score,
                "Education": education_score,
                "Experience": experience_score
            },
            missing_keywords=missing_keywords,
            formatting_issues=formatting_issues,
            suggestions=suggestions,
            reasoning="Fallback heuristic scoring used."
        )

    def _extract_keywords_from_jd_heuristic(self, jd: str) -> List[str]:
        """Simple keyword extraction for fallback."""
        try:
             if self.provider == "gemini": return [] 
             
             prompt = f"Extract 5 technical keywords from: {jd[:500]}. Return JSON list."
             resp = requests.post('http://localhost:11434/api/chat', json={
                 'model': 'qwen3.5:2b', 
                 'messages': [{'role': 'user', 'content': prompt}], 
                 'stream': False,
                 'think': self.think,
                 'options': {'num_predict': 100, 'num_gpu': OLLAMA_NUM_GPU}
             })
             if resp.status_code == 200:
                 content = resp.json()['message']['content']
                 import re
                 match = re.search(r'\[.*\]', content)
                 if match: return json.loads(match.group(0))
             return []
        except:
             return []

def calculate_ats_score_gemini(resume_data: ResumeData, job_description: str) -> ATSResult:
    """Convenience function for Gemini-based scoring (backward compatibility)."""
    scorer = ATSScorer(provider="gemini", model="gemini-2.5-flash")
    return scorer.calculate_score(resume_data, job_description)
