"""AI-powered structured data extraction from resume text using Ollama."""

import json
import logging
import os
import re
from typing import Optional, Generator
import requests
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.common.models import ResumeData

logger = logging.getLogger(__name__)
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "-1"))

# Global variable to store last thinking response for debug
_last_thinking = ""
_last_content = ""


def query_ollama(prompt: str, model: str = 'qwen3.5:2b', stream: bool = False, think: bool = True, json_mode: bool = True) -> tuple[str, str]:
    """Send prompt to local Ollama and return response text using /api/chat."""
    global _last_content, _last_thinking
    try:
        payload = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'stream': stream,
            'think': think,  # Enable qwen thinking mode
            'options': {
                'temperature': 0.1,  # Low temperature for consistent output
                'num_predict': 8192,  # Allow much longer output for complete JSON
                'num_gpu': OLLAMA_NUM_GPU,
            }
        }
        
        if json_mode:
            payload['format'] = 'json'

        response = requests.post(
            'http://localhost:11434/api/chat',
            json=payload,
            timeout=600,
            stream=stream,
        )
        response.raise_for_status()

        if not stream:
            result = response.json()
            logger.debug(f"Full Ollama response keys: {result.keys()}")
            message = result.get('message', {})
            content = message.get('content', '')
            thinking = message.get('thinking', '')

            logger.info(f"Content field: {len(content)} chars")
            logger.info(f"Thinking field: {len(thinking)} chars")

            if not content and thinking:
                logger.info("Content empty, using thinking field as content")
                content = thinking

            _last_content = content
            _last_thinking = thinking

            logger.info(f"Response preview: {content[:500] if content else 'EMPTY'}")
            return content, thinking

        # Streamed response
        content = ""
        thinking = ""
        last_content_len = 0
        last_thinking_len = 0

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            message = chunk.get('message', {})
            if 'content' in message and message['content']:
                content += message['content']
            if 'thinking' in message and message['thinking']:
                thinking += message['thinking']

            if len(thinking) > last_thinking_len:
                delta = thinking[last_thinking_len:]
                logger.info(f"[THINKING] {delta}")
                last_thinking_len = len(thinking)

            if len(content) > last_content_len:
                delta = content[last_content_len:]
                logger.info(f"[OUTPUT] {delta}")
                last_content_len = len(content)

            if chunk.get('done') is True:
                break

        _last_content = content
        _last_thinking = thinking

        return content, thinking
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")


def stream_ollama_response(prompt: str, model: str = "qwen3.5:2b", think: bool = True) -> Generator[tuple[str, str], None, None]:
    """Stream Ollama response and yield (content_so_far, thinking_so_far)."""
    payload = {
        'model': model,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'stream': True,
        'think': think,
        'format': 'json',
        'options': {
            'temperature': 0.1,
            'num_predict': 8192,
            'num_gpu': OLLAMA_NUM_GPU,
        }
    }

    try:
        response = requests.post(
            'http://localhost:11434/api/chat',
            json=payload,
            timeout=600,
            stream=True
        )
        response.raise_for_status()

        content = ""
        thinking = ""

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            message = chunk.get('message', {})
            if 'content' in message and message['content']:
                content += message['content']
            if 'thinking' in message and message['thinking']:
                thinking += message['thinking']

            yield content, thinking

            if chunk.get('done') is True:
                global _last_content, _last_thinking
                _last_content = content
                _last_thinking = thinking
                break
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")


def get_last_model_debug_output() -> tuple[str, str]:
    """Return latest (content, thinking) captured from Ollama responses."""
    return _last_content, _last_thinking


def _build_langchain_chain(model: str = "qwen3.5:2b"):
    """Build a LangChain pipeline for prompt -> LLM -> string output."""
    prompt = ChatPromptTemplate.from_messages([
        ("human", "{prompt}")
    ])

    llm = ChatOllama(
        model=model,
        temperature=0.1,
        format="json",
        options={"num_predict": 8192, "num_gpu": OLLAMA_NUM_GPU},
        model_kwargs={"think": True}
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def stream_langchain_response(prompt_text: str, model: str = "qwen3.5:2b") -> Generator[str, None, None]:
    """Stream model output via LangChain and yield accumulated text."""
    chain = _build_langchain_chain(model=model)
    content = ""

    for chunk in chain.stream({"prompt": prompt_text}):
        if not chunk:
            continue
        content += chunk
        yield content


def get_extraction_prompt(resume_text: str) -> str:
    """Generate extraction prompt with resume text and JSON schema."""
    return f"""You are a resume parser. Extract information from the resume and return valid JSON.

CRITICAL RULES:
1. PROJECTS vs EXPERIENCE:
   - "projects" = Personal projects, side projects, academic projects (things the person BUILT)
   - "experience" = PAID employment at companies (jobs, internships with company names)
   - If someone "Built an app" or "Developed a system" WITHOUT being employed, it's a PROJECT
   - If there is NO "Work Experience" or "Employment" section, experience should be EMPTY []

2. LINKEDIN/GITHUB:
   - Only extract actual usernames or URLs, not the word "LinkedIn" or "GitHub"
   - If you only see the label without a link/username, leave EMPTY ""

3. SKILLS:
   - Only actual technologies: Python, JavaScript, SQL, Docker, etc.
   - NO project names or descriptions

4. FORMAT:
   - Return ONLY valid JSON, no markdown, no explanations
   - Use "" for missing string fields
   - Use [] for missing array fields
5. SUGGESTED ROLES:
- Infer 1-2 standard job titles the candidate is best suited for (e.g. "Software Engineer", "Data Scientist")
- Based rigidly on their skills and experience

RESUME TEXT:
{resume_text}

Return valid JSON only:"""


def extract_resume_data(
    text: str,
    model: str = "qwen3.5:2b",
    timeout: int = 300,
    use_structured_output: bool = True,
    return_debug: bool = False,
    stream: bool = False,
    think: bool = True,
) -> Optional[ResumeData]:
    """
    Extract structured resume data from text using Ollama AI.
    
    Args:
        text: The resume text content
        model: Ollama model to use (default: qwen3.5:2b)
        timeout: Request timeout in seconds
        use_structured_output: Kept for compatibility but not used
        return_debug: If True, return tuple of (ResumeData, raw_response)
    
    Returns:
        ResumeData object with extracted information, or tuple if return_debug=True
    """
    if not text or len(text.strip()) < 50:
        raise ValueError("Resume text is too short or empty")
    
    prompt = get_extraction_prompt(text)
    raw_response = ""
    
    try:
        logger.info(f"Calling Ollama with model: {model}")
        content, thinking = query_ollama(prompt, model=model, stream=stream, think=think)
        raw_response = content  # For backward compatibility
        
        logger.info(f"Received response, content: {len(content)} chars, thinking: {len(thinking)} chars")
        logger.debug(f"Response preview: {content[:200] if content else 'EMPTY'}")
        
        # Store thinking for debug access
        global _last_thinking
        _last_thinking = thinking

        try:
            resume_data = parse_resume_data_from_response(text, content)
        except Exception as parse_err:
            if thinking:
                logger.warning(
                    f"Primary content JSON parse failed ({parse_err}); retrying parse from thinking text."
                )
                resume_data = parse_resume_data_from_response(text, thinking)
                raw_response = f"[content]\n{content}\n\n[thinking]\n{thinking}"
            else:
                raise

        if return_debug:
            return resume_data, raw_response
        return resume_data
            
    except ConnectionError:
        if return_debug:
            raise  # Still raise but caller has raw_response
        raise
    except Exception as e:
        logger.error(f"Unexpected error during extraction: {e}")
        if return_debug:
            # Return raw model output so callers can show it in debug UI.
            return None, raw_response
        raise


def _fix_project_experience_swap(data_dict: dict) -> dict:
    """
    Fix common LLM mistake of putting projects in experience field.
    Detects project-like entries and swaps them to the correct field.
    """
    experience = data_dict.get('experience', [])
    projects = data_dict.get('projects', [])
    
    if not isinstance(experience, list):
        experience = []
    if not isinstance(projects, list):
        projects = []
    
    # If projects is empty but experience has items, check if they're actually projects
    if len(projects) == 0 and len(experience) > 0:
        project_indicators = [
            'built a', 'developed a', 'created a', 'designed a', 'implemented a',
            'ai-powered', 'opportunity aggregator', 'walmart sales', 'leetcard',
            'data analysis', 'web application', 'full-stack'
        ]
        job_indicators = [
            'intern at', 'engineer at', 'developer at', 'analyst at',
            'worked at', 'employed at', 'company', 'corporation', 'inc.', 'ltd.'
        ]
        
        likely_projects = []
        likely_jobs = []
        
        for exp in experience:
            if not exp:
                continue
            exp_lower = str(exp).lower()
            
            # Check for project indicators
            has_project_words = any(ind in exp_lower for ind in project_indicators)
            has_job_words = any(ind in exp_lower for ind in job_indicators)
            
            if has_project_words and not has_job_words:
                likely_projects.append(exp)
            elif has_job_words:
                likely_jobs.append(exp)
            else:
                # Default: if no company name visible, treat as project
                likely_projects.append(exp)
        
        if likely_projects:
            logger.warning(f"Auto-swapping {len(likely_projects)} items from experience to projects")
            data_dict['projects'] = likely_projects
            data_dict['experience'] = likely_jobs
    
    return data_dict


def _clean_all_fields(data_dict: dict) -> dict:
    """Clean all fields: remove empty strings, fix list syntax, etc."""
    
    list_fields = ['projects', 'experience', 'education', 'certifications', 'achievements',
                  'publications', 'volunteer', 'awards', 'interests', 'skills', 
                  'languages', 'key_strengths']
    
    for field in list_fields:
        if field in data_dict:
            value = data_dict[field]
            
            # Convert None or empty string to empty list
            if value is None or value == "" or value == []:
                data_dict[field] = []
                continue
            
            if isinstance(value, list):
                cleaned = []
                for item in value:
                    if item is None:
                        continue
                    if isinstance(item, dict):
                        # Convert dict to string
                        cleaned.append(' - '.join(str(v) for v in item.values() if v))
                    else:
                        item_str = str(item).strip()
                        
                        # Clean Python list syntax: ['a', 'b', 'c']
                        item_str = _clean_list_syntax(item_str)
                        
                        # Skip empty strings
                        if item_str and item_str not in ['', "''", '""', '[]', "['']", '[""]']:
                            cleaned.append(item_str)
                
                data_dict[field] = cleaned
    
    # Clean string fields - remove the word "LinkedIn" or "GitHub" if it's just the label
    for field in ['linkedin', 'github']:
        if field in data_dict:
            value = data_dict[field]
            if value and str(value).lower() in ['linkedin', 'github', 'none', 'null']:
                data_dict[field] = ""
    
    return data_dict


def _clean_list_syntax(text: str) -> str:
    """Remove Python list syntax from strings like ['a', 'b'] -> a, b"""
    if not text:
        return text
    
    # Pattern: ['item1', 'item2', ...]
    pattern = r"\[(['\"])(.*?)\1(?:,\s*\1(.*?)\1)*\]"
    
    # Simple approach: remove brackets and quotes, join with commas
    cleaned = text
    
    # Remove patterns like ['...']
    cleaned = re.sub(r"\['", "", cleaned)
    cleaned = re.sub(r"'\]", "", cleaned)
    cleaned = re.sub(r'\["', "", cleaned)
    cleaned = re.sub(r'"\]', "", cleaned)
    
    # Clean up remaining quotes and commas
    cleaned = re.sub(r"',\s*'", ", ", cleaned)
    cleaned = re.sub(r'",\s*"', ", ", cleaned)
    
    # Remove standalone quotes at start/end
    cleaned = cleaned.strip("'\"")
    
    return cleaned.strip()


def _heuristic_extract_from_text(text: str) -> dict:
    """Heuristically extract key sections from raw resume text."""
    if not text:
        return {}

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    # Contact info
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"\+?\d[\d\s\-()]{8,}\d", text)
    linkedin_match = re.search(r"https?://(?:www\.)?linkedin\.com/\S+", text)
    github_match = re.search(r"https?://(?:www\.)?github\.com/\S+", text)
    portfolio_match = re.search(r"https?://\S+", text)

    # Section parsing
    section_headers = {
        "projects": "projects",
        "certifications": "certifications",
        "technical skills": "skills",
        "skills": "skills",
        "education": "education",
        "experience": "experience",
        "work experience": "experience",
    }

    current_section = None
    sections: dict[str, list[str]] = {
        "projects": [],
        "certifications": [],
        "skills": [],
        "education": [],
        "experience": [],
    }

    for ln in lines:
        header_key = ln.strip().lower()
        if header_key in section_headers:
            current_section = section_headers[header_key]
            continue
        if current_section:
            sections[current_section].append(ln)

    # Parse skills
    skills: list[str] = []
    for ln in sections.get("skills", []):
        if ":" in ln:
            _, rest = ln.split(":", 1)
            skills += [s.strip() for s in rest.split(",") if s.strip()]
        else:
            skills += [s.strip() for s in ln.split(",") if s.strip()]

    # Parse projects
    projects: list[str] = []
    project_lines = sections.get("projects", [])
    current_project = ""
    for ln in project_lines:
        is_bullet = ln.startswith("•") or ln.startswith("-") or ln.startswith("●")
        is_title_line = (" | " in ln) and not is_bullet
        if is_title_line:
            if current_project:
                projects.append(current_project.strip())
            current_project = ln
            continue
        if is_bullet:
            bullet = ln.lstrip("•-●").strip()
            if current_project:
                current_project += f" • {bullet}"
            else:
                current_project = bullet
        else:
            # Continuation line
            if current_project:
                current_project += f" {ln}"

    if current_project:
        projects.append(current_project.strip())

    # Parse certifications
    certifications: list[str] = []
    cert_lines = sections.get("certifications", [])
    current_cert = ""
    for ln in cert_lines:
        is_bullet = ln.startswith("•") or ln.startswith("-") or ln.startswith("●")
        has_year = bool(re.search(r"\b20\d{2}\b", ln))
        if not is_bullet and has_year:
            if current_cert:
                certifications.append(current_cert.strip())
            current_cert = ln
            continue
        if is_bullet:
            bullet = ln.lstrip("•-●").strip()
            if current_cert:
                current_cert += f" • {bullet}"
            else:
                current_cert = bullet
        else:
            if current_cert:
                current_cert += f" {ln}"
            else:
                current_cert = ln

    if current_cert:
        certifications.append(current_cert.strip())

    # Parse education
    education: list[str] = []
    edu_lines = sections.get("education", [])
    i = 0
    while i < len(edu_lines):
        line = edu_lines[i]
        next_line = edu_lines[i + 1] if i + 1 < len(edu_lines) else ""
        if next_line:
            education.append(f"{line} | {next_line}")
            i += 2
        else:
            education.append(line)
            i += 1

    return {
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
        "linkedin": linkedin_match.group(0) if linkedin_match else "",
        "github": github_match.group(0) if github_match else "",
        "portfolio": portfolio_match.group(0) if portfolio_match else "",
        "skills": skills,
        "projects": projects,
        "certifications": certifications,
        "education": education,
    }


def _merge_with_heuristics(data_dict: dict, heuristic: dict) -> dict:
    """Merge heuristic extraction into model output when missing or incomplete."""
    if not heuristic:
        return data_dict

    # String fields
    for field in ["email", "phone", "linkedin", "github", "portfolio"]:
        if not data_dict.get(field) and heuristic.get(field):
            data_dict[field] = heuristic[field]

    # List fields
    list_fields = ["skills", "projects", "certifications", "education"]
    for field in list_fields:
        model_list = data_dict.get(field, []) if isinstance(data_dict.get(field), list) else []
        heuristic_list = heuristic.get(field, []) if isinstance(heuristic.get(field), list) else []

        if not model_list and heuristic_list:
            data_dict[field] = heuristic_list
        elif heuristic_list and len(model_list) < len(heuristic_list):
            # Merge unique items preserving order
            merged = model_list[:]
            for item in heuristic_list:
                if item not in merged:
                    merged.append(item)
            data_dict[field] = merged

    return data_dict


def _dict_entry_to_text(entry: dict) -> str:
    """Convert structured dict entries into readable single-line text."""
    if not isinstance(entry, dict):
        return str(entry).strip()

    preferred_order = [
        "title",
        "role",
        "company",
        "institution",
        "degree",
        "issuer",
        "field_of_study",
        "technologies",
        "description",
        "date",
        "duration",
    ]

    parts = []
    used = set()
    for key in preferred_order:
        value = entry.get(key)
        if value is None or value == "":
            continue
        used.add(key)
        parts.append(str(value).strip())

    for key, value in entry.items():
        if key in used or value is None or value == "":
            continue
        if isinstance(value, list):
            value = ", ".join(str(v).strip() for v in value if str(v).strip())
        parts.append(str(value).strip())

    return " | ".join([p for p in parts if p])


def _normalize_to_str_list(value) -> list[str]:
    """Normalize a mixed value (list/dict/str) into a list of strings."""
    if value is None or value == "":
        return []

    if isinstance(value, str):
        clean = value.strip()
        return [clean] if clean else []

    if isinstance(value, dict):
        return [_dict_entry_to_text(value)]

    if isinstance(value, list):
        out = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):
                text = _dict_entry_to_text(item)
            elif isinstance(item, list):
                text = ", ".join(str(v).strip() for v in item if str(v).strip())
            else:
                text = str(item).strip()
            if text:
                out.append(text)
        return out

    return [str(value).strip()]


def _normalize_model_output_schema(data_dict: dict) -> dict:
    """Normalize common LLM output variants to match ResumeData schema."""
    normalized = dict(data_dict or {})

    # Common alias from some models.
    if "roles" in normalized and not normalized.get("suggested_roles"):
        normalized["suggested_roles"] = normalized.get("roles")

    # Flatten nested skills object (languages/tools/libraries -> one list).
    skills_value = normalized.get("skills")
    if isinstance(skills_value, dict):
        merged_skills = []
        for _, section_value in skills_value.items():
            merged_skills.extend(_normalize_to_str_list(section_value))
        deduped = []
        seen = set()
        for skill in merged_skills:
            key = skill.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(skill)
        normalized["skills"] = deduped

    list_fields = [
        "skills",
        "education",
        "experience",
        "achievements",
        "certifications",
        "projects",
        "publications",
        "languages",
        "volunteer",
        "awards",
        "interests",
        "key_strengths",
        "suggested_roles",
    ]
    for field in list_fields:
        normalized[field] = _normalize_to_str_list(normalized.get(field))

    # Ensure summary fields are plain strings when provided as dict/list.
    for field in ["summary", "ai_summary"]:
        value = normalized.get(field)
        if isinstance(value, dict):
            normalized[field] = _dict_entry_to_text(value)
        elif isinstance(value, list):
            normalized[field] = " ".join(
                _normalize_to_str_list(value)
            ).strip() or None

    return normalized


def parse_resume_data_from_response(text: str, content: str) -> ResumeData:
    """Parse and validate model output into ResumeData (LLM output only)."""
    # Parse JSON from response
    data_dict = _parse_json_from_response(content)

    # Normalize structured model output (dict objects/lists) to ResumeData schema.
    data_dict = _normalize_model_output_schema(data_dict)
    data_dict = _clean_all_fields(data_dict)

    # NOTE: Helper fixes are temporarily disabled as requested.
    # data_dict = _fix_project_experience_swap(data_dict)
    # heuristic = _heuristic_extract_from_text(text)
    # data_dict = _merge_with_heuristics(data_dict, heuristic)

    # Validate and create ResumeData object
    try:
        resume_data = ResumeData(**data_dict)
        logger.info(f"Successfully created ResumeData for: {resume_data.name or 'Unknown'}")
        return resume_data
    except Exception as e:
        logger.warning(f"Validation error: {e}")
        valid_fields = {}
        for field_name in ResumeData.model_fields.keys():
            if field_name in data_dict:
                valid_fields[field_name] = data_dict[field_name]
        return ResumeData(**valid_fields)


def _parse_json_from_response(response_text: str) -> dict:
    """Extract JSON from LLM response (handles reasoning text, code blocks, and plain JSON)"""
    if not response_text:
        raise ValueError("Empty response from model")
    
    # First try: direct parse (unlikely but worth trying)
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Second try: extract JSON between triple backticks
    try:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except json.JSONDecodeError:
        pass
    
    # Third try: find JSON object in the text (skip reasoning text)
    # Look for the pattern that starts a JSON object
    try:
        # Find all { positions and try to parse from each one
        start_positions = [i for i, c in enumerate(response_text) if c == '{']
        
        for start in start_positions:
            # Find matching closing brace
            brace_count = 0
            end = start
            for i in range(start, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            json_str = response_text[start:end]
            try:
                result = json.loads(json_str)
                logger.info("Extracted JSON from boundaries")
                return result
            except json.JSONDecodeError:
                continue
        
    except Exception as e:
        logger.warning(f"JSON extraction attempt failed: {e}")
    
    # Final attempt: use regex to find JSON-like pattern
    try:
        pattern = r'\{\s*"name"\s*:'  # Look for JSON starting with "name" field
        match = re.search(pattern, response_text)
        if match:
            start = match.start()
            # Find the matching closing brace
            brace_count = 0
            end = start
            for i in range(start, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            json_str = response_text[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Response: {response_text[:500]}")
        raise ValueError(f"Failed to parse AI response as JSON: {str(e)}")
    
    raise ValueError("Could not find valid JSON in response")


def check_ollama_connection(model: str = "qwen3.5:2b") -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        response.raise_for_status()
        data = response.json()
        
        available_models = []
        for m in data.get("models", []):
            name = m.get("name", "") or m.get("model", "")
            available_models.append(name)
            if ":" in name:
                available_models.append(name.split(":")[0])
                
        return model in available_models or model.split(":")[0] in available_models
    except Exception as e:
        logger.error(f"Ollama connection check failed: {e}")
        return False
