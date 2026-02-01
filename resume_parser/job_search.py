"""
Job Search Engine using Remote OK API.
"""

import logging
import requests
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class JobSearchEngine:
    """
    Search for jobs using the Remote OK API.
    API URL: https://remoteok.com/api
    """
    
    BASE_URL = "https://remoteok.com/api"
    
    def __init__(self):
        # Identify our client properly as requested by API etiquette
        self.headers = {
            "User-Agent": "AI-Job-Assistant/1.0 (Educational Project)"
        }
        
    def search_remote_ok(self, role: str) -> List[Dict[str, Any]]:
        """
        Search for remote jobs by fetching the feed and filtering locally.
        
        Args:
            role: Job role or keyword (e.g., "Python", "Backend Engineer")
            
        Returns:
            List of job dictionaries.
        """
        try:
            # Strategy: Fetch the full feed (or broad set) and filter locally
            # This handles multi-word titles like "Backend Engineer" much better than API tags
            query_url = self.BASE_URL
            
            logger.info(f"Fetching jobs from: {query_url}")
            
            response = requests.get(query_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            jobs = response.json()
            
            # First element is legal text
            if jobs and len(jobs) > 0 and "legal" in jobs[0]:
                jobs = jobs[1:]
                
            return self._filter_and_normalize(jobs, role)
            
        except Exception as e:
            logger.error(f"Job search failed: {e}")
            return []

    def _filter_and_normalize(self, jobs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Filter jobs based on query and normalize data."""
        normalized = []
        if not query:
            return []
            
        # Parse query: "Backend Engineer, Python" -> ["backend engineer", "python"]
        # Split by comma first
        parts = [p.strip().lower() for p in query.split(",")]
        
        # Further split by " or " if present? No, let's treat comma as OR.
        # But wait, user might type "Backend Engineer or Full Stack".
        # Let's do a simple heuristic: split by "," then handle " or "
        
        keywords = []
        for part in parts:
            if " or " in part:
                keywords.extend([k.strip() for k in part.split(" or ")])
            else:
                keywords.append(part)
        
        keywords = [k for k in keywords if k]
        
        for job in jobs:
            try:
                # Basic validation
                if not job.get("company") or not job.get("position"):
                    continue

                # Prepare search text
                title = job.get("position", "").lower()
                tags = [t.lower() for t in job.get("tags", [])]
                location = job.get("location", "").lower()
                description = job.get("description", "").lower() # Description is heavy, maybe search title/tags/location first
                
                # Check for matches
                # Logic: If ANY keyword is found in Title OR Tags, include it.
                # Use word boundary check or simple substring? Substring is easier and usually fine.
                
                match = False
                for kw in keywords:
                    if kw in title or kw in tags or kw in location: 
                        match = True
                        break
                    # Optional: Search description if not found in title? 
                    # can lead to noise. Let's stick to high-signal fields first.
                
                if not match:
                    continue
                
                # Extract key fields
                normalized.append({
                    "id": job.get("id"),
                    "title": job.get("position"),
                    "company": job.get("company"),
                    "location": job.get("location", "Remote"),
                    "date": job.get("date", "").split("T")[0] if job.get("date") else "Recent",
                    "tags": job.get("tags", []),
                    "url": job.get("url"), # Direct link to job on Remote OK
                    "apply_url": job.get("apply_url"),
                    "salary_min": job.get("salary_min"),
                    "salary_max": job.get("salary_max"),
                    "description": job.get("description", "")[:200] + "..." # Preview
                })
            except Exception as e:
                continue
                
        return normalized

# Global instance
_job_engine = JobSearchEngine()

def search_jobs(role: str, location: str = "") -> List[Dict[str, Any]]:
    """Public wrapper for job search."""
    # Location filtering is client-side for Remote OK API relative to tags
    # or requires a specific tag. For now, we search by role primarily.
    # If location is provided, we could try adding it as a tag, but Remote OK treats locations as tags too.
    
    search_term = role
    if location:
        search_term += f",{location}"
        
    return _job_engine.search_remote_ok(search_term)
