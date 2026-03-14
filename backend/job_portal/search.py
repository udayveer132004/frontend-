"""
Job Search Engine aggregating multiple remote job APIs.
"""

import logging
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class JobSearchEngine:
    """
    Search for jobs using multiple remote job providers.
    """

    REMOTE_OK_URL = "https://remoteok.com/api"
    WEWORKREMOTELY_RSS_URL = "https://weworkremotely.com/remote-jobs.rss"
    JOBICY_URL = "https://jobicy.com/api/v2/remote-jobs"
    REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
    
    def __init__(self):
        # Identify our client properly as requested by API etiquette
        self.headers = {
            "User-Agent": "AI-Job-Assistant/1.0 (Educational Project)"
        }
        
    def search_jobs(self, query: str) -> List[Dict[str, Any]]:
        """
        Search remote jobs by aggregating multiple sources and filtering locally.
        
        Args:
            query: Job role or keyword (e.g., "Python", "Backend Engineer")
            
        Returns:
            List of job dictionaries.
        """
        all_jobs: List[Dict[str, Any]] = []

        all_jobs.extend(self._fetch_remote_ok_jobs())
        all_jobs.extend(self._fetch_weworkremotely_jobs())
        all_jobs.extend(self._fetch_jobicy_jobs())
        all_jobs.extend(self._fetch_remotive_jobs())

        return self._filter_normalized_jobs(all_jobs, query)

    def _fetch_remote_ok_jobs(self) -> List[Dict[str, Any]]:
        """Fetch and normalize jobs from Remote OK."""
        try:
            logger.info(f"Fetching jobs from: {self.REMOTE_OK_URL}")
            response = requests.get(self.REMOTE_OK_URL, headers=self.headers, timeout=15)
            response.raise_for_status()

            jobs = response.json()
            if jobs and "legal" in jobs[0]:
                jobs = jobs[1:]

            normalized = []
            for job in jobs:
                title = (job.get("position") or "").strip()
                company = (job.get("company") or "").strip()
                if not title or not company:
                    continue

                normalized.append({
                    "id": f"remoteok:{job.get('id')}",
                    "title": title,
                    "company": company,
                    "location": job.get("location") or "Remote",
                    "date": (job.get("date") or "").split("T")[0] if job.get("date") else "Recent",
                    "tags": job.get("tags") or [],
                    "url": job.get("url"),
                    "apply_url": job.get("apply_url") or job.get("url"),
                    "salary_min": job.get("salary_min"),
                    "salary_max": job.get("salary_max"),
                    "description": self._truncate(job.get("description", "")),
                    "source": "Remote OK",
                })
            return normalized
        except Exception as e:
            logger.warning(f"Remote OK fetch failed: {e}")
            return []

    def _fetch_weworkremotely_jobs(self) -> List[Dict[str, Any]]:
        """Fetch and normalize jobs from We Work Remotely RSS."""
        try:
            logger.info(f"Fetching jobs from: {self.WEWORKREMOTELY_RSS_URL}")
            response = requests.get(self.WEWORKREMOTELY_RSS_URL, headers=self.headers, timeout=15)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            items = root.findall("./channel/item")

            normalized = []
            for item in items:
                raw_title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()

                category_nodes = item.findall("category")
                tags = [c.text.strip() for c in category_nodes if c is not None and c.text]

                company, title = self._split_company_and_title(raw_title)
                if not title:
                    continue

                desc = (item.findtext("description") or "").strip()

                normalized.append({
                    "id": f"wwr:{link}",
                    "title": title,
                    "company": company or "Unknown Company",
                    "location": "Remote",
                    "date": pub_date or "Recent",
                    "tags": tags,
                    "url": link,
                    "apply_url": link,
                    "salary_min": None,
                    "salary_max": None,
                    "description": self._truncate(desc),
                    "source": "We Work Remotely",
                })
            return normalized
        except Exception as e:
            logger.warning(f"We Work Remotely fetch failed: {e}")
            return []

    def _fetch_jobicy_jobs(self) -> List[Dict[str, Any]]:
        """Fetch and normalize jobs from Jobicy API."""
        try:
            logger.info(f"Fetching jobs from: {self.JOBICY_URL}")
            response = requests.get(self.JOBICY_URL, headers=self.headers, timeout=15)
            response.raise_for_status()

            payload = response.json()
            jobs = payload.get("jobs") if isinstance(payload, dict) else []

            normalized = []
            for job in jobs:
                title = self._first(job, ["jobTitle", "title", "position"], "")
                company = self._first(job, ["companyName", "company", "company_name"], "")
                if not title or not company:
                    continue

                tags = job.get("jobIndustries") or job.get("tags") or []
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]

                url = self._first(job, ["url", "jobUrl", "applyUrl"], "")
                location = self._first(job, ["jobGeo", "location", "candidate_required_location"], "Remote")
                posted = self._first(job, ["pubDate", "date", "createdAt"], "Recent")

                normalized.append({
                    "id": f"jobicy:{self._first(job, ['id', 'jobId'], url)}",
                    "title": title,
                    "company": company,
                    "location": location or "Remote",
                    "date": posted,
                    "tags": tags,
                    "url": url,
                    "apply_url": self._first(job, ["applyUrl", "url", "jobUrl"], url),
                    "salary_min": None,
                    "salary_max": None,
                    "description": self._truncate(self._first(job, ["jobDescription", "description"], "")),
                    "source": "Jobicy",
                })
            return normalized
        except Exception as e:
            logger.warning(f"Jobicy fetch failed: {e}")
            return []

    def _fetch_remotive_jobs(self) -> List[Dict[str, Any]]:
        """Fetch and normalize jobs from Remotive API."""
        try:
            logger.info(f"Fetching jobs from: {self.REMOTIVE_URL}")
            response = requests.get(self.REMOTIVE_URL, headers=self.headers, timeout=15)
            response.raise_for_status()

            payload = response.json()
            jobs = payload.get("jobs") if isinstance(payload, dict) else []

            normalized = []
            for job in jobs:
                title = (job.get("title") or "").strip()
                company = (job.get("company_name") or "").strip()
                if not title or not company:
                    continue

                normalized.append({
                    "id": f"remotive:{job.get('id')}",
                    "title": title,
                    "company": company,
                    "location": job.get("candidate_required_location") or "Remote",
                    "date": (job.get("publication_date") or "").split("T")[0] if job.get("publication_date") else "Recent",
                    "tags": job.get("tags") or [],
                    "url": job.get("url"),
                    "apply_url": job.get("url"),
                    "salary_min": None,
                    "salary_max": None,
                    "description": self._truncate(job.get("description", "")),
                    "source": "Remotive",
                })
            return normalized
        except Exception as e:
            logger.warning(f"Remotive fetch failed: {e}")
            return []

    def _filter_normalized_jobs(self, jobs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Filter pre-normalized jobs based on user query and deduplicate by URL."""
        filtered = []
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
        
        seen_urls = set()
        for job in jobs:
            try:
                if not job.get("company") or not job.get("title"):
                    continue

                title = job.get("title", "").lower()
                tags = [t.lower() for t in job.get("tags", [])]
                location = job.get("location", "").lower()
                description = job.get("description", "").lower()

                match = False
                for kw in keywords:
                    if kw in title or kw in location:
                        match = True
                        break
                    if any(kw in tag for tag in tags):
                        match = True
                        break
                    if kw in description:
                        match = True
                        break

                if not match:
                    continue

                dedupe_key = (job.get("apply_url") or job.get("url") or "").strip().lower()
                if dedupe_key and dedupe_key in seen_urls:
                    continue
                if dedupe_key:
                    seen_urls.add(dedupe_key)

                filtered.append(job)
            except Exception:
                continue

        return filtered

    @staticmethod
    def _split_company_and_title(raw_title: str) -> tuple[str, str]:
        """Heuristic split for feed titles often shaped as 'Company: Title'."""
        if ":" in raw_title:
            company, title = raw_title.split(":", 1)
            return company.strip(), title.strip()
        return "", raw_title.strip()

    @staticmethod
    def _first(data: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
        """Return the first available non-empty value from candidate keys."""
        for key in keys:
            value = data.get(key)
            if value is not None and value != "":
                return value
        return default

    @staticmethod
    def _truncate(text: str, limit: int = 200) -> str:
        """Create a short description preview."""
        if not text:
            return ""
        clean = " ".join(str(text).split())
        if len(clean) <= limit:
            return clean
        return clean[:limit] + "..."

# Global instance
_job_engine = JobSearchEngine()

def search_jobs(role: str, location: str = "") -> List[Dict[str, Any]]:
    """Public wrapper for job search."""
    search_term = role
    if location:
        search_term += f",{location}"
        
    return _job_engine.search_jobs(search_term)
