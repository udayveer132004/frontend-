"""
Job Matcher Engine.
Computes semantic similarity between resume and job descriptions.
Reuses the RAG engine's embedding model for efficiency.
"""

import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class JobMatcher:
    def __init__(self, embedding_model):
        """
        Initialize with an existing embedding model (LangChain Embeddings interface).
        Args:
            embedding_model: Instance with .embed_query() and .embed_documents()
        """
        self.embedding_model = embedding_model
        if not self.embedding_model:
            logger.warning("JobMatcher initialized without embedding model! Scoring will fail.")

    def match_jobs(self, resume_text: str, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank jobs by semantic similarity to resume text.
        
        Args:
            resume_text: The full text of the resume.
            jobs: List of job dictionaries (must have 'title', 'description', 'tags').
            
        Returns:
            List of job dictionaries with added 'match_score' (0-100), sorted descending.
        """
        if not self.embedding_model or not resume_text or not jobs:
            return jobs

        try:
            # 1. Embed Resume (Query)
            # Use embed_query for single document
            resume_vector = self.embedding_model.embed_query(resume_text)
            
            # 2. Prepare Job Texts
            # Combine relevant fields: Title is very important, tags are good keywords, description provides context.
            # Weighting idea: Repeat title to increase its importance? For now, flat concatenation.
            job_texts = []
            for job in jobs:
                # Sanitize fields
                title = job.get("title", "")
                desc = job.get("description", "")
                # Clean description (it might be HTML or long text) - naive approach for now
                tags = " ".join(job.get("tags", []))
                
                # Composite text
                text = f"{title} {tags} {desc[:1000]}" # Truncate description for speed/context window
                job_texts.append(text)
                
            # 3. Embed Jobs (Documents) - Batch
            if not job_texts:
                return jobs
                
            job_vectors = self.embedding_model.embed_documents(job_texts)
            
            # 4. Compute Cosine Similarity
            # Cosine Sim = (A . B) / (||A|| * ||B||)
            # SentenceTransformers often yield normalized vectors, so dot product is sufficient.
            # But let's do full calculation to be safe.
            
            resume_norm = np.linalg.norm(resume_vector)
            
            scored_jobs = []
            for i, job in enumerate(jobs):
                job_vec = job_vectors[i]
                job_norm = np.linalg.norm(job_vec)
                
                if resume_norm == 0 or job_norm == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(resume_vector, job_vec) / (resume_norm * job_norm)
                
                # Convert to percentage 0-100
                score = round(similarity * 100)
                
                # Update job object
                job_with_score = job.copy()
                job_with_score['match_score'] = score
                scored_jobs.append(job_with_score)
                
            # 5. Sort by Score Descending
            scored_jobs.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            
            return scored_jobs
            
        except Exception as e:
            logger.error(f"Job matching failed: {e}")
            return jobs
