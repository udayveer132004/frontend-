"""Pydantic models for structured resume data."""

from typing import Optional
from pydantic import BaseModel, Field


class ResumeData(BaseModel):
    """Structured resume data extracted from a resume document."""
    
    # Contact Information
    name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Location/Address")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    github: Optional[str] = Field(None, description="GitHub profile URL")
    portfolio: Optional[str] = Field(None, description="Portfolio or personal website")
    
    # Core Sections
    skills: list[str] = Field(default_factory=list, description="List of technical and soft skills")
    education: list[str] = Field(default_factory=list, description="List of education entries (degree, institution, dates)")
    experience: list[str] = Field(default_factory=list, description="List of work experience entries (role, company, dates, details)")
    
    # Additional Sections
    summary: Optional[str] = Field(None, description="Professional summary or objective")
    achievements: list[str] = Field(default_factory=list, description="List of notable achievements")
    certifications: list[str] = Field(default_factory=list, description="Professional certifications")
    projects: list[str] = Field(default_factory=list, description="List of projects with descriptions")
    publications: list[str] = Field(default_factory=list, description="List of publications")
    languages: list[str] = Field(default_factory=list, description="Spoken languages and proficiency")
    volunteer: list[str] = Field(default_factory=list, description="List of volunteer work")
    awards: list[str] = Field(default_factory=list, description="Awards and honors received")
    interests: list[str] = Field(default_factory=list, description="List of interests")
    
    # AI-Generated Insights
    ai_summary: Optional[str] = Field(None, description="AI-generated professional summary")
    key_strengths: list[str] = Field(default_factory=list, description="AI-identified key strengths")
    suggested_roles: list[str] = Field(default_factory=list, description="AI-suggested job titles (1-2) based on skills/experience")
    
    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+1-234-567-8900",
                "skills": ["Python", "Machine Learning", "FastAPI"],
                "education": "BS in Computer Science, Stanford University, 2020",
                "experience": "Software Engineer at Tech Corp (2020-2023)",
                "achievements": "Led team of 5 engineers to deliver ML platform",
                "certifications": ["AWS Certified Developer"],
                "ai_summary": "Experienced software engineer with ML expertise"
            }
        }
