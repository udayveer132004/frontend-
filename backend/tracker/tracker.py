"""
Application Tracker — persistent job-application CRUD backed by a JSON file.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

VALID_STATUSES = [
    "Applied",
    "Awaiting Interview",
    "Interviewed",
    "Offered",
    "Rejected",
    "Withdrawn",
]

VALID_ROLE_TYPES = [
    "Full-time",
    "Part-time",
    "Internship",
    "Contract",
    "Freelance",
]

# ─── Data Model ───────────────────────────────────────────────────────────────


class ApplicationEntry(BaseModel):
    """A single job-application record."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_title: str
    company: str
    role_type: str = "Full-time"
    status: str = "Applied"
    application_date: str = Field(
        default_factory=lambda: date.today().isoformat()
    )
    job_url: Optional[str] = None
    notes: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )


# ─── Tracker ──────────────────────────────────────────────────────────────────


class ApplicationTracker:
    """
    CRUD manager for job applications.

    All data is stored in *storage_path* as a JSON array so it survives
    across app restarts.
    """

    def __init__(self, storage_path: str = "data/applications.json") -> None:
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        self._applications: list[ApplicationEntry] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            self._applications = []
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._applications = [ApplicationEntry(**item) for item in raw]
            logger.info(
                "Loaded %d application(s) from %s",
                len(self._applications),
                self.storage_path,
            )
        except Exception as exc:
            logger.error("Could not load applications: %s", exc)
            self._applications = []

    def _save(self) -> None:
        try:
            with open(self.storage_path, "w", encoding="utf-8") as fh:
                json.dump(
                    [a.model_dump() for a in self._applications],
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as exc:
            logger.error("Could not save applications: %s", exc)

    # ── CRUD ────────────────────────────────────────────────────────────────

    def add(
        self,
        job_title: str,
        company: str,
        role_type: str = "Full-time",
        status: str = "Applied",
        application_date: str | None = None,
        job_url: str | None = None,
        notes: str | None = None,
    ) -> ApplicationEntry:
        """Create and persist a new application entry."""
        entry = ApplicationEntry(
            job_title=job_title.strip(),
            company=company.strip(),
            role_type=role_type,
            status=status,
            application_date=application_date or date.today().isoformat(),
            job_url=job_url.strip() if job_url else None,
            notes=notes.strip() if notes else None,
        )
        self._applications.append(entry)
        self._save()
        logger.info("Added application: %s at %s", job_title, company)
        return entry

    def update(
        self,
        app_id: str,
        job_title: str | None = None,
        company: str | None = None,
        role_type: str | None = None,
        status: str | None = None,
        application_date: str | None = None,
        job_url: str | None = None,
        notes: str | None = None,
    ) -> ApplicationEntry | None:
        """Update an existing entry by its UUID. Returns the updated entry or None."""
        for idx, app in enumerate(self._applications):
            if app.id == app_id:
                data = app.model_dump()
                if job_title is not None:
                    data["job_title"] = job_title.strip()
                if company is not None:
                    data["company"] = company.strip()
                if role_type is not None:
                    data["role_type"] = role_type
                if status is not None:
                    data["status"] = status
                if application_date is not None:
                    data["application_date"] = application_date
                if job_url is not None:
                    data["job_url"] = job_url.strip() if job_url.strip() else None
                if notes is not None:
                    data["notes"] = notes.strip() if notes.strip() else None
                data["updated_at"] = datetime.now().isoformat(timespec="seconds")
                updated = ApplicationEntry(**data)
                self._applications[idx] = updated
                self._save()
                logger.info("Updated application id=%s", app_id)
                return updated
        logger.warning("Application id=%s not found for update", app_id)
        return None

    def delete(self, app_id: str) -> bool:
        """Delete an entry by UUID. Returns True on success."""
        before = len(self._applications)
        self._applications = [a for a in self._applications if a.id != app_id]
        if len(self._applications) < before:
            self._save()
            logger.info("Deleted application id=%s", app_id)
            return True
        logger.warning("Application id=%s not found for deletion", app_id)
        return False

    def get_all(self) -> list[ApplicationEntry]:
        return list(self._applications)

    def get_by_id(self, app_id: str) -> ApplicationEntry | None:
        for app in self._applications:
            if app.id == app_id:
                return app
        return None

    # ── Statistics ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, int]:
        """Return counts grouped by interesting status buckets."""
        total = len(self._applications)
        interviews = sum(
            1 for a in self._applications
            if a.status in ("Awaiting Interview", "Interviewed")
        )
        offers = sum(1 for a in self._applications if a.status == "Offered")
        rejections = sum(1 for a in self._applications if a.status == "Rejected")
        return {
            "total": total,
            "interviews": interviews,
            "offers": offers,
            "rejections": rejections,
        }

    # ── DataFrame-ready helpers ─────────────────────────────────────────────

    def to_dataframe_rows(self) -> list[list]:
        """Return data as list-of-lists suitable for gr.DataFrame."""
        rows = []
        for i, app in enumerate(self._applications, start=1):
            rows.append(
                [
                    i,
                    app.job_title,
                    app.company,
                    app.role_type,
                    app.status,
                    app.application_date,
                    app.job_url or "",
                    app.id,  # hidden ID column used for edit/delete
                ]
            )
        return rows

    DATAFRAME_HEADERS = [
        "#",
        "Job Title",
        "Company",
        "Role Type",
        "Status",
        "Date",
        "URL",
        "ID",
    ]
