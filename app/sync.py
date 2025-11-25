"""
Curriculum Sync Service
Fetches curriculum from backend and caches locally.

Design:
- Check version hash before downloading
- Only download if version changed
- Store curriculum as JSON for fast loading
- Retry on network failures with exponential backoff
"""

import json
import os
import httpx
import logging
from typing import Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)


class CurriculumSync:
    def __init__(self, backend_url: str, data_dir: str = "./data"):
        self.backend_url = backend_url.rstrip("/")
        self.data_dir = data_dir
        self.timeout = httpx.Timeout(30.0)  # 30 second timeout
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
    
    def _get_local_version(self) -> str:
        """Get locally stored version hash"""
        version_path = os.path.join(self.data_dir, "version.txt")
        if os.path.exists(version_path):
            with open(version_path, "r") as f:
                return f.read().strip()
        return ""
    
    def _save_version(self, version: str):
        """Save version hash locally"""
        version_path = os.path.join(self.data_dir, "version.txt")
        with open(version_path, "w") as f:
            f.write(version)
    
    def _save_curriculum(self, data: dict):
        """Save curriculum data locally"""
        curriculum_path = os.path.join(self.data_dir, "curriculum.json")
        with open(curriculum_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def check_version(self) -> dict:
        """
        Check if backend has newer version.
        Retries up to 3 times on network errors.
        """
        local_version = self._get_local_version()
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.backend_url}/v1/curriculum/version",
                headers={"X-Local-Version": local_version}
            )
            response.raise_for_status()
            return response.json()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def fetch_curriculum(self) -> dict:
        """
        Fetch full curriculum from backend.
        Retries up to 3 times on network errors.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.backend_url}/v1/curriculum/export"
            )
            response.raise_for_status()
            return response.json()
    
    async def sync(self) -> dict:
        """
        Sync curriculum from backend.
        
        Returns:
        - success: bool
        - version: str
        - word_count: int
        - lesson_count: int
        - changed: bool (True if new data was downloaded)
        """
        try:
            # Check version first
            logger.info(f"Checking curriculum version at {self.backend_url}")
            version_info = await self.check_version()
            
            if not version_info.get("changed", True):
                # No changes, skip download
                logger.info("Curriculum unchanged, skipping download")
                return {
                    "success": True,
                    "version": version_info.get("version", ""),
                    "word_count": version_info.get("wordCount", 0),
                    "lesson_count": version_info.get("lessonCount", 0),
                    "changed": False
                }
            
            # Fetch new curriculum
            logger.info("Downloading curriculum update...")
            curriculum = await self.fetch_curriculum()
            
            # Save locally
            self._save_curriculum(curriculum)
            self._save_version(curriculum.get("version", ""))
            
            word_count = len(curriculum.get("words", {}))
            logger.info(f"Curriculum saved: {word_count} words")
            
            return {
                "success": True,
                "version": curriculum.get("version", ""),
                "word_count": word_count,
                "lesson_count": curriculum.get("lessonCount", 0),
                "changed": True
            }
        except Exception as e:
            logger.error(f"Sync failed after retries: {e}")
            return {
                "success": False,
                "version": "",
                "word_count": 0,
                "lesson_count": 0,
                "changed": False
            }
