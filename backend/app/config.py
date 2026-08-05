"""Application configuration, loaded from environment / backend/.env.

All secrets live here and never leave the server. See ../.env.example.
"""
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    retone_env: str = "development"
    frontend_origins: str = "http://localhost:5173"
    # Public base URL of this backend, used to build local-storage URLs in mock mode.
    backend_public_url: str = "http://localhost:8000"

    # --- Mock mode ---
    # When true, uploads go to local disk and "separation" is faked, so the whole
    # UI works with no RunPod/R2 credentials.
    mock_mode: bool = True

    # --- Cloudflare R2 ---
    r2_account_id: Optional[str] = None
    r2_access_key_id: Optional[str] = None
    r2_secret_access_key: Optional[str] = None
    r2_bucket: str = "retone"
    r2_endpoint_url: Optional[str] = None
    presign_ttl_seconds: int = 3600

    # --- RunPod ---
    runpod_api_key: Optional[str] = None
    runpod_endpoint_id: Optional[str] = None
    runpod_poll_interval_seconds: float = 3.0
    runpod_job_timeout_seconds: int = 600

    # --- Hugging Face (optional) ---
    hf_token: Optional[str] = None

    # --- Local-storage dir for mock mode (relative to backend/) ---
    local_storage_dir: str = "scratch/storage"

    @property
    def origins(self) -> List[str]:
        return [o.strip() for o in self.frontend_origins.split(",") if o.strip()]

    @property
    def resolved_r2_endpoint(self) -> Optional[str]:
        if self.r2_endpoint_url:
            return self.r2_endpoint_url
        if self.r2_account_id:
            return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"
        return None

    @property
    def r2_configured(self) -> bool:
        return bool(
            self.r2_access_key_id
            and self.r2_secret_access_key
            and self.resolved_r2_endpoint
        )

    @property
    def runpod_configured(self) -> bool:
        return bool(self.runpod_api_key and self.runpod_endpoint_id)


@lru_cache
def get_settings() -> Settings:
    return Settings()
