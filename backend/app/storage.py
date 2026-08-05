"""Storage abstraction.

Two interchangeable backends expose the same interface so the frontend upload/download
code is identical in mock and production:

- ``R2Storage``: presigns real Cloudflare R2 (S3 API) PUT/GET URLs. The browser and the
  RunPod worker talk to R2 directly; bytes never pass through this backend.
- ``LocalStorage``: for mock mode. Presigned URLs point back at this backend's
  ``/api/storage/{key}`` endpoint, which reads/writes ``scratch/storage`` on local disk.
"""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .config import Settings


class Storage(ABC):
    @abstractmethod
    def presign_put(self, key: str, content_type: str = "application/octet-stream") -> str:
        ...

    @abstractmethod
    def presign_get(self, key: str) -> str:
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abstractmethod
    def copy(self, src_key: str, dst_key: str) -> None:
        ...


class R2Storage(Storage):
    def __init__(self, settings: Settings):
        import boto3
        from botocore.config import Config as BotoConfig

        self.settings = settings
        self.bucket = settings.r2_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.resolved_r2_endpoint,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
            config=BotoConfig(signature_version="s3v4"),
        )

    def presign_put(self, key: str, content_type: str = "application/octet-stream") -> str:
        return self._client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=self.settings.presign_ttl_seconds,
        )

    def presign_get(self, key: str) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.settings.presign_ttl_seconds,
        )

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def copy(self, src_key: str, dst_key: str) -> None:
        self._client.copy_object(
            Bucket=self.bucket,
            Key=dst_key,
            CopySource={"Bucket": self.bucket, "Key": src_key},
        )


class LocalStorage(Storage):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path(settings.local_storage_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.base = settings.backend_public_url.rstrip("/")

    def _path(self, key: str) -> Path:
        # Prevent path traversal; keys are app-generated but be safe anyway.
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root)):
            raise ValueError("invalid key")
        return p

    def presign_put(self, key: str, content_type: str = "application/octet-stream") -> str:
        return f"{self.base}/api/storage/{key}"

    def presign_get(self, key: str) -> str:
        return f"{self.base}/api/storage/{key}"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def copy(self, src_key: str, dst_key: str) -> None:
        dst = self._path(dst_key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(src_key), dst)

    # Local-only helpers used by the /api/storage route.
    def write_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def read_path(self, key: str) -> Optional[Path]:
        p = self._path(key)
        return p if p.exists() else None


def build_storage(settings: Settings) -> Storage:
    if settings.mock_mode or not settings.r2_configured:
        return LocalStorage(settings)
    return R2Storage(settings)
