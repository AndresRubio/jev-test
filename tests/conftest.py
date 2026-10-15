import pytest

from local_jev.config import BatchSettings, LayaSettings, Settings


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings that keep every file inside tmp_path and never download weights."""
    return Settings(
        log_dir=tmp_path / "logs",
        results_dir=tmp_path / "results",
        batch=BatchSettings(allowed_roots=(tmp_path,)),
        laya=LayaSettings(prefetch=()),
    )
