"""The default engine: laya-mlx's Router on pinned, verified Laya checkpoints."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from local_jev import checkpoints
from local_jev.config import LayaSettings
from local_jev.engines.base import EngineCapabilities, EngineResult
from local_jev.errors import ModelUnavailable
from local_jev.schema import MODEL_HINT as UNKNOWN_MODEL_HINT

SHORTLIST_K = 20


def _laya() -> Any:
    import laya_mlx

    return laya_mlx


def default_router_factory(root: Path, dtype: str) -> Any:
    """A Router whose checkpoints point at the pinned snapshot instead of the Hub's newest revision."""
    models = {
        name: (str(root), checkpoint.subfolder) if checkpoint.subfolder else str(root)
        for name, checkpoint in checkpoints.CHECKPOINTS.items()
    }
    return _laya().Router(models=models, max_loaded=2, default="english", dtype=dtype)


def default_tokenizer_factory(path: Path) -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


class LayaMLXEngine:
    name = "laya-mlx"
    capabilities = EngineCapabilities(laya_guards=True)

    def __init__(
        self,
        settings: LayaSettings,
        *,
        cache_dir: Path | None = None,
        router_factory: Callable[[Path, str], Any] = default_router_factory,
        shortlist_fn: Callable[..., dict[str, Any]] | None = None,
        embed_factory: Callable[[Any], Any] | None = None,
        tokenizer_factory: Callable[[Path], Any] = default_tokenizer_factory,
    ) -> None:
        self._settings = settings
        self._cache_dir = cache_dir
        self._root = checkpoints.snapshot_root(settings.repo, settings.revision, cache_dir)
        self._router_factory = router_factory
        self._shortlist = shortlist_fn or (
            lambda *args, **kwargs: _laya().predict_shortlist(*args, **kwargs)
        )
        self._embed_factory = embed_factory or (lambda agent: _laya().embed_fn_from_agent(agent))
        self._tokenizer_factory = tokenizer_factory
        self._router = router_factory(self._root, settings.dtype)
        self._lock = threading.Lock()
        self._ready: set[str] = set()
        self._downloading: set[str] = set()
        self._tokenizers: dict[str, Any] = {}

    def decide(self, state: Any, questions: dict[str, dict[str, Any]], model: str) -> EngineResult:
        name, reason = self._pick(state, questions, model)
        self._require(name)
        try:
            raw = self._predict(self._router, state, questions, name)
        except FloatingPointError:
            raw = self._predict_float32(state, questions, name)
        return EngineResult(
            answers=raw["answers"],
            usage=raw.get("usage", {}),
            model=f"laya-{name}",
            checkpoint=name,
            routing_reason=reason,
            shortlist=raw.get("shortlist", {}),
        )

    def count_tokens(self, text: str, model: str) -> int:
        name, _reason = self._pick(text, None, model)
        self._require(name)
        with self._lock:
            tokenizer = self._tokenizers.get(name)
            if tokenizer is None:
                path = checkpoints.checkpoint_dir(self._root, name) / "tokenizer" / "tokenizer.json"
                tokenizer = self._tokenizer_factory(path)
                self._tokenizers[name] = tokenizer
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    def prefetch(self, names: tuple[str, ...]) -> None:
        for name in names:
            with self._lock:
                if name in self._ready or name in self._downloading:
                    continue
                self._downloading.add(name)
            try:
                root = checkpoints.fetch(
                    self._settings.repo,
                    self._settings.revision,
                    name,
                    allow_download=True,
                    cache_dir=self._cache_dir,
                )
                checkpoints.verify(root, name)
                with self._lock:
                    self._ready.add(name)
            finally:
                with self._lock:
                    self._downloading.discard(name)

    def is_loaded(self) -> bool:
        return bool(self._router.loaded)

    def unload(self) -> None:
        self._router.unload()

    def status(self) -> dict[str, Any]:
        with self._lock:
            ready, downloading = sorted(self._ready), sorted(self._downloading)
        return {
            "backend": self.name,
            "repo": self._settings.repo,
            "revision": self._settings.revision,
            "dtype": self._settings.dtype,
            "loaded": list(self._router.loaded),
            "ready": ready,
            "downloading": downloading,
            "cache": str(self._root),
        }

    def _pick(
        self, state: Any, questions: dict[str, Any] | None, model: str
    ) -> tuple[str, str | None]:
        if model == "auto":
            decision = self._router.route(state, questions)
            return decision["model"], decision.get("reason")
        try:
            return self._router.resolve(model), "requested"
        except ValueError as exc:
            raise ValueError(f"unknown model {model!r}; {UNKNOWN_MODEL_HINT}") from exc

    def _require(self, name: str) -> None:
        with self._lock:
            if name in self._ready:
                return
            downloading = name in self._downloading
        if downloading:
            raise ModelUnavailable(
                f"the {name} checkpoint is still being downloaded or verified", "retry in a minute"
            )
        root = checkpoints.fetch(
            self._settings.repo,
            self._settings.revision,
            name,
            allow_download=False,
            cache_dir=self._cache_dir,
        )
        checkpoints.verify(root, name)
        with self._lock:
            self._ready.add(name)

    def _predict(
        self, router: Any, state: Any, questions: dict[str, dict[str, Any]], name: str
    ) -> dict[str, Any]:
        if any(
            q["type"] == "choice" and len(q["criteria"]) > SHORTLIST_K for q in questions.values()
        ):
            embed = self._embed_factory(router.load(name))
            return self._shortlist(router, state, questions, embed, k=SHORTLIST_K, model=name)
        return router.predict(state, questions, model=name)

    def _predict_float32(
        self, state: Any, questions: dict[str, dict[str, Any]], name: str
    ) -> dict[str, Any]:
        router = self._router_factory(self._root, "float32")
        try:
            return self._predict(router, state, questions, name)
        except FloatingPointError as exc:
            raise ModelUnavailable(
                "the model produced non-finite outputs, even in float32",
                "rephrase or shorten the input",
            ) from exc
        finally:
            router.unload()
