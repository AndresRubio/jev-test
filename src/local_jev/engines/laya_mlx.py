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
LOAD_HINT = f"{checkpoints.PULL_HINT}, or check that laya-mlx is 0.4.0"

Questions = dict[str, dict[str, Any]]


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
        tournament_fn: Callable[..., dict[str, Any]] | None = None,
        tokenizer_factory: Callable[[Path], Any] = default_tokenizer_factory,
    ) -> None:
        self._settings = settings
        self._cache_dir = cache_dir
        self._root = checkpoints.snapshot_root(settings.repo, settings.revision, cache_dir)
        self._router_factory = router_factory
        self._tournament = tournament_fn or (
            lambda *args, **kwargs: _laya().predict_tournament(*args, **kwargs)
        )
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
            shortlist=_shortlist(raw.get("tournament", {})),
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
        # Cached count_tokens tokenizers count too, so the idle unload frees them. Reading the
        # dict's size needs no lock.
        return bool(self._router.loaded) or bool(self._tokenizers)

    def unload(self) -> None:
        """Free the checkpoints and the tokenizers count_tokens cached; each is rebuilt on next use."""
        self._router.unload()
        with self._lock:
            self._tokenizers.clear()

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
        self._load(router, name)
        if any(_needs_tournament(question) for question in questions.values()):
            # The group size equals the threshold, so a choice of 20 or fewer options in the same
            # request reaches the final call whole.
            recorder = _RoundRecorder(router, questions)
            raw = self._tournament(recorder, state, questions, group_size=SHORTLIST_K, model=name)
            usage = _with_round_warnings(raw.get("usage", {}), recorder.calls[:-1], questions)
            return {**raw, "usage": usage}
        return router.predict(state, questions, model=name)

    @staticmethod
    def _load(router: Any, name: str) -> None:
        """Build the checkpoint's Agent before predicting, so a failure to build it is a 503.

        laya-mlx raises a plain ValueError (a config it rejects) or OSError (a missing file) when
        it cannot build an Agent. Left alone, those reach the service as a bad request and the
        caller is told their questions are wrong. The Router keeps the Agent it built, so the
        predict() calls that follow reuse it.
        """
        try:
            router.load(name)
        except (ValueError, OSError) as exc:
            raise ModelUnavailable(
                f"the {name} checkpoint failed to load: {exc}", LOAD_HINT
            ) from exc

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


class _RoundRecorder:
    """The Router as predict_tournament sees it: every call is forwarded and kept.

    Every call but the last is an elimination round, whose questions carry the tournament's own
    ids ("0", "1", ...). A ValueError that names one of them is raised again naming the caller's
    question.
    """

    def __init__(self, router: Any, questions: Questions) -> None:
        self._router = router
        self._questions = questions
        self.calls: list[tuple[Questions, dict[str, Any]]] = []

    def predict(self, state: Any, questions: Questions, **kwargs: Any) -> dict[str, Any]:
        try:
            result = self._router.predict(state, questions, **kwargs)
        except ValueError as exc:
            renamed = _renamed(exc, questions, self._questions)
            if renamed is None:
                raise
            raise renamed from exc
        self.calls.append((questions, result.get("usage") or {}))
        return result


def _needs_tournament(question: dict[str, Any]) -> bool:
    return question["type"] == "choice" and len(question["criteria"]) > SHORTLIST_K


def _sources(group: dict[str, Any], questions: Questions) -> list[str]:
    """The caller's questions that a tournament question could have been cut from.

    predict_tournament cuts a narrowed choice's criteria to a subset and keeps its other fields,
    so a source needs the tournament, has the same other fields and has every label of the group.
    Identical sources cannot be told apart, so all of them are returned. Only choices are cut,
    and only their labels are sure to be strings (a score level may be an object).
    """
    if group.get("type") != "choice":
        return []
    labels = set(group["criteria"])
    fields = _without_criteria(group)
    return [
        qid
        for qid, question in questions.items()
        if _needs_tournament(question)
        and _without_criteria(question) == fields
        and labels <= set(question["criteria"])
    ]


def _without_criteria(question: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in question.items() if key != "criteria"}


def _with_round_warnings(
    usage: dict[str, Any], rounds: list[tuple[Questions, dict[str, Any]]], questions: Questions
) -> dict[str, Any]:
    """The final call's usage, plus the truncation and collapsed options of the rounds before it.

    Token counts stay the final call's. A round names its groups, so each group's warning goes to
    the caller's question the group was cut from.
    """
    merged = dict(usage)
    round_usages = [round_usage for _, round_usage in rounds]
    if any(round_usage.get("truncated") for round_usage in round_usages):
        truncating = [u for u in [*round_usages, usage] if u.get("truncated")]
        cut = set(usage.get("truncated_questions") or ())
        for call, round_usage in rounds:
            for gid in round_usage.get("truncated_questions") or ():
                cut.update(_sources(call[gid], questions))
        merged.update(
            truncated=True,
            truncated_questions=[qid for qid in questions if qid in cut],
            state_tokens=max(u.get("state_tokens", 0) for u in truncating),
            state_tokens_dropped=max(u.get("state_tokens_dropped", 0) for u in truncating),
        )
    collapsed = _round_collapses(rounds, questions)
    if collapsed:
        merged["options"] = {**(usage.get("options") or {}), **collapsed}
    return merged


def _round_collapses(
    rounds: list[tuple[Questions, dict[str, Any]]], questions: Questions
) -> dict[str, dict[str, int]]:
    """Each caller question whose groups collapsed, summed over all its groups in that round.

    A group that did not collapse counts all of its options as distinct, so the sums cover every
    option the round saw. The first round that collapsed a question wins, since it saw the most.
    """
    found: dict[str, dict[str, int]] = {}
    for call, round_usage in rounds:
        options = round_usage.get("options") or {}
        if not options:
            continue
        sources = {gid: _sources(group, questions) for gid, group in call.items()}
        hit = {qid for gid in options for qid in sources[gid]}
        for qid in questions:
            if qid not in hit or qid in found:
                continue
            total = distinct = 0
            for gid, group in call.items():
                if qid in sources[gid]:
                    entry = options.get(gid, {})
                    total += entry.get("total", len(group["criteria"]))
                    distinct += entry.get("distinct", len(group["criteria"]))
            found[qid] = {"total": total, "distinct": distinct}
    return found


def _renamed(error: ValueError, call: Questions, questions: Questions) -> ValueError | None:
    """The error naming the caller's question when it names a tournament group, else None.

    laya-mlx starts a question's error with "Question <id>". In the final call that id is the
    caller's own: the question is either unchanged or cut from the caller's question of that id.
    """
    message = str(error)
    for gid, group in call.items():
        prefix = f"Question {gid!r}"
        if not message.startswith(prefix):
            continue
        sources = _sources(group, questions)
        if not sources or gid in sources or group == questions.get(gid):
            return None
        named = " or ".join(repr(qid) for qid in sources)
        return ValueError(
            f"Question {named}{message[len(prefix) :]} (it was narrowed in groups of {SHORTLIST_K})"
        )
    return None


def _shortlist(tournament: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The choices the tournament narrowed, in the shortlist shape guard.finalize reads.

    A choice that needed no round (rounds == 0) went to the final call whole, so it is left out.
    """
    return {
        qid: {
            "labels": list(entry["labels"]),
            "n": entry["n"],
            "passthrough": False,
            "rounds": entry["rounds"],
        }
        for qid, entry in tournament.items()
        if entry["rounds"] > 0
    }
