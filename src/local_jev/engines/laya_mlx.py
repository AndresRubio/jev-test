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
            usage = _with_round_warnings(raw.get("usage", {}), recorder.rounds, questions)
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
    """The Router as predict_tournament sees it: every call is forwarded, and the rounds are kept.

    The next elimination round is a call whose questions are exactly that round's groups, "0",
    "1", ... in order. Any other call is the final one, which uses the caller's ids. Each round is
    kept with its usage and the caller's question that owns each group. A ValueError from a round
    that names a group is raised again naming the caller's question.
    """

    def __init__(self, router: Any, questions: Questions) -> None:
        self._router = router
        self._owners = _round_owners(questions)
        self.rounds: list[tuple[Questions, dict[str, Any], list[str]]] = []

    def predict(self, state: Any, questions: Questions, **kwargs: Any) -> dict[str, Any]:
        owners = self._next_round(questions)
        try:
            result = self._router.predict(state, questions, **kwargs)
        except ValueError as exc:
            renamed = None if owners is None else _renamed(exc, owners)
            if renamed is None:
                raise
            raise renamed from exc
        if owners is not None:
            self.rounds.append((questions, result.get("usage") or {}, owners))
        return result

    def _next_round(self, questions: Questions) -> list[str] | None:
        """The owners of the call's groups when the call is the next round, else None."""
        if len(self.rounds) == len(self._owners):
            return None
        owners = self._owners[len(self.rounds)]
        return owners if list(questions) == [str(i) for i in range(len(owners))] else None


def _needs_tournament(question: dict[str, Any]) -> bool:
    return question["type"] == "choice" and len(question["criteria"]) > SHORTLIST_K


def _round_owners(questions: Questions) -> list[list[str]]:
    """For each elimination round, the caller's question that each of its groups was cut from.

    owners[r][i] owns group str(i) of round r. This is how laya-mlx 0.4.0's predict_tournament
    numbers them: a round's groups follow the order of the request's choices, with each choice's
    groups next to each other. A choice of n labels gets ceil(n / SHORTLIST_K) groups while that
    is more than one, and each group sends one label on to the next round. Two choices with the
    same instructions can have groups with the same labels, so only the position of a group tells
    which choice it was cut from.
    """
    left = {qid: len(q["criteria"]) for qid, q in questions.items() if q["type"] == "choice"}
    rounds: list[list[str]] = []
    while True:
        groups = {qid: -(-n // SHORTLIST_K) for qid, n in left.items()}
        owners = [qid for qid, count in groups.items() if count > 1 for _ in range(count)]
        if not owners:
            return rounds
        rounds.append(owners)
        left = {qid: count for qid, count in groups.items() if count > 1}


def _with_round_warnings(
    usage: dict[str, Any],
    rounds: list[tuple[Questions, dict[str, Any], list[str]]],
    questions: Questions,
) -> dict[str, Any]:
    """The final call's usage, plus the truncation and collapsed options of the rounds before it.

    Token counts stay the final call's. Each group's warning goes to the caller's question that
    owns the group.
    """
    merged = dict(usage)
    round_usages = [round_usage for _, round_usage, _ in rounds]
    if any(round_usage.get("truncated") for round_usage in round_usages):
        truncating = [u for u in [*round_usages, usage] if u.get("truncated")]
        cut = set(usage.get("truncated_questions") or ())
        for _call, round_usage, owners in rounds:
            for gid in round_usage.get("truncated_questions") or ():
                cut.add(owners[int(gid)])
        merged.update(
            truncated=True,
            truncated_questions=[qid for qid in questions if qid in cut],
            state_tokens=max(u.get("state_tokens", 0) for u in truncating),
            state_tokens_dropped=max(u.get("state_tokens_dropped", 0) for u in truncating),
        )
    collapsed = _round_collapses(rounds)
    if collapsed:
        merged["options"] = {**(usage.get("options") or {}), **collapsed}
    return merged


def _round_collapses(
    rounds: list[tuple[Questions, dict[str, Any], list[str]]],
) -> dict[str, dict[str, int]]:
    """Each caller question whose groups collapsed, summed over all its groups in that round.

    A group that did not collapse counts all of its options as distinct, so the sums cover every
    option the round saw. The first round that collapsed a question wins, since it saw the most.
    """
    found: dict[str, dict[str, int]] = {}
    for call, round_usage, round_owners in rounds:
        options = round_usage.get("options") or {}
        hit = {round_owners[int(gid)] for gid in options}
        for qid in dict.fromkeys(round_owners):
            if qid not in hit or qid in found:
                continue
            groups = [str(i) for i, owner in enumerate(round_owners) if owner == qid]
            found[qid] = {
                key: sum(
                    options.get(gid, {}).get(key, len(call[gid]["criteria"])) for gid in groups
                )
                for key in ("total", "distinct")
            }
    return found


def _renamed(error: ValueError, owners: list[str]) -> ValueError | None:
    """A round's error naming the caller's question in place of a group, or None if it names none.

    laya-mlx starts a question's error with "Question <id>", and a round's questions are its groups.
    """
    message = str(error)
    for i, qid in enumerate(owners):
        prefix = f"Question {str(i)!r}"
        if message.startswith(prefix):
            return ValueError(
                f"Question {qid!r}{message[len(prefix) :]} "
                f"(it was narrowed in groups of {SHORTLIST_K})"
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
