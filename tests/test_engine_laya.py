import weakref

import pytest

from local_jev.config import LayaSettings
from local_jev.engines import laya_mlx as engine_module
from local_jev.engines.laya_mlx import LayaMLXEngine
from local_jev.errors import ModelUnavailable, NotDownloaded

NOUL = {"q": {"type": "noul", "instructions": "Does the customer ask for money back?"}}


def choice(count):
    return {"type": "choice", "instructions": "Which?", "criteria": [f"o{i}" for i in range(count)]}


class FakeRouter:
    """The parts of laya_mlx.Router the engine uses."""

    def __init__(self, root, dtype, fail_always=None, load_error=None):
        self.root = root
        self.dtype = dtype
        self.fail_always = fail_always
        self.load_error = load_error
        self.fail_next = None
        self.loaded: list[str] = []
        self.predictions: list[tuple] = []

    def route(self, state, questions=None, model=None, task=None, lang=None):
        spanish = any(ch in str(state) for ch in "¿ñáéíóú")
        return {
            "model": "multilingual" if spanish else "english",
            "reason": "fake script detection",
        }

    def resolve(self, name):
        name = {"en": "english", "laya": "english", "multi": "multilingual"}.get(name, name)
        if name not in ("english", "multilingual", "typed-decisions"):
            raise ValueError(f"unknown model {name!r}")
        return name

    def predict(self, state, questions, model=None):
        self.load(model)  # laya-mlx's Router.predict builds the Agent first, when it is missing
        if self.fail_always is not None:
            raise self.fail_always
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error
        self.predictions.append((state, questions, model))
        answers = {
            qid: {"type": "noul", "noul": 0.9, "confidence": 0.9, "answer_confidence": 0.9}
            for qid in questions
        }
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": 12, "output_tokens": 0},
            "routing": {"model": model},
        }

    def load(self, name):
        if self.load_error is not None:
            raise self.load_error
        self.loaded = [name]
        return f"agent:{name}"

    def unload(self, name=None):
        self.loaded = []


class FakeTournament:
    """Stands in for laya_mlx.predict_tournament: a choice above group_size keeps its first two labels.

    group_size defaults to 16 as in laya-mlx, so a call without group_size=20 also narrows a
    choice of 17 to 20 options.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, router, state, questions, group_size=16, **predict_kwargs):
        self.calls.append(
            {"router": router, "state": state, "questions": questions, "group_size": group_size}
            | predict_kwargs
        )
        tournament, final = {}, dict(questions)
        for qid, question in questions.items():
            if question["type"] != "choice":
                continue
            labels, rounds = list(question["criteria"]), 0
            if len(labels) > group_size:
                labels, rounds = labels[:2], 1
                final[qid] = dict(question, criteria=labels)
            tournament[qid] = {"labels": labels, "n": len(question["criteria"]), "rounds": rounds}
        result = router.predict(state, final, **predict_kwargs)
        result["tournament"] = tournament
        return result


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return type("Encoding", (), {"ids": text.split()})()


def make_engine(monkeypatch, *, fail_dtypes=(), load_errors=None, **kwargs):
    """load_errors maps a dtype to the exception that router's load() raises."""
    routers: list[FakeRouter] = []
    verified: list[str] = []

    def router_factory(root, dtype):
        router = FakeRouter(
            root,
            dtype,
            FloatingPointError("nan") if dtype in fail_dtypes else None,
            (load_errors or {}).get(dtype),
        )
        routers.append(router)
        return router

    monkeypatch.setattr(
        engine_module.checkpoints,
        "fetch",
        lambda repo, revision, name, *, allow_download, cache_dir=None: f"/fake/{name}",
    )
    monkeypatch.setattr(
        engine_module.checkpoints, "verify", lambda root, name: verified.append(name)
    )
    engine = LayaMLXEngine(LayaSettings(), router_factory=router_factory, **kwargs)
    return engine, routers, verified


def test_auto_routes_english_and_reports_the_checkpoint(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch)
    result = engine.decide("Please refund me", NOUL, "auto")
    assert result.model == "laya-english"
    assert result.checkpoint == "english"
    assert result.routing_reason == "fake script detection"
    assert routers[0].predictions[-1][2] == "english"
    assert result.answers["q"]["noul"] == 0.9


def test_auto_routes_spanish_to_multilingual(monkeypatch):
    engine, _, _ = make_engine(monkeypatch)
    assert engine.decide("¿Me devuelven el dinero?", NOUL, "auto").checkpoint == "multilingual"


def test_an_explicit_model_or_alias_skips_routing(monkeypatch):
    engine, _, _ = make_engine(monkeypatch)
    assert engine.decide("x", NOUL, "multilingual").routing_reason == "requested"
    assert engine.decide("x", NOUL, "laya").checkpoint == "english"


def test_an_unknown_model_is_a_value_error(monkeypatch):
    engine, _, _ = make_engine(monkeypatch)
    with pytest.raises(ValueError, match="unknown model 'gpt'"):
        engine.decide("x", NOUL, "gpt")


def test_a_checkpoint_is_verified_once(monkeypatch):
    engine, _, verified = make_engine(monkeypatch)
    engine.decide("a", NOUL, "auto")
    engine.decide("b", NOUL, "auto")
    assert verified == ["english"]


def test_a_missing_checkpoint_points_to_pull(monkeypatch):
    engine, _, _ = make_engine(monkeypatch)

    def missing(*args, **kwargs):
        raise NotDownloaded("the english checkpoint is not downloaded", "run `local-jev pull`")

    monkeypatch.setattr(engine_module.checkpoints, "fetch", missing)
    with pytest.raises(ModelUnavailable) as info:
        engine.decide("x", NOUL, "auto")
    assert "local-jev pull" in info.value.hint


def test_a_checkpoint_still_downloading_asks_to_retry(monkeypatch):
    engine, _, _ = make_engine(monkeypatch)
    engine._downloading.add("english")
    with pytest.raises(ModelUnavailable, match="still being downloaded"):
        engine.decide("x", NOUL, "auto")


def test_twenty_options_take_the_direct_path(monkeypatch):
    engine, routers, _ = make_engine(
        monkeypatch, tournament_fn=lambda *a, **k: pytest.fail("must not run a tournament")
    )
    twenty = {"q": choice(20)}
    result = engine.decide("x", twenty, "auto")
    assert routers[0].predictions == [("x", twenty, "english")]
    assert result.shortlist == {}


def test_more_than_20_options_go_through_the_tournament(monkeypatch):
    tournament = FakeTournament()
    engine, routers, _ = make_engine(monkeypatch, tournament_fn=tournament)
    questions = {"q": choice(21)}
    result = engine.decide("x", questions, "auto")
    assert tournament.calls == [
        {
            "router": routers[0],
            "state": "x",
            "questions": questions,
            "group_size": 20,
            "model": "english",
        }
    ]
    assert result.shortlist == {
        "q": {"labels": ["o0", "o1"], "n": 21, "passthrough": False, "rounds": 1}
    }


def test_only_choices_the_tournament_narrowed_are_shortlisted(monkeypatch):
    def tournament(router, state, questions, *, group_size, model):
        result = router.predict(state, questions, model=model)
        result["tournament"] = {
            "big": {"labels": ["o4", "o29"], "n": 30, "rounds": 1},
            "small": {"labels": [f"o{i}" for i in range(17)], "n": 17, "rounds": 0},
        }
        return result

    engine, _, _ = make_engine(monkeypatch, tournament_fn=tournament)
    result = engine.decide("x", {"big": choice(30), "small": choice(17)}, "auto")
    assert result.shortlist == {
        "big": {"labels": ["o4", "o29"], "n": 30, "passthrough": False, "rounds": 1}
    }
    assert set(result.answers) == {"big", "small"}
    assert result.usage == {"input_tokens": 12, "output_tokens": 0}


def test_a_mixed_request_narrows_only_the_choice_above_20(monkeypatch):
    tournament = FakeTournament()
    engine, _, _ = make_engine(monkeypatch, tournament_fn=tournament)
    questions = {"big": choice(30), "small": choice(17), **NOUL}
    result = engine.decide("x", questions, "auto")
    assert len(tournament.calls) == 1
    assert set(result.shortlist) == {"big"}
    assert set(result.answers) == {"big", "small", "q"}


def test_non_finite_outputs_are_retried_in_float32(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch)
    routers[0].fail_next = FloatingPointError("nan")
    result = engine.decide("x", NOUL, "auto")
    assert result.checkpoint == "english"
    assert [router.dtype for router in routers] == ["float16", "float32"]
    assert routers[1].loaded == []


def test_float32_failure_is_model_unavailable(monkeypatch):
    engine, _, _ = make_engine(monkeypatch, fail_dtypes=("float16", "float32"))
    with pytest.raises(ModelUnavailable, match="non-finite"):
        engine.decide("x", NOUL, "auto")


def test_the_float32_retry_also_runs_the_tournament(monkeypatch):
    tournament = FakeTournament()
    engine, routers, _ = make_engine(monkeypatch, tournament_fn=tournament)
    routers[0].fail_next = FloatingPointError("nan")
    result = engine.decide("x", {"q": choice(25)}, "auto")
    assert [router.dtype for router in routers] == ["float16", "float32"]
    assert [call["router"] for call in tournament.calls] == routers
    assert [call["group_size"] for call in tournament.calls] == [20, 20]
    assert result.shortlist["q"]["n"] == 25
    assert routers[1].loaded == []


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Unsupported encoder activation: 'silu'"),
        FileNotFoundError(2, "No such file or directory", "rl_agent_config.json"),
    ],
    ids=["value-error", "os-error"],
)
def test_a_checkpoint_that_fails_to_load_is_model_unavailable(monkeypatch, error):
    engine, routers, _ = make_engine(monkeypatch, load_errors={"float16": error})
    with pytest.raises(ModelUnavailable) as info:
        engine.decide("x", NOUL, "auto")
    assert info.value.message.startswith("the english checkpoint failed to load: ")
    assert str(error) in info.value.message
    assert "local-jev pull" in info.value.hint
    assert "laya-mlx is 0.4.0" in info.value.hint
    assert info.value.__cause__ is error
    assert routers[0].predictions == []
    assert engine.is_loaded() is False


def test_a_load_failure_in_the_tournament_path_is_model_unavailable(monkeypatch):
    engine, _, _ = make_engine(
        monkeypatch,
        load_errors={"float16": ValueError("bad config")},
        tournament_fn=lambda *a, **k: pytest.fail("must not run a tournament"),
    )
    with pytest.raises(ModelUnavailable, match="failed to load"):
        engine.decide("x", {"q": choice(25)}, "auto")


def test_the_float32_retry_that_fails_to_load_is_model_unavailable(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch, load_errors={"float32": ValueError("bad config")})
    routers[0].fail_next = FloatingPointError("nan")
    with pytest.raises(ModelUnavailable, match="failed to load"):
        engine.decide("x", NOUL, "auto")
    assert [router.dtype for router in routers] == ["float16", "float32"]
    assert routers[1].loaded == []


def test_a_failed_load_is_tried_again_on_the_next_request(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch, load_errors={"float16": ValueError("bad config")})
    with pytest.raises(ModelUnavailable):
        engine.decide("x", NOUL, "auto")
    routers[0].load_error = None
    assert engine.decide("x", NOUL, "auto").checkpoint == "english"


def test_a_token_budget_error_from_predict_stays_a_value_error(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch)
    routers[0].fail_next = ValueError("Question 'q' has too many options for the token budget")
    with pytest.raises(ValueError, match="token budget"):
        engine.decide("x", NOUL, "auto")


def test_prefetch_marks_checkpoints_ready(monkeypatch):
    engine, _, verified = make_engine(monkeypatch)
    engine.prefetch(("english", "multilingual"))
    status = engine.status()
    assert status["ready"] == ["english", "multilingual"]
    assert status["downloading"] == []
    assert verified == ["english", "multilingual"]


def test_unload_and_is_loaded(monkeypatch):
    engine, _, _ = make_engine(monkeypatch)
    assert engine.is_loaded() is False
    engine.decide("x", NOUL, "auto")
    assert engine.is_loaded() is True
    engine.unload()
    assert engine.is_loaded() is False


def test_count_tokens_uses_the_checkpoint_tokenizer(monkeypatch):
    paths = []

    def tokenizer_factory(path):
        paths.append(path)
        return FakeTokenizer()

    engine, _, _ = make_engine(monkeypatch, tokenizer_factory=tokenizer_factory)
    assert engine.count_tokens("three short words", "auto") == 3
    assert engine.count_tokens("again two", "auto") == 2
    assert len(paths) == 1
    assert paths[0].name == "tokenizer.json"


def test_unload_frees_the_cached_tokenizers(monkeypatch):
    built = []

    def tokenizer_factory(path):
        tokenizer = FakeTokenizer()
        built.append(weakref.ref(tokenizer))
        return tokenizer

    engine, _, _ = make_engine(monkeypatch, tokenizer_factory=tokenizer_factory)
    engine.count_tokens("one", "english")
    engine.count_tokens("uno", "multilingual")
    engine.unload()
    assert [ref() for ref in built] == [None, None]
    assert engine.count_tokens("one two", "english") == 2
    assert len(built) == 3


def test_status_reports_the_pinned_revision(monkeypatch):
    engine, _, _ = make_engine(monkeypatch)
    status = engine.status()
    assert status["backend"] == "laya-mlx"
    assert status["revision"] == LayaSettings().revision
    assert status["loaded"] == []


def test_the_default_router_points_every_checkpoint_at_the_pinned_snapshot(monkeypatch, tmp_path):
    seen = {}

    class FakeLaya:
        class Router:
            def __init__(self, **kwargs):
                seen.update(kwargs)

    monkeypatch.setattr(engine_module, "_laya", lambda: FakeLaya)
    engine_module.default_router_factory(tmp_path, "float32")
    assert seen == {
        "models": {
            "english": str(tmp_path),
            "multilingual": (str(tmp_path), "multilingual"),
            "typed-decisions": (str(tmp_path), "typed-decisions"),
        },
        "max_loaded": 2,
        "default": "english",
        "dtype": "float32",
    }


def test_the_default_tournament_comes_from_laya_mlx(monkeypatch):
    calls = []

    class FakeLaya:
        @staticmethod
        def predict_tournament(agent, state, questions, group_size=16, **predict_kwargs):
            calls.append((agent, group_size, predict_kwargs))
            return agent.predict(state, questions, **predict_kwargs)

    monkeypatch.setattr(engine_module, "_laya", lambda: FakeLaya)
    engine, routers, _ = make_engine(monkeypatch)
    engine.decide("x", {"q": choice(25)}, "auto")
    assert calls == [(routers[0], 20, {"model": "english"})]


def test_the_default_tokenizer_factory_loads_a_tokenizer_json(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "refund": 1, "me": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    loaded = engine_module.default_tokenizer_factory(path)
    assert loaded.encode("refund me refund", add_special_tokens=False).ids == [1, 2, 1]
