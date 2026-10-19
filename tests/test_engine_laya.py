import pytest

from local_jev.config import LayaSettings
from local_jev.engines import laya_mlx as engine_module
from local_jev.engines.laya_mlx import LayaMLXEngine
from local_jev.errors import ModelUnavailable, NotDownloaded

NOUL = {"q": {"type": "noul", "instructions": "Does the customer ask for money back?"}}


class FakeRouter:
    """The parts of laya_mlx.Router the engine uses."""

    def __init__(self, root, dtype, fail_always=None):
        self.root = root
        self.dtype = dtype
        self.fail_always = fail_always
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
        if self.fail_always is not None:
            raise self.fail_always
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error
        self.loaded = [model]
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
        self.loaded = [name]
        return f"agent:{name}"

    def unload(self, name=None):
        self.loaded = []


def make_engine(monkeypatch, *, fail_dtypes=(), **kwargs):
    routers: list[FakeRouter] = []
    verified: list[str] = []

    def router_factory(root, dtype):
        router = FakeRouter(
            root, dtype, FloatingPointError("nan") if dtype in fail_dtypes else None
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


def test_more_than_20_options_are_shortlisted(monkeypatch):
    calls = []

    def fake_shortlist(router, state, questions, embed, *, k, model):
        calls.append((k, model, embed))
        result = router.predict(state, questions, model=model)
        result["shortlist"] = {
            "q": {"labels": ["o0"], "scores": [0.9], "k": k, "n": 25, "passthrough": False}
        }
        return result

    engine, _, _ = make_engine(
        monkeypatch, shortlist_fn=fake_shortlist, embed_factory=lambda agent: f"embed:{agent}"
    )
    many = {
        "q": {"type": "choice", "instructions": "Which?", "criteria": [f"o{i}" for i in range(25)]}
    }
    result = engine.decide("x", many, "auto")
    assert calls == [(20, "english", "embed:agent:english")]
    assert result.shortlist["q"]["n"] == 25


def test_twenty_options_are_not_shortlisted(monkeypatch):
    engine, _, _ = make_engine(
        monkeypatch, shortlist_fn=lambda *a, **k: pytest.fail("must not shortlist")
    )
    twenty = {
        "q": {"type": "choice", "instructions": "Which?", "criteria": [f"o{i}" for i in range(20)]}
    }
    assert engine.decide("x", twenty, "auto").shortlist == {}


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

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return type("Encoding", (), {"ids": text.split()})()

    def tokenizer_factory(path):
        paths.append(path)
        return FakeTokenizer()

    engine, _, _ = make_engine(monkeypatch, tokenizer_factory=tokenizer_factory)
    assert engine.count_tokens("three short words", "auto") == 3
    assert engine.count_tokens("again two", "auto") == 2
    assert len(paths) == 1
    assert paths[0].name == "tokenizer.json"


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


def test_the_default_shortlist_and_embedding_come_from_laya_mlx(monkeypatch):
    calls = []

    class FakeLaya:
        @staticmethod
        def embed_fn_from_agent(agent):
            return f"embed:{agent}"

        @staticmethod
        def predict_shortlist(router, state, questions, embed_fn, k=20, **predict_kwargs):
            calls.append((embed_fn, k, predict_kwargs))
            return router.predict(state, questions, **predict_kwargs)

    monkeypatch.setattr(engine_module, "_laya", lambda: FakeLaya)
    engine, _, _ = make_engine(monkeypatch)
    many = {
        "q": {"type": "choice", "instructions": "Which?", "criteria": [f"o{i}" for i in range(25)]}
    }
    engine.decide("x", many, "auto")
    assert calls == [("embed:agent:english", 20, {"model": "english"})]


def test_the_default_tokenizer_factory_loads_a_tokenizer_json(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "refund": 1, "me": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    loaded = engine_module.default_tokenizer_factory(path)
    assert loaded.encode("refund me refund", add_special_tokens=False).ids == [1, 2, 1]
