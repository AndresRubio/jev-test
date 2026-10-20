import weakref

import pytest

from local_jev.config import LayaSettings
from local_jev.engines import laya_mlx as engine_module
from local_jev.engines.laya_mlx import LayaMLXEngine
from local_jev.errors import ModelUnavailable, NotDownloaded
from local_jev.guard import finalize

NOUL = {"q": {"type": "noul", "instructions": "Does the customer ask for money back?"}}


def choice(count, instructions="Which?"):
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": [f"o{i}" for i in range(count)],
    }


def fake_answer(question):
    """A choice picks its last label, so a tournament group of o0..o14 sends o14 on."""
    if question["type"] == "choice":
        labels = list(question["criteria"])
        return {
            "type": "choice",
            "choice": labels[-1],
            "probabilities": {label: float(label == labels[-1]) for label in labels},
            "confidence": 0.9,
            "answer_confidence": 0.9,
        }
    return {"type": "noul", "noul": 0.9, "confidence": 0.9, "answer_confidence": 0.9}


def laya_usage(truncates=lambda question: 0, collapses=lambda question: 0):
    """A FakeRouter usage_for that reports in laya-mlx's shape.

    truncates(question) is how many state tokens that question loses, and collapses(question) is
    how many of its options lose their own token span. input_tokens is 100 per question.
    """

    def usage_for(questions):
        dropped = {qid: truncates(question) for qid, question in questions.items()}
        usage = {
            "input_tokens": 100 * len(questions),
            "state_tokens": 400,
            "state_tokens_dropped": max(dropped.values()),
            "truncated": any(dropped.values()),
            "truncated_questions": [qid for qid, count in dropped.items() if count],
        }
        options = {
            qid: {
                "total": len(question["criteria"]),
                "distinct": len(question["criteria"]) - collapses(question),
                "tokens_per_option": 4,
            }
            for qid, question in questions.items()
            if collapses(question)
        }
        if options:
            usage["options"] = options
        return usage

    return usage_for


class FakeRouter:
    """The parts of laya_mlx.Router the engine uses.

    usage_for(questions) adds fields to each call's usage, and a question for which
    too_many_options(question) is true fails the way laya-mlx reports an option budget overflow.
    """

    def __init__(self, root, dtype, fail_always=None, load_error=None):
        self.root = root
        self.dtype = dtype
        self.fail_always = fail_always
        self.load_error = load_error
        self.fail_next = None
        self.usage_for = lambda questions: {}
        self.too_many_options = lambda question: False
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
        for qid, question in questions.items():
            if self.too_many_options(question):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
        self.predictions.append((state, questions, model))
        return {
            "model": "laya-rl-agent",
            "answers": {qid: fake_answer(question) for qid, question in questions.items()},
            "usage": {"input_tokens": 12, "output_tokens": 0, **self.usage_for(questions)},
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
            {"state": state, "questions": questions, "group_size": group_size} | predict_kwargs
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
        {"state": "x", "questions": questions, "group_size": 20, "model": "english"}
    ]
    assert [(state, model) for state, _, model in routers[0].predictions] == [("x", "english")]
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


# The next tests run the real laya_mlx.predict_tournament over FakeRouter, so the round
# questions carry laya-mlx's own ids ("0", "1", ...). No weights are loaded.


def test_a_round_that_cut_the_state_warns_on_the_narrowed_question(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch)
    # A group of 15 options leaves less room for the state than the 2 finalists do.
    routers[0].usage_for = laya_usage(
        truncates=lambda question: 60 if len(question.get("criteria", ())) > 10 else 0
    )
    questions = {"dept": choice(30), **NOUL}
    result = engine.decide("x", questions, "auto")
    assert result.usage == {
        "input_tokens": 200,
        "output_tokens": 0,
        "state_tokens": 400,
        "state_tokens_dropped": 60,
        "truncated": True,
        "truncated_questions": ["dept"],
    }
    answers = finalize(result, questions, backend="laya-mlx", latency_ms=0.0)["answers"]
    assert [warning.split(":")[0] for warning in answers["dept"]["warnings"]] == [
        "state_truncated",
        "shortlisted",
    ]
    assert answers["q"]["warnings"] == []


def test_only_the_narrowed_question_whose_rounds_were_cut_is_marked(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch)

    def truncates(question):
        if question["type"] == "noul":
            return 10
        if question["instructions"] == "Which topic?" and len(question["criteria"]) > 10:
            return 75
        return 0

    routers[0].usage_for = laya_usage(truncates=truncates)
    questions = {"topic": choice(30, "Which topic?"), "refund": NOUL["q"], "team": choice(25)}
    result = engine.decide("x", questions, "auto")
    # The round cut both groups of "topic" by 75 tokens and neither group of "team"; the final
    # call cut only "refund", by 10. The list keeps the request's order.
    assert result.usage["truncated_questions"] == ["topic", "refund"]
    assert result.usage["state_tokens_dropped"] == 75
    assert result.usage["input_tokens"] == 300


def test_options_that_collapse_in_a_round_are_summed_over_its_groups(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch)

    def collapses(question):
        labels = set(question["criteria"])
        return 2 * ("o3" in labels) + ({"o14", "o29"} <= labels)

    routers[0].usage_for = laya_usage(collapses=collapses)
    result = engine.decide("x", {"big": choice(30), "small": choice(17)}, "auto")
    # Round groups o0..o14 (2 collapse) and o15..o29 (none) cover all 30 options of "big", so
    # their sum replaces the final call's figure for its finalists o14 and o29. "small" never
    # went through a round and keeps the final call's entry.
    assert result.usage["options"] == {
        "big": {"total": 30, "distinct": 28},
        "small": {"total": 17, "distinct": 15, "tokens_per_option": 4},
    }


def test_warnings_from_the_final_call_alone_are_kept_as_reported(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch)
    routers[0].usage_for = laya_usage(
        truncates=lambda question: 30 if len(question["criteria"]) == 17 else 0,
        collapses=lambda question: 2 if len(question["criteria"]) == 17 else 0,
    )
    result = engine.decide("x", {"big": choice(30), "small": choice(17)}, "auto")
    assert result.usage == {
        "input_tokens": 200,
        "output_tokens": 0,
        "state_tokens": 400,
        "state_tokens_dropped": 30,
        "truncated": True,
        "truncated_questions": ["small"],
        "options": {"small": {"total": 17, "distinct": 15, "tokens_per_option": 4}},
    }


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
    assert [len(router.predictions) for router in routers] == [0, 1]
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


def test_a_round_over_the_token_budget_names_the_caller_question(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch)
    routers[0].too_many_options = lambda question: len(question.get("criteria", ())) > 10
    with pytest.raises(ValueError) as info:
        engine.decide("x", {"dept": choice(30), **NOUL}, "auto")
    assert str(info.value) == (
        "Question 'dept' has too many options for the token budget "
        "(it was narrowed in groups of 20)"
    )


def test_a_final_call_over_the_token_budget_keeps_its_message(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch)
    small = choice(20)
    routers[0].too_many_options = lambda question: question == small
    with pytest.raises(ValueError) as info:
        engine.decide("x", {"big": choice(30), "small": small}, "auto")
    assert str(info.value) == "Question 'small' has too many options for the token budget"


def test_a_final_call_error_on_a_score_with_object_levels_stays_a_value_error(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch)
    routers[0].too_many_options = lambda question: question["type"] == "score"
    urgency = {"type": "score", "instructions": "How urgent?", "criteria": [{"a": 1}, {"b": 2}]}
    with pytest.raises(ValueError) as info:
        engine.decide("x", {"big": choice(30), "urgency": urgency}, "auto")
    assert str(info.value) == "Question 'urgency' has too many options for the token budget"


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


def test_cached_tokenizers_count_as_loaded_so_the_idle_unload_frees_them(monkeypatch):
    engine, routers, _ = make_engine(monkeypatch, tokenizer_factory=lambda path: FakeTokenizer())
    engine.count_tokens("one two", "multilingual")
    assert routers[0].loaded == []
    assert engine.is_loaded() is True
    engine.unload()
    assert engine.is_loaded() is False


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
            calls.append((group_size, predict_kwargs))
            return agent.predict(state, questions, **predict_kwargs)

    monkeypatch.setattr(engine_module, "_laya", lambda: FakeLaya)
    engine, routers, _ = make_engine(monkeypatch)
    engine.decide("x", {"q": choice(25)}, "auto")
    assert calls == [(20, {"model": "english"})]
    assert [model for _, _, model in routers[0].predictions] == ["english"]


def test_the_default_tokenizer_factory_loads_a_tokenizer_json(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "refund": 1, "me": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    loaded = engine_module.default_tokenizer_factory(path)
    assert loaded.encode("refund me refund", add_special_tokens=False).ids == [1, 2, 1]
