import functools
import io
import json
import logging
import os
import plistlib
import runpy
import stat
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx2
import pytest
import uvicorn
from starlette.testclient import TestClient

from local_jev import app as app_module
from local_jev import checkpoints, cli, launchd
from local_jev.config import LayaSettings, Settings
from local_jev.engines import laya_mlx
from local_jev.engines.laya_mlx import LayaMLXEngine
from local_jev.errors import ChecksumMismatch, ModelUnavailable
from local_jev.service import DecisionService
from tests.fakes import FakeEngine

QUESTIONS = '{"q": {"type": "noul", "instructions": "?"}}'
REFUND = '{"refund": {"type": "noul", "instructions": "Does the customer want money back?"}}'
BAD_TOKEN_MESSAGE = "could not send the request: LOCAL_JEV_TOKEN is not a valid HTTP header value\n"

# Taken before no_side_effects replaces them, for the few tests that use the real thing on purpose.
REAL_REQUEST = httpx2.request
REAL_INSTALL = launchd.install
REAL_UNINSTALL = launchd.uninstall


def _refuse(what):
    def refuse(*args, **kwargs):
        pytest.fail(f"a CLI test reached {what}; patch it inside the test that needs it")

    return refuse


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_JEV_CONFIG", str(tmp_path / "none.toml"))
    for name in ("LOCAL_JEV_TOKEN", "LOCAL_JEV_PORT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_side_effects(monkeypatch, tmp_path):
    """Keep every test off this machine: no launchctl, claude, server or download, and no real home.

    Each call below is replaced by a call to pytest.fail. A test that needs one of them replaces it
    again with monkeypatch.setattr, and the later setattr wins.
    """
    # install-agent copies these from the environment, so a developer's own settings must not leak in.
    others = [
        name for name in os.environ if name.startswith("LOCAL_JEV_") and name != "LOCAL_JEV_CONFIG"
    ]
    for name in [*others, "HF_HOME", "HF_HUB_CACHE"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_JEV_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOCAL_JEV_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(checkpoints.constants, "HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    # cli.subprocess and launchd.subprocess are one module, so these two lines patch the same name.
    monkeypatch.setattr(cli.subprocess, "run", _refuse("subprocess.run"))
    monkeypatch.setattr(launchd.subprocess, "run", _refuse("subprocess.run"))
    # install and uninstall bind run=subprocess.run at import, so REAL_INSTALL would still run launchctl.
    monkeypatch.setitem(REAL_INSTALL.__kwdefaults__, "run", _refuse("launchctl"))
    monkeypatch.setitem(REAL_UNINSTALL.__kwdefaults__, "run", _refuse("launchctl"))
    monkeypatch.setattr(cli.launchd, "install", _refuse("launchd.install"))
    monkeypatch.setattr(cli.launchd, "uninstall", _refuse("launchd.uninstall"))
    monkeypatch.setattr(cli.httpx2, "request", _refuse("an HTTP request"))
    monkeypatch.setattr(uvicorn, "run", _refuse("uvicorn.run"))
    monkeypatch.setattr(checkpoints, "snapshot_download", _refuse("a model download"))


@pytest.fixture
def real_http(monkeypatch):
    """Let a test send real requests, for a loopback server it starts itself."""
    monkeypatch.setattr(cli.httpx2, "request", REAL_REQUEST)


@contextmanager
def loopback_server(status=200, body=None):
    """A small HTTP server on 127.0.0.1. Yields its port and the (request line, Authorization) pairs it saw."""
    seen = []
    reply = json.dumps({"ok": True} if body is None else body).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.requestline, self.headers.get("Authorization")))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def sandboxed_install(monkeypatch, tmp_path, run):
    """Make install-agent call the real launchd.install on tmp_path/home, with `run` in place of launchctl."""
    monkeypatch.setattr(
        cli.launchd, "install", functools.partial(REAL_INSTALL, home=tmp_path / "home", run=run)
    )
    return tmp_path / "home" / "Library" / "LaunchAgents" / f"{launchd.LABEL}.plist"


def test_version(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])
    assert info.value.code == 0
    assert "local-jev 0.1.0" in capsys.readouterr().out


def test_connect_prints_the_claude_code_command(capsys):
    assert cli.main(["connect"]) == 0
    assert capsys.readouterr().out.strip() == (
        "claude mcp add --transport http --scope user local-jev http://127.0.0.1:7711/mcp"
    )


def test_connect_never_prints_the_token(capsys, monkeypatch):
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    assert cli.main(["connect"]) == 0
    out = capsys.readouterr().out
    assert '--header "Authorization: Bearer $LOCAL_JEV_TOKEN"' in out
    assert "s3cret" not in out


def test_connect_apply_runs_claude_with_the_real_token(monkeypatch):
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    calls = []

    def fake_run(argv, check):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.main(["connect", "--apply"]) == 0
    # claude mcp add declares --header as <header...>, so the name and the URL must come before it.
    assert calls == [
        [
            "claude",
            "mcp",
            "add",
            "--transport",
            "http",
            "--scope",
            "user",
            "local-jev",
            "http://127.0.0.1:7711/mcp",
            "--header",
            "Authorization: Bearer s3cret",
        ]
    ]


def test_connect_prints_the_header_after_the_name_and_url(capsys, monkeypatch):
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    assert cli.main(["connect"]) == 0
    assert capsys.readouterr().out == (
        "claude mcp add --transport http --scope user local-jev http://127.0.0.1:7711/mcp "
        '--header "Authorization: Bearer $LOCAL_JEV_TOKEN"\n'
    )


def test_connect_codex_snippet(capsys, monkeypatch):
    monkeypatch.setenv("LOCAL_JEV_PORT", "7800")
    assert cli.main(["connect", "--harness", "codex"]) == 0
    out = capsys.readouterr().out
    assert "[mcp_servers.local-jev]" in out
    assert 'url = "http://127.0.0.1:7800/mcp"' in out


def test_apply_is_for_claude_code_only():
    assert cli.main(["connect", "--harness", "cursor", "--apply"]) == 1


def test_decide_posts_to_the_running_server(monkeypatch, tmp_path, capsys):
    questions = tmp_path / "q.json"
    questions.write_text(QUESTIONS)
    seen = {}

    def fake_request(method, url, *, payload, headers, timeout):
        seen.update(method=method, url=url, payload=payload)
        return 200, {"answers": {"q": {"noul": 0.9}}}

    monkeypatch.setattr(cli, "_request", fake_request)
    assert cli.main(["decide", "--state", "hello", "--questions-file", str(questions)]) == 0
    assert seen == {
        "method": "POST",
        "url": "http://127.0.0.1:7711/v1/systemone",
        "payload": {"state": "hello", "questions": json.loads(QUESTIONS), "model": "auto"},
    }
    assert json.loads(capsys.readouterr().out)["answers"]["q"]["noul"] == 0.9


def test_batch_sends_an_absolute_path_and_the_filter(monkeypatch, tmp_path):
    (tmp_path / "q.json").write_text(QUESTIONS)
    data = tmp_path / "items.txt"
    data.write_text("a\n")
    seen = {}

    def fake_request(method, url, *, payload, headers, timeout):
        seen.update(payload=payload)
        return 200, {}

    monkeypatch.setattr(cli, "_request", fake_request)
    monkeypatch.chdir(tmp_path)
    filter_json = '{"question": "q", "op": ">=", "value": 0.5}'
    assert (
        cli.main(
            ["batch", "--questions-file", "q.json", "--path", "items.txt", "--filter", filter_json]
        )
        == 0
    )
    assert seen["payload"]["path"] == str(data.resolve())
    assert seen["payload"]["filter"] == {"question": "q", "op": ">=", "value": 0.5}


def test_status_when_nothing_is_running(monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise httpx2.ConnectError("refused")

    monkeypatch.setattr(cli, "_request", refuse)
    assert cli.main(["status"]) == 1
    assert "not running" in capsys.readouterr().err


def test_install_agent_masks_the_token_and_installs(monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    installed = {}

    def fake_install(program, log_dir, env):
        installed.update(program=program, env=env)
        return Path("/tmp/io.github.andresrubio.local-jev.plist")

    monkeypatch.setattr(cli.launchd, "install", fake_install)
    assert cli.main(["install-agent", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "***" in out
    assert "s3cret" not in out
    assert installed["env"]["LOCAL_JEV_TOKEN"] == "s3cret"
    assert installed["program"][-1] == "serve"


def test_install_agent_can_be_cancelled(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    monkeypatch.setattr(
        cli.launchd, "install", lambda *args, **kwargs: pytest.fail("must not install")
    )
    assert cli.main(["install-agent"]) == 1


# pull


def fake_pull_engine(monkeypatch, prefetch):
    """Replace the engine that `pull` builds, so no Router is made and nothing is downloaded."""

    class Engine:
        def __init__(self, settings):
            self.settings = settings

        def prefetch(self, names):
            prefetch(names)

    monkeypatch.setattr(laya_mlx, "LayaMLXEngine", Engine)


def test_pull_fetches_english_and_multilingual(monkeypatch, capsys):
    asked = []
    fake_pull_engine(monkeypatch, asked.append)
    assert cli.main(["pull"]) == 0
    assert asked == [("english",), ("multilingual",)]
    out = capsys.readouterr().out
    assert "english: downloading if needed, then verifying (843 MB)" in out
    assert "english: ready" in out
    assert "multilingual: ready" in out
    assert "typed-decisions" not in out


def test_pull_all_adds_typed_decisions(monkeypatch, capsys):
    asked = []
    fake_pull_engine(monkeypatch, asked.append)
    assert cli.main(["pull", "--all"]) == 0
    assert asked == [("english",), ("multilingual",), ("typed-decisions",)]
    assert "typed-decisions: ready" in capsys.readouterr().out


def test_pull_prints_the_hint_of_a_failed_checkpoint(monkeypatch, capsys):
    hint = "delete /cache/model.safetensors and run `local-jev pull`"

    def prefetch(names):
        raise ChecksumMismatch("/cache/model.safetensors does not match its pinned SHA-256", hint)

    fake_pull_engine(monkeypatch, prefetch)
    assert cli.main(["pull"]) == 1
    captured = capsys.readouterr()
    assert captured.err == (
        "english: failed: /cache/model.safetensors does not match its pinned SHA-256\n"
        f"  hint: {hint}\n"
    )
    assert "multilingual" not in captured.out


@pytest.mark.parametrize(
    "error",
    [OSError("no route to host"), ModelUnavailable("no route to host")],
    ids=["os", "empty-hint"],
)
def test_pull_prints_no_hint_line_when_there_is_no_hint(monkeypatch, capsys, error):
    def prefetch(names):
        raise error

    fake_pull_engine(monkeypatch, prefetch)
    assert cli.main(["pull"]) == 1
    assert capsys.readouterr().err == "english: failed: no route to host\n"


def test_pull_on_a_truncated_download_names_the_file_to_delete(monkeypatch, tmp_path, capsys):
    cache = tmp_path / "cache"
    laya = LayaSettings()
    root = checkpoints.snapshot_root(laya.repo, laya.revision, cache)
    for item in checkpoints.REQUIRED:
        (root / item).parent.mkdir(parents=True, exist_ok=True)
        (root / item).write_bytes(b"truncated")
    monkeypatch.setattr(
        laya_mlx,
        "LayaMLXEngine",
        lambda settings: LayaMLXEngine(
            settings, cache_dir=cache, router_factory=lambda root, dtype: None
        ),
    )
    assert cli.main(["pull"]) == 1
    weights = root / "model.safetensors"
    assert capsys.readouterr().err == (
        f"english: failed: {weights} is 9 bytes, expected {checkpoints.CHECKPOINTS['english'].size}\n"
        f"  hint: delete {weights.resolve()} and run `local-jev pull`\n"
    )


def test_pull_only_applies_to_the_laya_backend(monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_JEV_BACKEND", "remote")
    monkeypatch.setenv("LOCAL_JEV_REMOTE_URL", "http://127.0.0.1:9")
    assert cli.main(["pull"]) == 1
    assert "pull only applies to the laya-mlx backend" in capsys.readouterr().err


# serve


def test_serve_sets_up_logging_before_it_builds_the_app(monkeypatch, tmp_path):
    events = []

    def create_app(settings):
        events.append(("create_app", settings.port))
        return "the app"

    monkeypatch.setattr(
        cli, "_configure_logging", lambda log_dir: events.append(("logging", log_dir))
    )
    monkeypatch.setattr(app_module, "create_app", create_app)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: events.append(("run", app, kwargs)))
    assert cli.main(["serve"]) == 0
    assert events == [
        ("logging", tmp_path / "logs"),
        ("create_app", 7711),
        (
            "run",
            "the app",
            {"host": "127.0.0.1", "port": 7711, "log_config": None, "lifespan": "on"},
        ),
    ]


def test_serve_reports_a_certificate_file_that_cannot_be_loaded_as_a_config_error(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("LOCAL_JEV_BACKEND", "remote")
    monkeypatch.setenv("LOCAL_JEV_REMOTE_URL", "https://127.0.0.1:8443/v1/systemone")
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    monkeypatch.setattr(cli, "_configure_logging", lambda log_dir: None)
    monkeypatch.setattr(uvicorn, "run", _refuse("uvicorn.run"))
    assert cli.main(["serve"]) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"config error: SSL_CERT_FILE is {tmp_path / 'missing.pem'}, which ")
    assert "Traceback" not in err


def test_serve_port_option_reaches_the_app_and_the_server(monkeypatch):
    # The app's Host allowlist is built from settings.port, so both must see the same port.
    seen = {}

    def create_app(settings):
        seen["app"] = settings.port

    monkeypatch.setattr(cli, "_configure_logging", lambda log_dir: None)
    monkeypatch.setattr(app_module, "create_app", create_app)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: seen.update(server=kwargs["port"]))
    assert cli.main(["serve", "--port", "7800"]) == 0
    assert seen == {"app": 7800, "server": 7800}


class Terminal(io.StringIO):
    """A stderr that says it is a terminal."""

    def isatty(self):
        return True


@contextmanager
def logging_handlers(monkeypatch, tmp_path, stderr):
    """The handlers _configure_logging sets up while sys.stderr is `stderr`, closed afterwards."""
    configured = {}
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: configured.update(kwargs))
    cli._configure_logging(tmp_path / "logs")
    try:
        assert configured["level"] == logging.INFO
        yield configured["handlers"]
    finally:
        for handler in configured["handlers"]:
            handler.close()


def test_logging_goes_to_a_rotating_server_log_and_to_a_terminal(monkeypatch, tmp_path):
    terminal = Terminal()
    with logging_handlers(monkeypatch, tmp_path, terminal) as handlers:
        file_handler, stream_handler = handlers
        assert isinstance(file_handler, RotatingFileHandler)
        assert Path(file_handler.baseFilename) == tmp_path / "logs" / "server.log"
        assert (file_handler.maxBytes, file_handler.backupCount) == (10 * 1024 * 1024, 5)
        assert type(stream_handler) is logging.StreamHandler
        assert stream_handler.stream is terminal
        record = logging.LogRecord("local_jev.x", logging.INFO, __file__, 1, "started", None, None)
        assert " INFO local_jev.x: started" in file_handler.format(record)


def test_logging_goes_only_to_the_server_log_when_stderr_is_not_a_terminal(monkeypatch, tmp_path):
    # Under launchd stderr is launchd.err.log, which never rotates, so it must not get every line.
    with logging_handlers(monkeypatch, tmp_path, io.StringIO()) as handlers:
        [file_handler] = handlers
        assert isinstance(file_handler, RotatingFileHandler)
        assert Path(file_handler.baseFilename) == tmp_path / "logs" / "server.log"


def test_logging_goes_only_to_the_server_log_when_there_is_no_stderr(monkeypatch, tmp_path):
    # Python sets sys.stderr to None when serve starts with its stderr closed.
    with logging_handlers(monkeypatch, tmp_path, None) as handlers:
        [file_handler] = handlers
        assert Path(file_handler.baseFilename) == tmp_path / "logs" / "server.log"


# the agent


def test_install_agent_shows_the_plist_before_it_asks(monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    asked = {}

    def answer(prompt):
        asked.update(prompt=prompt, shown=capsys.readouterr().out)
        return "n"

    monkeypatch.setattr("builtins.input", answer)
    assert cli.main(["install-agent"]) == 1
    assert asked["prompt"] == "Install and start the login agent? [y/N] "
    shown = asked["shown"]
    assert f"This writes {launchd.plist_path()} (mode 600)" in shown
    assert "<key>RunAtLoad</key>" in shown
    assert "s3cret" not in shown
    assert capsys.readouterr().out == "cancelled\n"


def fake_agent_install(monkeypatch, tmp_path):
    """Replace launchd.install with a recorder; return the dict of what install-agent passed to it."""
    installed = {}

    def fake_install(program, log_dir, env):
        installed.update(program=program, log_dir=log_dir, env=env)
        return tmp_path / "agent.plist"

    monkeypatch.setattr(cli.launchd, "install", fake_install)
    return installed


def test_install_agent_passes_the_config_path_and_the_log_folder(monkeypatch, tmp_path, capsys):
    installed = fake_agent_install(monkeypatch, tmp_path)
    assert cli.main(["install-agent", "--yes"]) == 0
    assert installed["log_dir"] == tmp_path / "logs"
    assert installed["env"] == {
        "LOCAL_JEV_CONFIG": str(tmp_path / "none.toml"),
        "LOCAL_JEV_LOG_DIR": str(tmp_path / "logs"),
        "LOCAL_JEV_RESULTS_DIR": str(tmp_path / "results"),
    }
    assert capsys.readouterr().out.endswith(
        f"installed {tmp_path / 'agent.plist'}; local-jev now starts at login (logs in {tmp_path / 'logs'})\n"
    )


def test_install_agent_copies_every_setting_the_server_reads(monkeypatch, tmp_path, capsys):
    # The login agent does not read the shell profile, so a setting made there would reach `status`
    # and `connect` but not the server.
    copied = {
        "LOCAL_JEV_PORT": "7800",
        "LOCAL_JEV_BACKEND": "remote",
        "LOCAL_JEV_REMOTE_URL": "http://127.0.0.1:9",
        "LOCAL_JEV_IDLE_UNLOAD_MINUTES": "5",
        "LOCAL_JEV_NOUL_MODE": "ab_labels",
        "LOCAL_JEV_LOG_INPUTS": "true",
        "LOCAL_JEV_TOKEN": "s3cret",
    }
    for name, value in copied.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf" / "hub"))
    # With the remote backend the server also reads the API key, from the variable that
    # remote.api_key_env names (the user's decision of 2026-10-20).
    (tmp_path / "none.toml").write_text('[remote]\napi_key_env = "JEV_API_KEY"\n')
    monkeypatch.setenv("JEV_API_KEY", "jev_secret")
    # Not settings the server reads, or credentials that must stay out of the plist.
    monkeypatch.setenv("LOCAL_JEV_SOMETHING_ELSE", "1")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    installed = fake_agent_install(monkeypatch, tmp_path)
    assert cli.main(["install-agent", "--yes"]) == 0
    assert installed["env"] == {
        **copied,
        "LOCAL_JEV_CONFIG": str(tmp_path / "none.toml"),
        "LOCAL_JEV_LOG_DIR": str(tmp_path / "logs"),
        "LOCAL_JEV_RESULTS_DIR": str(tmp_path / "results"),
        "HF_HOME": str(tmp_path / "hf"),
        "HF_HUB_CACHE": str(tmp_path / "hf" / "hub"),
        "JEV_API_KEY": "jev_secret",
    }
    captured = capsys.readouterr()
    assert shown_environment(captured.out) == {
        **installed["env"],
        "LOCAL_JEV_TOKEN": "***",
        "JEV_API_KEY": "***",
    }
    for secret in ("s3cret", "hf_secret", "jev_secret"):
        assert secret not in captured.out + captured.err
    assert captured.err == ""


def shown_environment(out):
    """The EnvironmentVariables of the plist that install-agent printed."""
    plist = out[out.index("<?xml") : out.index("</plist>") + len("</plist>")]
    return plistlib.loads(plist.encode())["EnvironmentVariables"]


REMOTE_WITH_A_KEY = (
    'backend = "remote"\n\n'
    '[remote]\nurl = "http://127.0.0.1:8009/v1/systemone"\napi_key_env = "JEV_API_KEY"\n'
)


@pytest.mark.parametrize("value", [None, "", "  \n"], ids=["unset", "empty", "blank"])
def test_install_agent_warns_when_the_remote_api_key_is_not_set_and_still_installs(
    monkeypatch, tmp_path, capsys, value
):
    # build_engine strips the key and sends none when nothing is left.
    (tmp_path / "none.toml").write_text(REMOTE_WITH_A_KEY)
    if value is None:
        monkeypatch.delenv("JEV_API_KEY", raising=False)
    else:
        monkeypatch.setenv("JEV_API_KEY", value)
    installed = fake_agent_install(monkeypatch, tmp_path)
    assert cli.main(["install-agent", "--yes"]) == 0
    assert "JEV_API_KEY" not in installed["env"]
    captured = capsys.readouterr()
    assert captured.err == (
        "warning: JEV_API_KEY is not set, so the login agent will send no API key"
        " to the remote backend\n"
    )
    assert captured.out.endswith(f"(logs in {tmp_path / 'logs'})\n")


def test_install_agent_copies_no_api_key_for_the_laya_backend(monkeypatch, tmp_path, capsys):
    (tmp_path / "none.toml").write_text('[remote]\napi_key_env = "JEV_API_KEY"\n')
    monkeypatch.setenv("JEV_API_KEY", "jev_secret")
    installed = fake_agent_install(monkeypatch, tmp_path)
    assert cli.main(["install-agent", "--yes"]) == 0
    assert "JEV_API_KEY" not in installed["env"]
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "jev_secret" not in captured.out


def test_install_agent_makes_the_copied_paths_absolute(monkeypatch, tmp_path, capsys):
    # The agent starts in /, so a relative path would point somewhere else than it does here.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOCAL_JEV_CONFIG", "conf/local.toml")
    monkeypatch.setenv("LOCAL_JEV_LOG_DIR", "logs")
    monkeypatch.setenv("LOCAL_JEV_RESULTS_DIR", "~/results")
    monkeypatch.setenv("HF_HOME", "hf/../models")
    # huggingface_hub expands $VAR in its two cache variables, and the agent has no shell to do it.
    monkeypatch.setenv("HF_HUB_CACHE", "$HOME/hub")
    installed = fake_agent_install(monkeypatch, tmp_path)
    assert cli.main(["install-agent", "--yes"]) == 0
    assert installed["env"] == {
        "LOCAL_JEV_CONFIG": str(tmp_path / "conf" / "local.toml"),
        "LOCAL_JEV_LOG_DIR": str(tmp_path / "logs"),
        "LOCAL_JEV_RESULTS_DIR": str(tmp_path / "home" / "results"),
        "HF_HOME": str(tmp_path / "models"),
        "HF_HUB_CACHE": str(tmp_path / "home" / "hub"),
    }
    assert installed["log_dir"] == tmp_path / "logs"
    assert capsys.readouterr().out.endswith(f"(logs in {tmp_path / 'logs'})\n")


def test_install_agent_does_not_copy_a_setting_that_is_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_JEV_NOUL_MODE", "")  # config reads an empty value as not set
    monkeypatch.setenv("HF_HOME", "")
    installed = fake_agent_install(monkeypatch, tmp_path)
    assert cli.main(["install-agent", "--yes"]) == 0
    assert "LOCAL_JEV_NOUL_MODE" not in installed["env"]
    assert "HF_HOME" not in installed["env"]


def test_install_agent_writes_the_real_token_to_a_private_plist_and_shows_it_masked(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    monkeypatch.setenv("LOCAL_JEV_PORT", "7800")
    launchctl = []
    plist = sandboxed_install(monkeypatch, tmp_path, lambda argv, **kw: launchctl.append(argv))
    assert cli.main(["install-agent", "--yes"]) == 0
    written = plistlib.loads(plist.read_bytes())["EnvironmentVariables"]
    assert (written["LOCAL_JEV_TOKEN"], written["LOCAL_JEV_PORT"]) == ("s3cret", "7800")
    assert stat.S_IMODE(plist.stat().st_mode) == 0o600
    assert [argv[1] for argv in launchctl] == ["bootout", "bootstrap"]
    out = capsys.readouterr().out
    assert "s3cret" not in out
    assert out.endswith(
        f"installed {plist}; local-jev now starts at login (logs in {tmp_path / 'logs'})\n"
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write into a read-only folder")
def test_install_agent_that_cannot_write_the_plist_says_so_and_stops_nothing(
    monkeypatch, tmp_path, capsys
):
    launchctl = []
    plist = sandboxed_install(monkeypatch, tmp_path, lambda argv, **kw: launchctl.append(argv))
    plist.parent.mkdir(parents=True)
    plist.parent.chmod(0o500)
    try:
        assert cli.main(["install-agent", "--yes"]) == 1
    finally:
        plist.parent.chmod(0o700)
    captured = capsys.readouterr()
    assert captured.err.startswith("could not install the login agent: ")
    assert "Permission denied" in captured.err
    assert captured.err.endswith("the agent that is running now, if any, was left alone\n")
    assert "installed" not in captured.out
    assert launchctl == []


def test_install_agent_whose_launchctl_fails_says_what_it_left_behind(
    monkeypatch, tmp_path, capsys
):
    def run(argv, **kwargs):
        if argv[1] == "bootstrap":
            raise subprocess.CalledProcessError(5, argv)

    plist = sandboxed_install(monkeypatch, tmp_path, run)
    assert cli.main(["install-agent", "--yes"]) == 1
    captured = capsys.readouterr()
    assert plist.is_file()
    assert captured.err == (
        "launchctl could not load the login agent (exit status 5).\n"
        f"The plist is written at {plist}, so the agent will load at your next login."
        " An agent that was running before has been stopped.\n"
        f"To load it now: launchctl bootstrap gui/$UID {plist}\n"
        "To remove it: local-jev uninstall-agent\n"
    )
    assert "installed" not in captured.out


@pytest.mark.parametrize(
    ("found", "expected"),
    [
        ("/opt/bin/local-jev", ["/opt/bin/local-jev", "serve"]),
        (None, [sys.executable, "-m", "local_jev", "serve"]),
    ],
    ids=["installed-command", "python-dash-m"],
)
def test_the_agent_runs_serve(monkeypatch, found, expected):
    monkeypatch.setattr(cli.shutil, "which", lambda name: found)
    installed = {}
    monkeypatch.setattr(
        cli.launchd, "install", lambda program, log_dir, env: installed.update(program=program)
    )
    assert cli.main(["install-agent", "--yes"]) == 0
    assert installed["program"] == expected


def test_uninstall_agent_can_be_cancelled(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert cli.main(["uninstall-agent"]) == 1
    assert capsys.readouterr().out == "cancelled\n"


@pytest.mark.parametrize(
    ("removed", "message"),
    [(True, "removed the login agent"), (False, "no login agent was installed")],
)
def test_uninstall_agent_says_what_it_did(monkeypatch, capsys, removed, message):
    monkeypatch.setattr(cli.launchd, "uninstall", lambda: removed)
    assert cli.main(["uninstall-agent", "--yes"]) == 0
    assert capsys.readouterr().out == f"{message}\n"


def test_a_closed_stdin_answers_no(monkeypatch):
    def end_of_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", end_of_input)
    assert cli.main(["uninstall-agent"]) == 1


@pytest.mark.parametrize(
    ("answer", "status"),
    [("y", 0), ("Y", 0), ("yes", 0), (" YES ", 0), ("", 1), ("n", 1), ("no", 1), ("yep", 1)],
)
def test_only_yes_confirms(monkeypatch, answer, status):
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    monkeypatch.setattr(cli.launchd, "uninstall", lambda: True)
    assert cli.main(["uninstall-agent"]) == status


@pytest.mark.parametrize(
    ("config", "env"),
    [("port = 80\n", {}), ("port = [\n", {}), ("", {"LOCAL_JEV_PORT": "80"})],
    ids=["bad value", "bad toml", "bad variable"],
)
def test_uninstall_agent_works_while_the_settings_are_broken(
    monkeypatch, tmp_path, capsys, config, env
):
    # A config error makes the login agent exit, and launchd starts it again every 10 seconds.
    (tmp_path / "none.toml").write_text(config)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cli.launchd, "uninstall", lambda: True)
    assert cli.main(["uninstall-agent", "--yes"]) == 0
    assert capsys.readouterr().out == "removed the login agent\n"
    assert cli.main(["status"]) == 2  # every other command still stops on the config error


# connect


def test_connect_cursor_snippet(capsys, monkeypatch):
    monkeypatch.setenv("LOCAL_JEV_PORT", "7800")
    assert cli.main(["connect", "--harness", "cursor"]) == 0
    captured = capsys.readouterr()
    # The comment goes to stderr, so the whole of stdout can be pasted as JSON.
    assert captured.err == "// add to ~/.cursor/mcp.json\n"
    assert json.loads(captured.out) == {
        "mcpServers": {"local-jev": {"url": "http://127.0.0.1:7800/mcp"}}
    }


def test_connect_cursor_snippet_has_cursor_read_the_token_from_the_environment(capsys, monkeypatch):
    # Cursor fills in ${env:NAME} itself, so the token stays out of mcp.json.
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    assert cli.main(["connect", "--harness", "cursor"]) == 0
    captured = capsys.readouterr()
    assert captured.err == "// add to ~/.cursor/mcp.json\n"
    assert json.loads(captured.out) == {
        "mcpServers": {
            "local-jev": {
                "url": "http://127.0.0.1:7711/mcp",
                "headers": {"Authorization": "Bearer ${env:LOCAL_JEV_TOKEN}"},
            }
        }
    }


def test_connect_json_snippet_names_the_token_variable_without_its_value(capsys, monkeypatch):
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    assert cli.main(["connect", "--harness", "json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == "// add to your agent's MCP configuration\n"
    assert json.loads(captured.out) == {
        "mcpServers": {
            "local-jev": {
                "type": "http",
                "url": "http://127.0.0.1:7711/mcp",
                "headers": {"Authorization": "Bearer <your LOCAL_JEV_TOKEN>"},
            }
        }
    }


def test_connect_codex_snippet_is_printed_whole_on_stdout(capsys, monkeypatch):
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    assert cli.main(["connect", "--harness", "codex"]) == 0
    captured = capsys.readouterr()
    assert captured.out == (
        "# add to ~/.codex/config.toml\n"
        "[mcp_servers.local-jev]\n"
        'url = "http://127.0.0.1:7711/mcp"\n'
        'bearer_token_env_var = "LOCAL_JEV_TOKEN"\n'
    )
    assert captured.err == ""


@pytest.mark.parametrize("harness", ["codex", "cursor", "json"])
def test_no_snippet_prints_the_token(capsys, monkeypatch, harness):
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    assert cli.main(["connect", "--harness", harness]) == 0
    captured = capsys.readouterr()
    assert "s3cret" not in captured.out + captured.err


def test_connect_codex_snippet_reads_the_token_from_the_environment(capsys, monkeypatch):
    assert cli.main(["connect", "--harness", "codex"]) == 0
    assert "bearer_token_env_var" not in capsys.readouterr().out
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    assert cli.main(["connect", "--harness", "codex"]) == 0
    assert 'bearer_token_env_var = "LOCAL_JEV_TOKEN"' in capsys.readouterr().out


def test_connect_apply_needs_claude_on_the_path(monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.main(["connect", "--apply"]) == 1
    assert "the claude command is not on PATH" in capsys.readouterr().err


def test_connect_apply_can_be_cancelled(monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert cli.main(["connect", "--apply"]) == 1
    assert capsys.readouterr().out.endswith("cancelled\n")


def test_connect_apply_without_a_token_sends_no_header(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    calls = []

    def fake_run(argv, check):
        calls.append((argv, check))
        return subprocess.CompletedProcess(argv, 3)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.main(["connect", "--apply"]) == 3
    assert calls == [
        (
            [
                "claude",
                "mcp",
                "add",
                "--transport",
                "http",
                "--scope",
                "user",
                "local-jev",
                "http://127.0.0.1:7711/mcp",
            ],
            False,
        )
    ]


# decide, batch and status


def record_requests(monkeypatch, reply=(200, {})):
    """Replace the HTTP seam and return the list of requests the command sends."""
    sent = []

    def fake_request(method, url, *, payload, headers, timeout):
        sent.append(
            {
                "method": method,
                "url": url,
                "payload": payload,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return reply

    monkeypatch.setattr(cli, "_request", fake_request)
    return sent


def test_decide_reads_a_json_state_file_as_data_and_any_other_file_as_text(monkeypatch, tmp_path):
    (tmp_path / "q.json").write_text(QUESTIONS)
    (tmp_path / "ticket.json").write_text('{"subject": "refund"}')
    (tmp_path / "ORDER.JSON").write_text('{"order": 7}')
    (tmp_path / "note.txt").write_text("plain words\n")
    sent = record_requests(monkeypatch)
    for name in ("ticket.json", "ORDER.JSON", "note.txt"):
        argv = ["decide", "--state-file", str(tmp_path / name), "--questions-file"]
        assert cli.main([*argv, str(tmp_path / "q.json")]) == 0
    assert [request["payload"]["state"] for request in sent] == [
        {"subject": "refund"},
        {"order": 7},
        "plain words\n",
    ]


@pytest.mark.parametrize(
    "flags", [[], ["--state", "a", "--state-file", "b.txt"]], ids=["none", "both"]
)
def test_decide_needs_exactly_one_state_source(flags):
    with pytest.raises(SystemExit) as info:
        cli.main(["decide", "--questions-file", "q.json", *flags])
    assert info.value.code == 2


def test_decide_forwards_the_model(monkeypatch, tmp_path):
    (tmp_path / "q.json").write_text(QUESTIONS)
    sent = record_requests(monkeypatch)
    argv = ["decide", "--state", "x", "--questions-file", str(tmp_path / "q.json")]
    assert cli.main([*argv, "--model", "multilingual"]) == 0
    assert sent[0]["payload"]["model"] == "multilingual"


def test_batch_expands_a_leading_tilde(monkeypatch, tmp_path):
    (tmp_path / "q.json").write_text(QUESTIONS)
    items = tmp_path / "home" / "items.txt"  # HOME is tmp_path / "home" in these tests
    items.parent.mkdir()
    items.write_text("a\n")
    sent = record_requests(monkeypatch)
    argv = ["batch", "--questions-file", str(tmp_path / "q.json"), "--path", "~/items.txt"]
    assert cli.main(argv) == 0
    assert sent[0]["payload"]["path"] == str(items.resolve())


def test_batch_defaults(monkeypatch, tmp_path):
    (tmp_path / "q.json").write_text(QUESTIONS)
    items = tmp_path / "items.txt"
    items.write_text("a\n")
    sent = record_requests(monkeypatch)
    argv = ["batch", "--questions-file", str(tmp_path / "q.json"), "--path", str(items)]
    assert cli.main(argv) == 0
    assert sent[0]["payload"] == {
        "questions": json.loads(QUESTIONS),
        "path": str(items.resolve()),
        "format": "auto",
        "chunk_tokens": 256,
        "top_k": 20,
        "model": "auto",
    }


def test_batch_forwards_every_option(monkeypatch, tmp_path):
    (tmp_path / "q.json").write_text(QUESTIONS)
    items = tmp_path / "items.jsonl"
    items.write_text("{}\n")
    sent = record_requests(monkeypatch)
    argv = [
        "batch", "--questions-file", str(tmp_path / "q.json"), "--path", str(items),
        "--format", "jsonl", "--field", "body.text", "--chunk-tokens", "128", "--top-k", "5",
        "--model", "english",
    ]  # fmt: skip
    assert cli.main(argv) == 0
    assert sent[0]["payload"] == {
        "questions": json.loads(QUESTIONS),
        "path": str(items.resolve()),
        "format": "jsonl",
        "field": "body.text",
        "chunk_tokens": 128,
        "top_k": 5,
        "model": "english",
    }


def test_each_command_sends_the_token_and_waits_longer_than_the_server(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
    (tmp_path / "q.json").write_text(QUESTIONS)
    (tmp_path / "items.txt").write_text("a\n")
    sent = record_requests(monkeypatch)
    questions = str(tmp_path / "q.json")
    batch = ["batch", "--questions-file", questions, "--path", str(tmp_path / "items.txt")]
    assert cli.main(["status"]) == 0
    assert cli.main(["decide", "--state", "x", "--questions-file", questions]) == 0
    assert cli.main(batch) == 0
    assert [(r["method"], r["url"], r["timeout"]) for r in sent] == [
        ("GET", "http://127.0.0.1:7711/v1/status", 10),
        ("POST", "http://127.0.0.1:7711/v1/systemone", 35),
        ("POST", "http://127.0.0.1:7711/v1/batch", 330),
    ]
    assert sent[0]["payload"] is None
    assert all(r["headers"] == {"Authorization": "Bearer s3cret"} for r in sent)


def test_requests_use_the_configured_port_and_no_header_without_a_token(monkeypatch):
    monkeypatch.setenv("LOCAL_JEV_PORT", "7800")
    sent = record_requests(monkeypatch)
    assert cli.main(["status"]) == 0
    assert sent[0]["url"] == "http://127.0.0.1:7800/v1/status"
    assert sent[0]["headers"] == {}


def test_an_error_body_is_printed_and_the_exit_status_is_1(monkeypatch, capsys):
    body = {"error": {"code": "invalid_request", "message": "bad question", "hint": "fix é it"}}
    record_requests(monkeypatch, reply=(422, body))
    assert cli.main(["status"]) == 1
    out = capsys.readouterr().out
    assert json.loads(out) == body
    assert "fix é it" in out


def test_a_failed_request_is_reported(monkeypatch, capsys):
    def time_out(*args, **kwargs):
        raise httpx2.ReadTimeout("took too long")

    monkeypatch.setattr(cli, "_request", time_out)
    assert cli.main(["status"]) == 1
    assert capsys.readouterr().err == "request failed: took too long\n"


def test_request_returns_the_status_and_the_decoded_body(monkeypatch):
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update(method=method, url=url, **kwargs)
        return httpx2.Response(201, json={"ok": True})

    monkeypatch.setattr(cli.httpx2, "request", fake_request)
    reply = cli._request(
        "POST", "http://127.0.0.1:7711/v1/batch", payload={"a": 1}, headers={"X": "y"}, timeout=3
    )
    assert reply == (201, {"ok": True})
    assert seen == {
        "method": "POST",
        "url": "http://127.0.0.1:7711/v1/batch",
        "json": {"a": 1},
        "headers": {"X": "y"},
        "timeout": 3,
        "trust_env": False,
    }


def test_loopback_requests_never_go_through_a_proxy(monkeypatch, real_http):
    # With HTTP_PROXY honoured, the bearer token and the request body would be handed to the proxy.
    with (
        loopback_server() as (port, target),
        loopback_server(502, {"error": "a proxy"}) as (proxy_port, proxy),
    ):
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, f"http://127.0.0.1:{proxy_port}")
        monkeypatch.setenv("LOCAL_JEV_PORT", str(port))
        monkeypatch.setenv("LOCAL_JEV_TOKEN", "probe-token-1")
        assert cli.main(["status"]) == 0
    assert proxy == []
    assert target == [("GET /v1/status HTTP/1.1", "Bearer probe-token-1")]


UNSENDABLE_TOKENS = pytest.mark.parametrize(
    "token",
    ["probe-token-2 ", "probe\ntoken-4", "probe\rtoken-5", "probe-tökén-6"],
    ids=["trailing-space", "newline", "carriage-return", "non-ascii"],
)


@UNSENDABLE_TOKENS
def test_a_token_that_is_not_a_header_value_stops_every_command_unseen(
    monkeypatch, capsys, real_http, token
):
    # The server would answer every request with a 401, so the settings refuse the token.
    with loopback_server() as (port, seen):
        monkeypatch.setenv("LOCAL_JEV_PORT", str(port))
        monkeypatch.setenv("LOCAL_JEV_TOKEN", token)
        assert cli.main(["status"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "config error: LOCAL_JEV_TOKEN must be printable ASCII with no whitespace at either end\n"
    )
    assert seen == []


@UNSENDABLE_TOKENS
def test_the_request_seam_never_prints_or_sends_a_token_that_is_not_a_header_value(
    capsys, real_http, token
):
    # httpx2 refuses such a header with a message that repeats the whole value, token included,
    # and cannot encode a non-ASCII one. _call keeps its own check for Settings made in code.
    with loopback_server() as (port, seen):
        assert (
            cli._call(Settings(port=port, token=token), "GET", "/v1/status", None, timeout=5) == 1
        )
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", BAD_TOKEN_MESSAGE)
    assert seen == []


def test_request_wraps_a_reply_that_is_not_json(monkeypatch):
    text = "Internal Server Error" + " ." * 400
    monkeypatch.setattr(
        cli.httpx2, "request", lambda *args, **kwargs: httpx2.Response(500, text=text)
    )
    status, body = cli._request(
        "GET", "http://127.0.0.1:7711/v1/status", payload=None, headers={}, timeout=3
    )
    assert status == 500
    assert body == {"error": {"code": "bad_response", "message": text[:500], "hint": ""}}


def test_a_bad_setting_is_reported_and_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_JEV_PORT", "80")
    assert cli.main(["connect"]) == 2
    captured = capsys.readouterr()
    assert captured.err == "config error: port must be between 1024 and 65535\n"
    assert captured.out == ""


# the real routes, in process


@contextmanager
def real_app(monkeypatch, settings):
    """The real app on a fake engine. The CLI's HTTP seam calls it in process, with no socket."""
    monkeypatch.setenv("LOCAL_JEV_PORT", str(settings.port))
    app = app_module.create_app(settings, service=DecisionService(settings, FakeEngine()))
    with TestClient(app, base_url=f"http://127.0.0.1:{settings.port}") as client:

        def through_the_app(method, url, *, payload, headers, timeout):
            response = client.request(method, url, json=payload, headers=headers)
            return response.status_code, response.json()

        monkeypatch.setattr(cli, "_request", through_the_app)
        yield


def test_decide_works_against_the_real_routes(monkeypatch, settings, tmp_path, capsys):
    questions = tmp_path / "q.json"
    questions.write_text(REFUND)
    argv = ["decide", "--state", "yes, a refund please", "--questions-file", str(questions)]
    with real_app(monkeypatch, settings):
        assert cli.main(argv) == 0
    answers = json.loads(capsys.readouterr().out)["answers"]
    assert answers["refund"]["noul"] == pytest.approx(0.9)


def test_batch_works_against_the_real_routes(monkeypatch, settings, tmp_path, capsys):
    (tmp_path / "q.json").write_text(REFUND)
    (tmp_path / "items.txt").write_text("yes, refund me\nno thanks\nyes please\n")
    argv = [
        "batch", "--questions-file", str(tmp_path / "q.json"), "--path", str(tmp_path / "items.txt"),
        "--format", "lines", "--top-k", "1",
        "--filter", '{"question": "refund", "op": ">=", "value": 0.5}',
    ]  # fmt: skip
    with real_app(monkeypatch, settings):
        assert cli.main(argv) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["count"] == 3
    assert len(body["matches"]) == 1


def test_status_works_against_the_real_routes(monkeypatch, settings, capsys):
    with real_app(monkeypatch, settings):
        assert cli.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out)["backend"] == "fake"


def test_the_server_error_is_printed_when_it_rejects_the_questions(
    monkeypatch, settings, tmp_path, capsys
):
    (tmp_path / "q.json").write_text("{}")
    argv = ["decide", "--state", "x", "--questions-file", str(tmp_path / "q.json")]
    with real_app(monkeypatch, settings):
        assert cli.main(argv) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "invalid_request"
    assert error["hint"]


def test_the_token_the_cli_sends_is_the_one_the_server_checks(monkeypatch, settings, capsys):
    with real_app(monkeypatch, replace(settings, token="s3cret")):
        monkeypatch.setenv("LOCAL_JEV_TOKEN", "s3cret")
        assert cli.main(["status"]) == 0
        assert json.loads(capsys.readouterr().out)["config"]["token_required"] is True
        monkeypatch.setenv("LOCAL_JEV_TOKEN", "wrong")
        assert cli.main(["status"]) == 1
        assert json.loads(capsys.readouterr().out)["error"]["code"] == "unauthorized"
        monkeypatch.delenv("LOCAL_JEV_TOKEN")
        assert cli.main(["status"]) == 1
        assert json.loads(capsys.readouterr().out)["error"]["code"] == "unauthorized"


# python -m local_jev


@pytest.mark.parametrize(
    ("argv", "status"),
    [(["--version"], 0), (["connect", "--harness", "cursor", "--apply"], 1)],
    ids=["version", "failure"],
)
def test_python_dash_m_runs_the_cli_and_exits_with_its_status(monkeypatch, argv, status):
    monkeypatch.setattr(sys, "argv", ["local-jev", *argv])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("local_jev", run_name="__main__")
    assert info.value.code == status
