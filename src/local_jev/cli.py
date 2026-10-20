"""The local-jev command line."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import httpx2

from local_jev import __version__, launchd
from local_jev.config import ENV_VARS, PATH_ENV_VARS, ConfigError, Settings, load_settings

NOT_RUNNING = (
    "local-jev is not running; start it with `local-jev serve` or `local-jev install-agent`"
)
BAD_TOKEN = "could not send the request: LOCAL_JEV_TOKEN is not a valid HTTP header value"
HF_ENV_VARS = ("HF_HOME", "HF_HUB_CACHE")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    return args.handler(args, settings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-jev", description="Local, open-weight Jev-style typed decisions."
    )
    parser.add_argument("--version", action="version", version=f"local-jev {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the server in the foreground")
    serve.add_argument("--port", type=int)
    serve.set_defaults(handler=cmd_serve)

    pull = sub.add_parser("pull", help="download and verify the pinned model weights")
    pull.add_argument(
        "--all", action="store_true", help="also download the typed-decisions checkpoint"
    )
    pull.set_defaults(handler=cmd_pull)

    decide = sub.add_parser("decide", help="answer typed questions about one state")
    source = decide.add_mutually_exclusive_group(required=True)
    source.add_argument("--state")
    source.add_argument("--state-file", type=Path)
    decide.add_argument("--questions-file", type=Path, required=True)
    decide.add_argument("--model", default="auto")
    decide.set_defaults(handler=cmd_decide)

    batch = sub.add_parser("batch", help="ask the same questions about every item in a file")
    batch.add_argument("--questions-file", type=Path, required=True)
    batch.add_argument("--path", type=Path, required=True)
    batch.add_argument(
        "--format", default="auto", choices=["auto", "jsonl", "csv", "lines", "chunks"]
    )
    batch.add_argument("--field")
    batch.add_argument("--chunk-tokens", type=int, default=256)
    batch.add_argument(
        "--filter", help='JSON, e.g. {"question": "relevant", "op": ">=", "value": 0.8}'
    )
    batch.add_argument("--top-k", type=int, default=20)
    batch.add_argument("--model", default="auto")
    batch.set_defaults(handler=cmd_batch)

    status = sub.add_parser("status", help="show what the running server is doing")
    status.set_defaults(handler=cmd_status)

    connect = sub.add_parser("connect", help="show how to connect an agent over MCP")
    connect.add_argument(
        "--harness", choices=["claude-code", "codex", "cursor", "json"], default="claude-code"
    )
    connect.add_argument(
        "--apply", action="store_true", help="run the claude mcp add command (Claude Code only)"
    )
    connect.set_defaults(handler=cmd_connect)

    install = sub.add_parser("install-agent", help="start local-jev at login (launchd)")
    install.add_argument("--yes", action="store_true")
    install.set_defaults(handler=cmd_install_agent)

    uninstall = sub.add_parser("uninstall-agent", help="stop starting local-jev at login")
    uninstall.add_argument("--yes", action="store_true")
    uninstall.set_defaults(handler=cmd_uninstall_agent)
    return parser


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from local_jev.app import create_app

    if args.port:
        settings = replace(settings, port=args.port)
    _configure_logging(settings.log_dir)
    uvicorn.run(
        create_app(settings), host="127.0.0.1", port=settings.port, log_config=None, lifespan="on"
    )
    return 0


def cmd_pull(args: argparse.Namespace, settings: Settings) -> int:
    if settings.backend != "laya-mlx":
        print("pull only applies to the laya-mlx backend", file=sys.stderr)
        return 1
    from local_jev.checkpoints import CHECKPOINTS
    from local_jev.engines.laya_mlx import LayaMLXEngine

    names = tuple(CHECKPOINTS) if args.all else ("english", "multilingual")
    engine = LayaMLXEngine(settings.laya)
    for name in names:
        print(
            f"{name}: downloading if needed, then verifying ({CHECKPOINTS[name].size / 1e6:.0f} MB)",
            flush=True,
        )
        try:
            engine.prefetch((name,))
        except Exception as exc:
            print(f"{name}: failed: {exc}", file=sys.stderr)
            hint = getattr(exc, "hint", None)
            if hint:
                print(f"  hint: {hint}", file=sys.stderr)
            return 1
        print(f"{name}: ready")
    return 0


def cmd_decide(args: argparse.Namespace, settings: Settings) -> int:
    state = args.state if args.state is not None else _read_state(args.state_file)
    payload = {"state": state, "questions": _read_json(args.questions_file), "model": args.model}
    return _call(
        settings, "POST", "/v1/systemone", payload, timeout=settings.request_timeout_seconds + 5
    )


def cmd_batch(args: argparse.Namespace, settings: Settings) -> int:
    payload: dict[str, Any] = {
        "questions": _read_json(args.questions_file),
        "path": str(args.path.expanduser().resolve()),
        "format": args.format,
        "chunk_tokens": args.chunk_tokens,
        "top_k": args.top_k,
        "model": args.model,
    }
    if args.field:
        payload["field"] = args.field
    if args.filter:
        payload["filter"] = json.loads(args.filter)
    return _call(
        settings, "POST", "/v1/batch", payload, timeout=settings.batch.timeout_seconds + 30
    )


def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    return _call(settings, "GET", "/v1/status", None, timeout=10)


def cmd_connect(args: argparse.Namespace, settings: Settings) -> int:
    url = f"http://127.0.0.1:{settings.port}/mcp"
    if args.harness == "claude-code":
        # claude mcp add reads everything after --header as another header, so it goes last.
        argv = ["claude", "mcp", "add", "--transport", "http", "--scope", "user", "local-jev", url]
        shown = " ".join(argv)
        if settings.token:
            argv += ["--header", f"Authorization: Bearer {settings.token}"]
            shown += ' --header "Authorization: Bearer $LOCAL_JEV_TOKEN"'
        print(shown)
        if not args.apply:
            return 0
        if shutil.which("claude") is None:
            print("the claude command is not on PATH", file=sys.stderr)
            return 1
        if not _confirm("Add local-jev to Claude Code for your user?"):
            print("cancelled")
            return 1
        return subprocess.run(argv, check=False).returncode
    if args.apply:
        print("--apply is only supported with --harness claude-code", file=sys.stderr)
        return 1
    if args.harness == "codex":
        lines = ["# add to ~/.codex/config.toml", "[mcp_servers.local-jev]", f'url = "{url}"']
        if settings.token:
            lines.append('bearer_token_env_var = "LOCAL_JEV_TOKEN"')
        print("\n".join(lines))
        return 0
    server: dict[str, Any] = (
        {"url": url} if args.harness == "cursor" else {"type": "http", "url": url}
    )
    if settings.token:
        server["headers"] = {"Authorization": "Bearer <your LOCAL_JEV_TOKEN>"}
    where = "~/.cursor/mcp.json" if args.harness == "cursor" else "your agent's MCP configuration"
    print(f"// add to {where}")
    print(json.dumps({"mcpServers": {"local-jev": server}}, indent=2))
    return 0


def cmd_install_agent(args: argparse.Namespace, settings: Settings) -> int:
    program = _serve_command()
    env = _agent_environment()
    log_dir = settings.log_dir.resolve()
    shown = {key: ("***" if key == "LOCAL_JEV_TOKEN" else value) for key, value in env.items()}
    plist = launchd.plist_path()
    print(f"This writes {plist} (mode 600) and loads it with launchctl:\n")
    print(launchd.render_plist(program, log_dir, shown).decode())
    if not (args.yes or _confirm("Install and start the login agent?")):
        print("cancelled")
        return 1
    try:
        path = launchd.install(program, log_dir, env)
    except subprocess.CalledProcessError as exc:
        # launchctl printed its own reason. The plist is on disk and the old agent is stopped.
        print(
            f"launchctl could not load the login agent (exit status {exc.returncode}).\n"
            f"The plist is written at {plist}, so the agent will load at your next login."
            " An agent that was running before has been stopped.\n"
            f"To load it now: launchctl bootstrap gui/$UID {plist}\n"
            "To remove it: local-jev uninstall-agent",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        # Raised while writing the plist or starting launchctl, both before anything was stopped.
        print(
            f"could not install the login agent: {exc}\n"
            "the agent that is running now, if any, was left alone",
            file=sys.stderr,
        )
        return 1
    print(f"installed {path}; local-jev now starts at login (logs in {log_dir})")
    return 0


def cmd_uninstall_agent(args: argparse.Namespace, settings: Settings) -> int:
    if not (args.yes or _confirm("Stop local-jev and remove the login agent?")):
        print("cancelled")
        return 1
    removed = launchd.uninstall()
    print("removed the login agent" if removed else "no login agent was installed")
    return 0


def _call(settings: Settings, method: str, path: str, payload: Any, *, timeout: float) -> int:
    url = f"http://127.0.0.1:{settings.port}{path}"
    headers = {"Authorization": f"Bearer {settings.token}"} if settings.token else {}
    if settings.token and not settings.token.isascii():
        # httpx2 cannot encode it as a header, and its error names the character that failed.
        print(BAD_TOKEN, file=sys.stderr)
        return 1
    try:
        status, body = _request(method, url, payload=payload, headers=headers, timeout=timeout)
    except httpx2.ConnectError:
        print(NOT_RUNNING, file=sys.stderr)
        return 1
    except httpx2.LocalProtocolError:
        # The error repeats the whole header value, and the header holds the token.
        print(BAD_TOKEN, file=sys.stderr)
        return 1
    except httpx2.HTTPError as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(body, indent=2, ensure_ascii=False))
    return 0 if status == 200 else 1


def _request(
    method: str, url: str, *, payload: Any, headers: dict[str, str], timeout: float
) -> tuple[int, Any]:
    # trust_env=False keeps the token and the body off any proxy named in the environment.
    response = httpx2.request(
        method, url, json=payload, headers=headers, timeout=timeout, trust_env=False
    )
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {
            "error": {"code": "bad_response", "message": response.text[:500], "hint": ""}
        }


def _agent_environment() -> dict[str, str]:
    """The settings the login agent needs, copied from this environment.

    launchd does not read the shell profile, so a setting made there would reach `status` and
    `connect` but not the server. Paths are made absolute because the agent starts in `/`. Empty
    values are skipped, as load_settings skips them.
    """
    env: dict[str, str] = {}
    for name in (*ENV_VARS, *HF_ENV_VARS):
        value = os.environ.get(name)
        if not value:
            continue
        if name in PATH_ENV_VARS or name in HF_ENV_VARS:
            value = os.path.expanduser(value)
            if name in HF_ENV_VARS:
                value = os.path.expandvars(value)  # huggingface_hub expands $VAR in these two
            value = str(Path(value).resolve())
        env[name] = value
    return env


def _read_state(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix.lower() == ".json" else text


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _serve_command() -> list[str]:
    executable = shutil.which("local-jev")
    return [executable, "serve"] if executable else [sys.executable, "-m", "local_jev", "serve"]


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "server.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()])
