import os
import plistlib
import stat
from pathlib import Path

import pytest

from local_jev import launchd

LABEL = "io.github.andresrubio.local-jev"


def _refuse(*args, **kwargs):
    pytest.fail("a launchd test reached the real launchctl; pass run= to install and uninstall")


@pytest.fixture(autouse=True)
def no_launchctl(monkeypatch, tmp_path):
    """Keep every test away from launchctl and from the real ~/Library/LaunchAgents."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(launchd.subprocess, "run", _refuse)
    # install and uninstall bind run=subprocess.run when launchd is imported, so patching
    # subprocess.run does not reach a call that leaves out run=. Replace those defaults as well.
    monkeypatch.setitem(launchd.install.__kwdefaults__, "run", _refuse)
    monkeypatch.setitem(launchd.uninstall.__kwdefaults__, "run", _refuse)


def test_render_plist_contents(tmp_path):
    data = plistlib.loads(
        launchd.render_plist(["/bin/local-jev", "serve"], tmp_path, {"LOCAL_JEV_TOKEN": "t"})
    )
    assert data["Label"] == "io.github.andresrubio.local-jev"
    assert data["ProgramArguments"] == ["/bin/local-jev", "serve"]
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] is True
    assert data["StandardErrorPath"] == str(tmp_path / "launchd.err.log")
    assert data["EnvironmentVariables"] == {"LOCAL_JEV_TOKEN": "t"}


def test_render_plist_without_environment(tmp_path):
    assert "EnvironmentVariables" not in plistlib.loads(launchd.render_plist(["x"], tmp_path, {}))


def test_install_writes_a_private_plist_and_bootstraps_it(tmp_path):
    calls = []
    path = launchd.install(
        ["/bin/local-jev", "serve"],
        tmp_path / "logs",
        {},
        home=tmp_path,
        run=lambda argv, **kw: calls.append(argv),
    )
    assert path == tmp_path / "Library" / "LaunchAgents" / "io.github.andresrubio.local-jev.plist"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert calls[0][:2] == ["launchctl", "bootout"]
    assert calls[1][:2] == ["launchctl", "bootstrap"]
    assert calls[1][-1] == str(path)


def test_uninstall_removes_the_plist(tmp_path):
    launchd.install(["x"], tmp_path / "logs", {}, home=tmp_path, run=lambda argv, **kw: None)
    calls = []
    assert launchd.uninstall(home=tmp_path, run=lambda argv, **kw: calls.append(argv)) is True
    assert calls[0][:2] == ["launchctl", "bootout"]
    assert launchd.uninstall(home=tmp_path, run=lambda argv, **kw: None) is False


def test_render_plist_holds_exactly_the_keys_launchd_needs(tmp_path):
    rendered = launchd.render_plist(["/bin/local-jev", "serve"], tmp_path, {"LOCAL_JEV_TOKEN": "t"})
    assert plistlib.loads(rendered) == {
        "Label": LABEL,
        "ProgramArguments": ["/bin/local-jev", "serve"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": str(tmp_path / "launchd.out.log"),
        "StandardErrorPath": str(tmp_path / "launchd.err.log"),
        "EnvironmentVariables": {"LOCAL_JEV_TOKEN": "t"},
    }


def test_plist_path_defaults_to_the_home_folder():
    assert (
        launchd.plist_path()
        == Path(os.environ["HOME"]) / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    )


def test_install_makes_its_folders_and_tightens_a_looser_file(tmp_path):
    path = launchd.plist_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * 5000)
    path.chmod(0o644)
    logs = tmp_path / "state" / "logs"
    launchd.install(
        ["x"], logs, {"LOCAL_JEV_TOKEN": "t"}, home=tmp_path, run=lambda argv, **kw: None
    )
    assert logs.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert plistlib.loads(path.read_bytes())["EnvironmentVariables"] == {"LOCAL_JEV_TOKEN": "t"}


def test_the_plist_is_created_private_not_only_tightened_afterwards(monkeypatch, tmp_path):
    # The file holds the token. With chmod disabled, the mode it was created with must already be 600.
    monkeypatch.setattr(Path, "chmod", lambda self, mode: None)
    path = launchd.install(
        ["x"],
        tmp_path / "logs",
        {"LOCAL_JEV_TOKEN": "t"},
        home=tmp_path,
        run=lambda argv, **kw: None,
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_install_boots_out_then_bootstraps_a_plist_that_is_already_written(tmp_path):
    calls = []
    path = launchd.plist_path(tmp_path)

    def run(argv, **kwargs):
        calls.append((argv, kwargs, path.exists()))

    launchd.install(["x"], tmp_path / "logs", {}, home=tmp_path, run=run)
    domain = f"gui/{os.getuid()}"
    assert calls == [
        (
            ["launchctl", "bootout", f"{domain}/{LABEL}"],
            {"check": False, "capture_output": True},
            False,
        ),
        (["launchctl", "bootstrap", domain, str(path)], {"check": True}, True),
    ]


def test_uninstall_boots_the_agent_out_even_when_the_plist_is_gone(tmp_path):
    launchd.install(["x"], tmp_path / "logs", {}, home=tmp_path, run=lambda argv, **kw: None)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))

    assert launchd.uninstall(home=tmp_path, run=run) is True
    assert launchd.uninstall(home=tmp_path, run=run) is False
    bootout = (
        ["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"],
        {"check": False, "capture_output": True},
    )
    assert calls == [bootout, bootout]
