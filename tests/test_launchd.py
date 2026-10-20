import errno
import os
import plistlib
import stat
import subprocess
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
    # The file holds the token. With every chmod disabled, the mode it was created with must be 600.
    monkeypatch.setattr(Path, "chmod", lambda self, mode: None)
    monkeypatch.setattr(os, "fchmod", lambda fd, mode: None)
    path = launchd.install(
        ["x"],
        tmp_path / "logs",
        {"LOCAL_JEV_TOKEN": "t"},
        home=tmp_path,
        run=lambda argv, **kw: None,
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_install_writes_the_plist_before_it_boots_the_old_agent_out(tmp_path):
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
            True,
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


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write into a read-only folder")
def test_a_folder_that_cannot_be_written_stops_the_install_before_the_old_agent_is_booted_out(
    tmp_path,
):
    path = launchd.plist_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.parent.chmod(0o500)
    calls = []
    try:
        with pytest.raises(PermissionError):
            launchd.install(
                ["x"],
                tmp_path / "logs",
                {},
                home=tmp_path,
                run=lambda argv, **kw: calls.append(argv),
            )
    finally:
        path.parent.chmod(0o700)
    assert calls == []
    assert list(path.parent.iterdir()) == []


def test_a_failed_replace_keeps_the_old_plist_removes_the_temporary_file_and_boots_nothing_out(
    monkeypatch, tmp_path
):
    path = launchd.plist_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old plist")
    calls = []

    def disk_full(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "replace", disk_full)
    with pytest.raises(OSError, match="No space left"):
        launchd.install(
            ["x"],
            tmp_path / "logs",
            {"LOCAL_JEV_TOKEN": "t"},
            home=tmp_path,
            run=lambda argv, **kw: calls.append(argv),
        )
    assert calls == []
    assert path.read_bytes() == b"old plist"
    assert [item.name for item in path.parent.iterdir()] == [path.name]


def test_the_plist_replaces_the_old_one_in_one_step_from_a_private_complete_file(
    monkeypatch, tmp_path
):
    seen = {}
    real_replace = os.replace

    def replace(src, dst):
        seen.update(
            src=Path(src),
            dst=Path(dst),
            mode=stat.S_IMODE(os.stat(src).st_mode),
            label=plistlib.loads(Path(src).read_bytes())["Label"],
        )
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    path = launchd.install(
        ["x"],
        tmp_path / "logs",
        {"LOCAL_JEV_TOKEN": "t"},
        home=tmp_path,
        run=lambda argv, **kw: None,
    )
    assert seen["dst"] == path
    assert seen["src"].parent == path.parent
    assert seen["src"] != path
    assert seen["src"].suffix == ".tmp"  # launchd would load a leftover *.plist as a job
    assert (seen["mode"], seen["label"]) == (0o600, LABEL)
    assert [item.name for item in path.parent.iterdir()] == [path.name]


def test_a_failed_bootstrap_raises_and_leaves_the_new_plist_on_disk(tmp_path):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv[1])
        if argv[1] == "bootstrap":
            raise subprocess.CalledProcessError(5, argv)

    with pytest.raises(subprocess.CalledProcessError) as info:
        launchd.install(["x"], tmp_path / "logs", {"LOCAL_JEV_TOKEN": "t"}, home=tmp_path, run=run)
    path = launchd.plist_path(tmp_path)
    assert info.value.returncode == 5
    assert calls == ["bootout", "bootstrap"]
    assert plistlib.loads(path.read_bytes())["EnvironmentVariables"] == {"LOCAL_JEV_TOKEN": "t"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_plist_that_cannot_be_rendered_changes_nothing(tmp_path):
    calls = []
    with pytest.raises(ValueError):  # a plist cannot hold a control character
        launchd.install(
            ["x"],
            tmp_path / "logs",
            {"LOCAL_JEV_TOKEN": "a\x01b"},
            home=tmp_path,
            run=lambda argv, **kw: calls.append(argv),
        )
    assert calls == []
    assert list(tmp_path.iterdir()) == []
