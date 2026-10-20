from pathlib import Path

import pytest

from local_jev import config as config_module
from local_jev.config import LAYA_REVISION, ConfigError, Settings, load_settings


def test_defaults_when_the_file_is_missing(tmp_path):
    settings = load_settings(tmp_path / "missing.toml", env={})
    assert settings == Settings()
    assert settings.port == 7711
    assert settings.backend == "laya-mlx"
    assert settings.noul_mode == "ab_labels"
    assert settings.laya.revision == LAYA_REVISION
    assert settings.laya.prefetch == ("english", "multilingual")
    assert settings.batch.max_items == 5000


def test_file_values_are_read(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'port = 7800\nnoul_mode = "native"\nlog_inputs = true\n\n'
        '[laya]\nprefetch = ["english"]\n\n'
        '[batch]\nallowed_roots = ["~/data"]\nmax_items = 10\n'
    )
    settings = load_settings(config, env={})
    assert settings.port == 7800
    assert settings.noul_mode == "native"
    assert settings.log_inputs is True
    assert settings.laya.prefetch == ("english",)
    assert settings.batch.allowed_roots == (Path("~/data").expanduser(),)
    assert settings.batch.max_items == 10


def test_environment_overrides_the_file(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("port = 7800\n")
    settings = load_settings(
        config,
        env={
            "LOCAL_JEV_PORT": "7900",
            "LOCAL_JEV_TOKEN": "s3cret",
            "LOCAL_JEV_LOG_INPUTS": "yes",
            "LOCAL_JEV_IDLE_UNLOAD_MINUTES": "0.5",
        },
    )
    assert settings.port == 7900
    assert settings.token == "s3cret"
    assert settings.log_inputs is True
    assert settings.idle_unload_minutes == 0.5


def test_config_path_can_come_from_the_environment(tmp_path):
    config = tmp_path / "other.toml"
    config.write_text("port = 7801\n")
    assert load_settings(env={"LOCAL_JEV_CONFIG": str(config)}).port == 7801


def test_remote_url_from_the_environment(tmp_path):
    settings = load_settings(
        tmp_path / "missing.toml",
        env={
            "LOCAL_JEV_BACKEND": "remote",
            "LOCAL_JEV_REMOTE_URL": "http://127.0.0.1:8009/v1/systemone",
        },
    )
    assert settings.backend == "remote"
    assert settings.remote.url == "http://127.0.0.1:8009/v1/systemone"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("port = 80\n", "port"),
        ('port = "seven"\n', "port"),
        ('backend = "gpu"\n', "backend"),
        ('noul_mode = "ab_choice"\n', "noul_mode"),
        ('colour = "blue"\n', "unknown setting"),
        ('[laya]\nrevision = "main"\n', "revision"),
        ('[laya]\nprefetch = ["klingon"]\n', "prefetch"),
        ('backend = "remote"\n', "remote.url"),
        ('token = "x"\n', "unknown setting"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, text, message):
    config = tmp_path / "config.toml"
    config.write_text(text)
    with pytest.raises(ConfigError, match=message):
        load_settings(config, env={})


class _Reads(dict):
    """An environment that remembers which names were looked up."""

    def __init__(self, data):
        super().__init__(data)
        self.names = set()

    def get(self, key, default=None):
        self.names.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.names.add(key)
        return super().__getitem__(key)


def test_env_vars_lists_every_variable_load_settings_reads(tmp_path):
    env = _Reads({"LOCAL_JEV_CONFIG": str(tmp_path / "missing.toml")})
    load_settings(env=env)
    assert set(config_module.ENV_VARS) == env.names
    assert len(set(config_module.ENV_VARS)) == len(config_module.ENV_VARS)


def test_path_env_vars_are_the_ones_that_hold_a_path():
    assert set(config_module.PATH_ENV_VARS) == {
        "LOCAL_JEV_CONFIG",
        "LOCAL_JEV_LOG_DIR",
        "LOCAL_JEV_RESULTS_DIR",
    }
    assert set(config_module.PATH_ENV_VARS) <= set(config_module.ENV_VARS)


# Final review: relative paths, the token and remote.url are checked when the settings load.


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ('[batch]\nallowed_roots = ["."]\n', "batch.allowed_roots"),
        ('[batch]\nallowed_roots = ["~/data", "data"]\n', "batch.allowed_roots"),
        ('log_dir = "logs"\n', "log_dir"),
        ('results_dir = "./results"\n', "results_dir"),
    ],
)
def test_a_relative_path_in_the_file_is_refused(tmp_path, monkeypatch, text, key):
    # The login agent starts in /, where "." would open the whole disk to decide_batch.
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text(text)
    with pytest.raises(ConfigError) as info:
        load_settings(config, env={})
    assert str(info.value) == f"{key} must be an absolute path (a leading ~ is allowed)"


def test_absolute_paths_and_paths_from_the_home_folder_are_accepted(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        f'log_dir = "~/logs"\nresults_dir = "{tmp_path / "results"}"\n\n'
        f'[batch]\nallowed_roots = ["~", "{tmp_path}"]\n'
    )
    settings = load_settings(config, env={})
    assert settings.log_dir == Path.home() / "logs"
    assert settings.results_dir == tmp_path / "results"
    assert settings.batch.allowed_roots == (Path.home(), tmp_path)


@pytest.mark.parametrize(
    ("var", "name"), [("LOCAL_JEV_LOG_DIR", "log_dir"), ("LOCAL_JEV_RESULTS_DIR", "results_dir")]
)
def test_a_relative_path_in_the_environment_is_taken_from_the_current_folder(
    tmp_path, monkeypatch, var, name
):
    monkeypatch.chdir(tmp_path)
    settings = load_settings(tmp_path / "missing.toml", env={var: "state/here"})
    assert getattr(settings, name) == tmp_path / "state" / "here"


NO_SUCH_USER = "~local-jev-no-such-user"


@pytest.mark.parametrize(
    ("text", "env", "where"),
    [
        (f'log_dir = "{NO_SUCH_USER}/logs"\n', {}, "log_dir"),
        (f'[batch]\nallowed_roots = ["{NO_SUCH_USER}"]\n', {}, "batch.allowed_roots"),
        ("", {"LOCAL_JEV_RESULTS_DIR": f"{NO_SUCH_USER}/results"}, "LOCAL_JEV_RESULTS_DIR"),
    ],
    ids=["file path", "file path list", "environment path"],
)
def test_a_path_from_the_home_folder_of_an_unknown_user_is_refused(tmp_path, text, env, where):
    # Path.expanduser raises RuntimeError for it, which no command catches.
    config = tmp_path / "config.toml"
    config.write_text(text)
    with pytest.raises(ConfigError) as info:
        load_settings(config, env=env)
    assert str(info.value) == f"{where} starts with {NO_SUCH_USER}, a user this Mac does not have"


def test_a_config_path_in_the_home_folder_of_an_unknown_user_is_refused():
    with pytest.raises(ConfigError) as info:
        load_settings(env={"LOCAL_JEV_CONFIG": f"{NO_SUCH_USER}/config.toml"})
    assert str(info.value) == (
        f"LOCAL_JEV_CONFIG starts with {NO_SUCH_USER}, a user this Mac does not have"
    )


def test_the_token_stays_out_of_the_settings_repr(tmp_path):
    settings = load_settings(tmp_path / "missing.toml", env={"LOCAL_JEV_TOKEN": "probe-token-8"})
    assert settings.token == "probe-token-8"
    assert "probe-token-8" not in repr(settings)
    assert "probe-token-8" not in str(settings)


@pytest.mark.parametrize(
    "token",
    [
        "probe-token-9 ",
        " probe-token-9",
        "probe-token-9\n",
        "probe\ttoken-9",
        "probe-tökén-9",
        "probe-token-9\x7f",
        b"probe-token-9\xff".decode("utf-8", "surrogateescape"),
    ],
    ids=["trailing-space", "leading-space", "newline", "tab", "non-ascii", "delete", "bad-utf8"],
)
def test_a_token_no_request_can_carry_is_refused_without_showing_it(tmp_path, token):
    # Every request would get a 401, because the HTTP stack cannot send it as a header value.
    with pytest.raises(ConfigError) as info:
        load_settings(tmp_path / "missing.toml", env={"LOCAL_JEV_TOKEN": token})
    message = str(info.value)
    assert message == ("LOCAL_JEV_TOKEN must be printable ASCII with no whitespace at either end")
    assert "probe" not in message


REMOTE = {"LOCAL_JEV_BACKEND": "remote"}


NOT_A_URL = (
    "remote.url (or LOCAL_JEV_REMOTE_URL) must be an http or https URL with a host, "
    "such as http://127.0.0.1:8009/v1/systemone"
)
USERINFO = (
    "remote.url (or LOCAL_JEV_REMOTE_URL) must not hold a user name or password; "
    "put the API key in the variable that remote.api_key_env names"
)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://127.0.0.1:8009/v1/systemone",
        "127.0.0.1:8009/v1/systemone",
        "/v1/systemone",
        " http://127.0.0.1:8009/v1/systemone",
        "http:///v1/systemone",
        "http://127.0.0.1:port/v1/systemone",
        "http://[::1/v1/systemone",
        "http://127.0.0.1\n:8009/v1/systemone",
    ],
)
def test_a_remote_url_the_engine_cannot_post_to_is_refused(tmp_path, url):
    with pytest.raises(ConfigError) as info:
        load_settings(tmp_path / "missing.toml", env={**REMOTE, "LOCAL_JEV_REMOTE_URL": url})
    assert str(info.value) == NOT_A_URL
    assert info.value.__context__ is None


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://kev:pa55word@127.0.0.1:8009/v1/systemone", USERINFO),
        ("https://pa55word@jev.example/v1/systemone", USERINFO),
        ("http://kev:pa55word@127.0.0.1:port/v1/systemone", NOT_A_URL),
    ],
)
def test_a_remote_url_with_a_user_name_or_password_is_refused_without_showing_it(
    tmp_path, url, message
):
    # httpx2 sends the userinfo as Basic auth, and status() would show it.
    config = tmp_path / "config.toml"
    config.write_text(f'backend = "remote"\n\n[remote]\nurl = "{url}"\n')
    with pytest.raises(ConfigError) as info:
        load_settings(config, env={})
    assert str(info.value) == message
    assert info.value.__context__ is None


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8009/v1/systemone",
        "http://localhost:8009/v1/systemone",
        "http://[::1]:8009/v1/systemone",
        "HTTPS://jev.example/v1/systemone",
    ],
)
def test_a_usable_remote_url_is_accepted(tmp_path, url):
    settings = load_settings(tmp_path / "missing.toml", env={**REMOTE, "LOCAL_JEV_REMOTE_URL": url})
    assert settings.remote.url == url
