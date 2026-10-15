from pathlib import Path

import pytest

from local_jev.config import LAYA_REVISION, ConfigError, Settings, load_settings


def test_defaults_when_the_file_is_missing(tmp_path):
    settings = load_settings(tmp_path / "missing.toml", env={})
    assert settings == Settings()
    assert settings.port == 7711
    assert settings.backend == "laya-mlx"
    assert settings.noul_mode == "native"
    assert settings.laya.revision == LAYA_REVISION
    assert settings.laya.prefetch == ("english", "multilingual")
    assert settings.batch.max_items == 5000


def test_file_values_are_read(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'port = 7800\nnoul_mode = "ab_labels"\nlog_inputs = true\n\n'
        '[laya]\nprefetch = ["english"]\n\n'
        '[batch]\nallowed_roots = ["~/data"]\nmax_items = 10\n'
    )
    settings = load_settings(config, env={})
    assert settings.port == 7800
    assert settings.noul_mode == "ab_labels"
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
