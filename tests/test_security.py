import os

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from local_jev.errors import ServiceError
from local_jev.security import FilePolicy, SecurityMiddleware, is_loopback_origin


def make_app(token=None):
    async def ok(request):
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[Route("/health", ok), Route("/v1/status", ok)],
        middleware=[Middleware(SecurityMiddleware, port=7711, token=token)],
    )


def get(path="/v1/status", host="127.0.0.1:7711", token=None, **headers):
    with TestClient(make_app(token), base_url=f"http://{host}") as client:
        return client.get(path, headers=headers)


@pytest.mark.parametrize("host", ["127.0.0.1:7711", "localhost:7711", "[::1]:7711"])
def test_loopback_hosts_are_served(host):
    assert get(host=host).status_code == 200


@pytest.mark.parametrize("host", ["evil.example:7711", "127.0.0.1:9999", "192.168.1.5:7711"])
def test_other_hosts_are_misdirected(host):
    response = get(host=host)
    assert response.status_code == 421
    assert response.json()["error"]["code"] == "misdirected"


@pytest.mark.parametrize(
    ("origin", "status"),
    [
        ("http://localhost:3000", 200),
        ("http://127.0.0.1:7711", 200),
        ("http://evil.example", 403),
        ("null", 403),
        ("https://localhost:3000", 403),
        ("http://127.0.0.1", 200),
        ("http://localhost:7711", 200),
        ("http://[::1]:7711", 200),
        ("HTTP://LOCALHOST:7711", 200),
        ("http://evil.example@localhost", 403),
        ("http://localhost:3000, http://evil.example", 403),
        ("http://localhost:abc", 403),
        ("http://localhost/x?y", 403),
        ("http://[", 403),
        ("http://[127.0.0.1]", 403),
        ("http://]", 403),
        ("http://[x]", 403),
        ("http://[::1", 403),
        ("http://localhost]", 403),
    ],
)
def test_web_page_origins_are_refused(origin, status):
    response = get(Origin=origin)
    assert response.status_code == status
    if status == 403:
        assert response.json()["error"]["code"] == "forbidden_origin"


def test_the_token_is_required_except_on_health():
    assert get(token="s3cret").status_code == 401
    assert get(token="s3cret", Authorization="Bearer wrong").status_code == 401
    assert get(token="s3cret", Authorization="Bearer s3cret").status_code == 200
    assert get("/health", token="s3cret").status_code == 200


def test_is_loopback_origin():
    assert is_loopback_origin("http://[::1]:8080")
    assert not is_loopback_origin("http://localhost.evil.example")
    assert not is_loopback_origin("http://[127.0.0.1]")


@pytest.mark.parametrize(
    ("header", "values", "status"),
    [
        ("Host", ["evil.example:7711", "127.0.0.1:7711"], 421),
        ("Host", ["127.0.0.1:7711", "evil.example:7711"], 421),
        ("Origin", ["http://evil.example", "http://localhost:3000"], 403),
        ("Origin", ["http://localhost:3000", "http://evil.example"], 403),
    ],
)
def test_a_repeated_host_or_origin_cannot_hide_a_foreign_one(header, values, status):
    with TestClient(make_app(), base_url="http://127.0.0.1:7711") as client:
        response = client.get("/v1/status", headers=[(header, value) for value in values])
    assert response.status_code == status


def test_a_token_whose_bytes_are_not_utf8_fails_closed():
    token = b"s3cret\xff".decode("utf-8", "surrogateescape")  # how os.environ holds such bytes
    assert get(token=token, Authorization=b"Bearer s3cret\xff").status_code == 401


def test_a_second_authorization_header_is_refused():
    headers = [("Authorization", "Bearer wrong"), ("Authorization", "Bearer s3cret")]
    with TestClient(make_app("s3cret"), base_url="http://127.0.0.1:7711") as client:
        assert client.get("/v1/status", headers=headers).status_code == 401


@pytest.fixture
def policy(tmp_path):
    return FilePolicy((tmp_path,), max_bytes=10)


def test_a_file_inside_the_allowed_root_is_accepted(policy, tmp_path):
    path = tmp_path / "items.txt"
    path.write_text("hello")
    assert policy.check(str(path)) == path.resolve()


def test_relative_paths_are_rejected(policy):
    with pytest.raises(ServiceError, match="must be absolute"):
        policy.check("items.txt")


def test_missing_files_are_rejected(policy, tmp_path):
    with pytest.raises(ServiceError, match="no such file"):
        policy.check(str(tmp_path / "missing.txt"))


@pytest.mark.parametrize(
    "name", ["items\x00.txt", "\ud800.txt"], ids=["nul-byte", "lone-surrogate"]
)
def test_paths_the_os_cannot_represent_are_rejected(policy, tmp_path, name):
    with pytest.raises(ServiceError, match="no such file") as info:
        policy.check(f"{tmp_path}/{name}")
    assert info.value.status == 422


def test_files_outside_the_allowed_roots_are_rejected(policy, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "x.txt"
    outside.write_text("hi")
    with pytest.raises(ServiceError, match="outside the allowed folders"):
        policy.check(str(outside))


def test_symlinks_cannot_escape_the_allowed_roots(policy, tmp_path, tmp_path_factory):
    secret = tmp_path_factory.mktemp("secret") / "x.txt"
    secret.write_text("hi")
    link = tmp_path / "link.txt"
    os.symlink(secret, link)
    with pytest.raises(ServiceError, match="outside the allowed folders"):
        policy.check(str(link))


def test_directories_are_rejected(policy, tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    with pytest.raises(ServiceError, match="not a regular file"):
        policy.check(str(folder))


def test_large_files_are_rejected(policy, tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("x" * 11)
    with pytest.raises(ServiceError) as info:
        policy.check(str(path))
    assert info.value.status == 413
