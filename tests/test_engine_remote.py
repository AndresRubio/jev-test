import ipaddress
import json
import ssl
import threading
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx2
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from local_jev.engines.remote import ENDPOINT_HINT, RemoteJevEngine
from local_jev.errors import ModelUnavailable

SECRET = "sekrit-token"
URL = "http://127.0.0.1:8009/v1/systemone"
QUESTIONS = {"refund": {"type": "noul", "instructions": "Money back?"}}
ANSWER = {
    "model": "kev-4b",
    "answers": {"refund": {"type": "noul", "noul": 0.8}},
    "usage": {"input_tokens": 9, "output_tokens": 0},
}


def engine_with(handler, api_key=None):
    return RemoteJevEngine(URL, api_key=api_key, transport=httpx2.MockTransport(handler))


def test_posts_a_jev_payload_and_returns_the_answers():
    seen = {}

    def handler(request):
        seen.update(
            url=str(request.url),
            auth=request.headers.get("authorization"),
            body=json.loads(request.content),
        )
        return httpx2.Response(200, json=ANSWER)

    result = engine_with(handler, api_key="k").decide("refund please", QUESTIONS, "auto")
    assert seen == {
        "url": URL,
        "auth": "Bearer k",
        "body": {"model": "jev-latest", "state": "refund please", "questions": QUESTIONS},
    }
    assert result.model == "kev-4b"
    assert result.answers["refund"]["noul"] == 0.8
    assert result.usage["input_tokens"] == 9


def test_an_explicit_model_is_forwarded():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx2.Response(200, json=ANSWER)

    engine_with(handler).decide("x", QUESTIONS, "kev-0.8b")
    assert seen["model"] == "kev-0.8b"


def test_validation_errors_become_value_errors():
    engine = engine_with(lambda request: httpx2.Response(422, json={"detail": "bad criteria"}))
    with pytest.raises(ValueError, match="bad criteria"):
        engine.decide("x", QUESTIONS, "auto")


@pytest.mark.parametrize(("status", "fragment"), [(401, "API key"), (500, "HTTP 500")])
def test_backend_failures_are_model_unavailable(status, fragment):
    engine = engine_with(lambda request: httpx2.Response(status, text="nope"))
    with pytest.raises(ModelUnavailable, match=fragment):
        engine.decide("x", QUESTIONS, "auto")


def test_an_unreachable_backend_is_model_unavailable():
    def handler(request):
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(ModelUnavailable, match="unreachable"):
        engine_with(handler).decide("x", QUESTIONS, "auto")


def test_an_unexpected_payload_is_model_unavailable():
    engine = engine_with(lambda request: httpx2.Response(200, json={"hello": "world"}))
    with pytest.raises(ModelUnavailable, match="unexpected payload"):
        engine.decide("x", QUESTIONS, "auto")


def test_status_and_token_estimate():
    engine = engine_with(lambda request: httpx2.Response(200, json=ANSWER))
    assert engine.status() == {"backend": "remote", "url": URL, "api_key": False}
    assert engine.count_tokens("x" * 40, "auto") == 10
    assert engine.is_loaded() is False


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param([1, 2], id="list"),
        pytest.param("lots", id="string"),
        pytest.param([], id="empty-list"),
    ],
)
def test_a_usage_that_is_not_an_object_is_model_unavailable(usage):
    engine = engine_with(lambda request: httpx2.Response(200, json={**ANSWER, "usage": usage}))
    with pytest.raises(ModelUnavailable, match="unexpected payload") as raised:
        engine.decide("x", QUESTIONS, "auto")
    assert raised.value.hint == ENDPOINT_HINT


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({key: value for key, value in ANSWER.items() if key != "usage"}, id="missing"),
        pytest.param({**ANSWER, "usage": None}, id="null"),
    ],
)
def test_a_missing_or_null_usage_becomes_an_empty_dict(payload):
    engine = engine_with(lambda request: httpx2.Response(200, json=payload))
    assert engine.decide("x", QUESTIONS, "auto").usage == {}


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(f"{SECRET}\n", id="newline"),
        pytest.param(f"{SECRET} ", id="trailing-space"),
        pytest.param(f"{SECRET}\r\nX-Injected: 1", id="header-injection"),
        pytest.param("sekrit-tökën", id="non-ascii"),
    ],
)
def test_a_key_that_cannot_go_in_a_header_is_refused_and_never_echoed(key):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx2.Response(200, json=ANSWER)

    with pytest.raises(ModelUnavailable, match="API key") as raised:
        engine_with(handler, api_key=key).decide("x", QUESTIONS, "auto")
    error = raised.value
    assert sent == []
    assert "sekrit" not in f"{error} {error.hint}"
    assert error.__cause__ is None
    assert error.__context__ is None


def test_status_and_repr_never_show_the_api_key():
    engine = engine_with(lambda request: httpx2.Response(200, json=ANSWER), api_key=SECRET)
    assert engine.status() == {"backend": "remote", "url": URL, "api_key": True}
    assert SECRET not in repr(engine)


# Final review: a proxy named in the environment or the system settings, with the user's decision
# of 2026-10-20 to ignore it for a loopback remote.url and keep it for every other host.


@contextmanager
def recording_server(certificate=None):
    """A server on 127.0.0.1 that answers every POST with ANSWER, over https with a certificate.

    Yields its port and the (request line, Authorization) pairs it saw. A proxy sees the full URL
    in the request line. certificate is a (certificate file, key file) pair.
    """
    seen = []
    reply = json.dumps(ANSWER).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            seen.append((self.requestline, self.headers.get("Authorization")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if certificate is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(*certificate)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def use_proxy(monkeypatch, port):
    """Name the proxy in every variable httpx2 reads, with no host exempted."""
    for name in ("NO_PROXY", "no_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, f"http://127.0.0.1:{port}")


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_a_loopback_backend_is_never_reached_through_a_proxy(monkeypatch, host):
    # Through the proxy, a local Kev would hand every state and the API key over in clear text.
    with recording_server() as (port, target), recording_server() as (proxy_port, proxy):
        use_proxy(monkeypatch, proxy_port)
        engine = RemoteJevEngine(f"http://{host}:{port}/v1/systemone", api_key=SECRET)
        assert engine.decide("x", QUESTIONS, "auto").answers == ANSWER["answers"]
    assert proxy == []
    assert target == [("POST /v1/systemone HTTP/1.1", f"Bearer {SECRET}")]


def test_any_other_backend_still_goes_through_the_configured_proxy(monkeypatch):
    with recording_server() as (proxy_port, proxy):
        use_proxy(monkeypatch, proxy_port)
        engine = RemoteJevEngine("http://kev.example:8009/v1/systemone", api_key=SECRET)
        assert engine.decide("x", QUESTIONS, "auto").answers == ANSWER["answers"]
    assert proxy == [("POST http://kev.example:8009/v1/systemone HTTP/1.1", f"Bearer {SECRET}")]


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.8.9.10", True),
        ("[::1]", True),
        ("localhost", True),
        ("LocalHost", True),
        ("10.0.0.1", False),
        ("128.0.0.1", False),
        ("[::2]", False),
        ("kev.example", False),
        ("localhost.example", False),
    ],
)
def test_the_environment_is_ignored_only_for_a_loopback_host(host, loopback):
    # 127.0.0.0/8, ::1 and localhost; the two tests above send real requests for two of them.
    engine = RemoteJevEngine(f"http://{host}:8009/v1/systemone")
    assert engine._client.trust_env is not loopback


@pytest.fixture
def loopback_certificate(tmp_path):
    """A self-signed certificate for 127.0.0.1 and its key, as two PEM files."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_file, key_file = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_file.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_file, key_file


def test_a_loopback_https_backend_is_trusted_through_ssl_cert_file(
    monkeypatch, loopback_certificate
):
    # Ignoring the environment's proxies for a loopback host must not also drop SSL_CERT_FILE,
    # which a local server with its own certificate needs.
    monkeypatch.setenv("SSL_CERT_FILE", str(loopback_certificate[0]))
    with recording_server(loopback_certificate) as (port, seen):
        engine = RemoteJevEngine(f"https://127.0.0.1:{port}/v1/systemone", api_key=SECRET)
        assert engine.decide("x", QUESTIONS, "auto").answers == ANSWER["answers"]
    assert seen == [("POST /v1/systemone HTTP/1.1", f"Bearer {SECRET}")]


def test_a_loopback_https_backend_with_an_untrusted_certificate_is_refused(
    monkeypatch, loopback_certificate
):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    with recording_server(loopback_certificate) as (port, seen):
        engine = RemoteJevEngine(f"https://127.0.0.1:{port}/v1/systemone", api_key=SECRET)
        with pytest.raises(ModelUnavailable, match="unreachable"):
            engine.decide("x", QUESTIONS, "auto")
    assert seen == []
