"""Exercise the Data Space HTTP workflow without credentials or large downloads."""

import io
import zipfile
from unittest.mock import Mock

import pytest
import requests

from src.utils.download import copernicus


def response(payload=None, status=200, chunks=()):
    result = Mock(status_code=status, is_redirect=False)
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.json.return_value = payload
    result.iter_content.return_value = iter(chunks)
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(response=result)
    return result


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("COPERNICUS_USERNAME", "test-user")
    monkeypatch.setenv("COPERNICUS_PASSWORD", "test-password")
    return copernicus.CopernicusDataSpace()


@pytest.fixture
def http(monkeypatch):
    get, post = Mock(), Mock()
    monkeypatch.setattr(copernicus.requests, "get", get)
    monkeypatch.setattr(copernicus.requests, "post", post)
    monkeypatch.setattr(copernicus.time, "sleep", Mock())
    post.return_value = response({"access_token": "token", "expires_in": 600})
    return get, post


def archive():
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as output:
        output.writestr("scene.SAFE/measurement/test.tiff", b"test")
    return content.getvalue()


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("COPERNICUS_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUS_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="COPERNICUS_USERNAME"):
        copernicus.CopernicusDataSpace()


def test_download_resolves_name_and_atomically_caches_zip(client, http, tmp_path):
    get, post = http
    data = archive()
    get.side_effect = [
        response({"value": [{"Id": "cdse-id", "Online": True}]}),
        response(chunks=[data[:20], b"", data[20:]]),
    ]
    client.download("scene.SAFE", str(tmp_path))
    assert (tmp_path / "scene.zip").read_bytes() == data
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scene.zip"]
    assert get.call_args_list[0].kwargs["params"]["$filter"] == "Name eq 'scene.SAFE'"
    assert get.call_args_list[1].args[0] == f"{copernicus.DOWNLOAD_URL}(cdse-id)/$value"
    assert get.call_args_list[1].kwargs["headers"] == {"Authorization": "Bearer token"}
    assert post.call_args.kwargs["data"] == {
        "client_id": "cdse-public", "grant_type": "password",
        "username": "test-user", "password": "test-password",
    }


@pytest.mark.parametrize("products, message", [
    ([], "not found"),
    ([{"Id": "id", "Online": False}], "offline"),
    ([{"Id": "one"}, {"Id": "two"}], "Multiple"),
])
def test_unavailable_products(client, http, products, message):
    get, post = http
    get.return_value = response({"value": products})
    with pytest.raises(RuntimeError, match=message):
        client.find_product("scene.SAFE")
    post.assert_not_called()


def test_odata_name_escaping(client, http):
    get, _ = http
    get.return_value = response({"value": [{"Id": "id"}]})
    client.find_product("a'b.SAFE")
    assert get.call_args.kwargs["params"]["$filter"] == "Name eq 'a''b.SAFE'"


def test_redirect_preserves_token_and_refreshes_on_401(client, http):
    get, post = http
    redirected = response(status=302)
    redirected.is_redirect = True
    redirected.headers = {"Location": "https://zipper.dataspace.copernicus.eu/product"}
    denied = response(status=401)
    success = response()
    post.side_effect = [
        response({"access_token": "old", "expires_in": 600}),
        response({"access_token": "new", "expires_in": 600}),
    ]
    get.side_effect = [redirected, denied, success]
    assert client._download_response("id") is success
    assert get.call_args_list[1].kwargs["headers"]["Authorization"] == "Bearer old"
    assert get.call_args_list[2].kwargs["headers"]["Authorization"] == "Bearer new"
    assert get.call_args_list[2].args == (redirected.headers["Location"],)
    redirected.close.assert_called_once()
    denied.close.assert_called_once()


@pytest.mark.parametrize("location", [
    "https://attacker.example/product", "http://download.dataspace.copernicus.eu/product",
    "https://download.dataspace.copernicus.eu.attacker.example/product",
])
def test_redirect_rejects_untrusted_destinations(client, http, location):
    get, _ = http
    redirect = response(status=302)
    redirect.is_redirect = True
    redirect.headers = {"Location": location}
    get.return_value = redirect
    with pytest.raises(RuntimeError, match="redirect"):
        client._download_response("id")
    assert get.call_count == 1


def test_expired_token_is_renewed(client, http):
    _, post = http
    assert client._access_token() == "token"
    assert client._access_token() == "token"
    assert post.call_count == 1
    client._expires_at = 0
    client._access_token()
    assert post.call_count == 2


def test_invalid_credentials_are_explained(client, http):
    _, post = http
    post.return_value = response(status=401)
    with pytest.raises(RuntimeError, match="authentication failed"):
        client._access_token()


def test_interrupted_download_is_retried_without_publishing_partial(client, http, tmp_path):
    get, _ = http
    interrupted = response()

    def broken_stream(**kwargs):
        yield b"partial"
        assert not (tmp_path / "scene.zip").exists()
        raise requests.ConnectionError("connection lost")

    interrupted.iter_content.side_effect = broken_stream
    get.side_effect = [
        response({"value": [{"Id": "id"}]}), interrupted, response(chunks=[archive()]),
    ]
    client.download("scene.SAFE", str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scene.zip"]
    assert zipfile.is_zipfile(tmp_path / "scene.zip")


def test_failed_download_cleans_up_and_stops_after_four_attempts(client, http, tmp_path):
    get, _ = http
    get.side_effect = [response({"value": [{"Id": "id"}]})] + [
        requests.Timeout("timed out") for _ in range(4)
    ]
    with pytest.raises(requests.Timeout):
        client.download("scene.SAFE", str(tmp_path))
    assert list(tmp_path.iterdir()) == []
    assert get.call_count == 5


def test_non_zip_response_never_overwrites_cache(client, http, tmp_path):
    get, _ = http
    (tmp_path / "scene.zip").write_bytes(b"existing cache")
    get.side_effect = [response({"value": [{"Id": "id"}]}), response(chunks=[b"error page"])]
    with pytest.raises(RuntimeError, match="not a ZIP"):
        client.download("scene.SAFE", str(tmp_path))
    assert (tmp_path / "scene.zip").read_bytes() == b"existing cache"
    assert len(list(tmp_path.iterdir())) == 1


def test_permanent_http_error_is_not_retried(client, http, tmp_path):
    get, _ = http
    get.side_effect = [response({"value": [{"Id": "id"}]}), response(status=403)]
    with pytest.raises(requests.HTTPError):
        client.download("scene.SAFE", str(tmp_path))
    assert get.call_count == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status", [429, 503])
def test_transient_http_error_is_retried(client, http, tmp_path, status):
    get, _ = http
    get.side_effect = [
        response({"value": [{"Id": "id"}]}),
        response(status=status),
        response(chunks=[archive()]),
    ]
    client.download("scene.SAFE", str(tmp_path))
    assert zipfile.is_zipfile(tmp_path / "scene.zip")
    assert get.call_count == 3


def test_repeated_unauthorized_response_stops_after_one_renewal(client, http):
    get, post = http
    get.side_effect = [response(status=401), response(status=401)]
    with pytest.raises(requests.HTTPError):
        client._download_response("id")
    assert post.call_count == 2
    assert get.call_count == 2
