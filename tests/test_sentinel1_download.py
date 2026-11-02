"""Check the download/preprocessing boundary without requiring GDAL."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock, patch
import zipfile

import pytest


@pytest.fixture
def sentinel1():
    # Raster processing is outside these tests. Load the real retrieval code with
    # only its GDAL-dependent imports stubbed, without affecting other tests.
    preprocess = ModuleType("src.data.preprocess")
    preprocess.preprocess = Mock()
    warp = ModuleType("src.data.warp")
    warp.get_image_info = Mock()
    warp.warp = Mock()
    path = Path(__file__).parents[1] / "src/utils/download/sentinel1.py"
    spec = importlib.util.spec_from_file_location("sentinel1_download_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"src.data.preprocess": preprocess, "src.data.warp": warp}):
        spec.loader.exec_module(module)
    return module


def test_cached_archive_needs_no_credentials_and_preserves_uuid(sentinel1, monkeypatch, tmp_path):
    cache = tmp_path / "image_cache"
    cache.mkdir()
    (cache / "scene.zip").touch()
    api = Mock()
    paths = Mock(return_value=["retrieved"])
    monkeypatch.setattr(sentinel1, "CopernicusDataSpace", api)
    monkeypatch.setattr(sentinel1, "retrieve_paths", paths)
    images = [{"uuid": "original-database-uuid", "name": "scene.SAFE"}]
    result = sentinel1.retrieve(images, "scratch", str(tmp_path), workers=2)
    assert result == ["retrieved"]
    api.assert_not_called()
    paths.assert_called_once_with(
        {"original-database-uuid": str(cache / "scene.zip")}, "scratch", workers=2
    )


def test_preprocessed_images_are_skipped(sentinel1, monkeypatch, tmp_path):
    (tmp_path / "preprocess" / "done").mkdir(parents=True)
    api, paths = Mock(), Mock(return_value=[])
    monkeypatch.setattr(sentinel1, "CopernicusDataSpace", api)
    monkeypatch.setattr(sentinel1, "retrieve_paths", paths)
    assert sentinel1.retrieve([{"uuid": "done", "name": "scene.SAFE"}], "scratch", str(tmp_path)) == []
    api.assert_not_called()
    paths.assert_called_once_with({}, "scratch", workers=0)


def test_missing_archives_are_downloaded_once_per_name(sentinel1, monkeypatch, tmp_path):
    api = Mock()
    paths = Mock(return_value=[])
    monkeypatch.setattr(sentinel1, "CopernicusDataSpace", Mock(return_value=api))
    monkeypatch.setattr(sentinel1, "retrieve_paths", paths)
    images = [{"uuid": uuid, "name": "scene.SAFE"} for uuid in ("one", "two")]
    sentinel1.retrieve(images, "scratch", str(tmp_path))
    api.download.assert_called_once_with("scene.SAFE", str(tmp_path / "image_cache"))
    assert set(paths.call_args.args[0]) == {"one", "two"}


def test_download_failure_stops_preprocessing(sentinel1, monkeypatch, tmp_path):
    api, paths = Mock(), Mock()
    api.download.side_effect = RuntimeError("not found")
    monkeypatch.setattr(sentinel1, "CopernicusDataSpace", Mock(return_value=api))
    monkeypatch.setattr(sentinel1, "retrieve_paths", paths)
    with pytest.raises(RuntimeError, match="not found"):
        sentinel1.retrieve([{"uuid": "uuid", "name": "scene.SAFE"}], "scratch", str(tmp_path))
    paths.assert_not_called()


def test_downloaded_safe_zip_produces_existing_channel_metadata(sentinel1, tmp_path):
    scene = "S1A_IW_GRDH_1SDV_20211105T024716_20211105T024741_040430_04CB0F_8DF4.SAFE"
    archive = tmp_path / scene.replace(".SAFE", ".zip")
    with zipfile.ZipFile(archive, "w") as output:
        for channel in ("vv", "vh"):
            output.writestr(f"{scene}/measurement/s1a-iw-grd-{channel}-test.tiff", b"test")
    images = sentinel1.retrieve_paths({"legacy-uuid": str(archive)}, str(tmp_path))
    assert len(images) == 1
    assert images[0].uuid == "legacy-uuid"
    assert images[0].name == scene
    assert images[0].time.isoformat() == "2021-11-05T02:47:16+00:00"
    assert {channel["Name"] for channel in images[0].channels} == {"vv", "vh"}
    assert all(Path(channel["Path"]).exists() for channel in images[0].channels)
