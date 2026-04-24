from moto import mock_aws
from zodb_s3blobs import migrate
from zodb_s3blobs.s3client import S3Client
from zodb_s3blobs.storage import _oid_hex
from zodb_s3blobs.storage import _tid_hex
from ZODB.utils import p64

import boto3
import os
import pytest
import tempfile


@pytest.fixture
def s3_env():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        yield


@pytest.fixture
def s3_client(s3_env):
    return S3Client(bucket_name="test-bucket", region_name="us-east-1")


class FakeStorage:
    """Minimal stand-in for a RelStorage for the migration tests.

    The real migration reads ``blob_chunk`` directly via the adapter and
    streams Postgres LOBs into a temp file. These tests replace both the
    enumeration (``iter_blob_pairs``), the size lookup (``blob_size_in_db``),
    and the materialisation (``_materialize_blob``) with in-memory versions
    backed by this storage's ``blobs`` dict.
    """

    def __init__(self, blobs):
        self.blobs = dict(blobs)  # {(oid_bytes, tid_bytes): bytes}
        self.materialize_calls = 0

    def close(self):
        pass


def _install_patches(monkeypatch, storage):
    monkeypatch.setattr(
        migrate, "iter_blob_pairs", lambda _s: iter(storage.blobs.keys())
    )
    monkeypatch.setattr(
        migrate,
        "blob_size_in_db",
        lambda _s, oid, tid: len(storage.blobs[(oid, tid)]),
    )

    def fake_materialize(_s, oid, tid, tmp_dir):
        storage.materialize_calls += 1
        data = storage.blobs[(oid, tid)]
        fd, path = tempfile.mkstemp(suffix=".blob.tmp", dir=tmp_dir)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path, len(data)

    monkeypatch.setattr(migrate, "_materialize_blob", fake_materialize)


def _expected_key(oid, tid):
    return f"blobs/{_oid_hex(oid)}/{_tid_hex(tid)}.blob"


def test_happy_path_uploads_each_blob(s3_client, tmp_path, monkeypatch):
    blobs = {
        (p64(1), p64(0x100)): b"hello",
        (p64(2), p64(0x200)): b"world!!",
        (p64(0xDEAD), p64(0xBEEF)): b"x" * 1024,
    }
    storage = FakeStorage(blobs)
    _install_patches(monkeypatch, storage)

    counts = migrate.run_migration(
        storage, s3_client, workers=2, progress_interval=1, tmp_dir=str(tmp_path)
    )

    assert counts["scanned"] == 3
    assert counts["uploaded"] == 3
    assert counts["skipped"] == 0
    assert counts["failed"] == 0
    assert counts["bytes_uploaded"] == sum(len(v) for v in blobs.values())

    keys = sorted(s3_client.list_objects("blobs/"))
    expected = sorted(_expected_key(o, t) for (o, t) in blobs)
    assert keys == expected


def test_second_run_is_idempotent(s3_client, tmp_path, monkeypatch):
    blobs = {(p64(1), p64(0x100)): b"payload"}
    storage = FakeStorage(blobs)
    _install_patches(monkeypatch, storage)

    migrate.run_migration(storage, s3_client, workers=1, tmp_dir=str(tmp_path))
    second = migrate.run_migration(
        storage, s3_client, workers=1, tmp_dir=str(tmp_path)
    )

    assert second["scanned"] == 1
    assert second["uploaded"] == 0
    assert second["skipped"] == 1


def test_overwrite_reuploads(s3_client, tmp_path, monkeypatch):
    blobs = {(p64(1), p64(0x100)): b"payload"}
    storage = FakeStorage(blobs)
    _install_patches(monkeypatch, storage)

    migrate.run_migration(storage, s3_client, workers=1, tmp_dir=str(tmp_path))
    second = migrate.run_migration(
        storage, s3_client, workers=1, overwrite=True, tmp_dir=str(tmp_path)
    )

    assert second["uploaded"] == 1
    assert second["skipped"] == 0


def test_dry_run_does_not_upload(s3_client, tmp_path, monkeypatch):
    blobs = {(p64(1), p64(0x100)): b"payload"}
    storage = FakeStorage(blobs)
    _install_patches(monkeypatch, storage)

    counts = migrate.run_migration(
        storage, s3_client, workers=1, dry_run=True, tmp_dir=str(tmp_path)
    )

    assert counts["would_upload"] == 1
    assert counts["uploaded"] == 0
    assert list(s3_client.list_objects("blobs/")) == []
    assert storage.materialize_calls == 0


def test_verify_size_detects_mismatch_and_reuploads(
    s3_client, tmp_path, monkeypatch
):
    oid, tid = p64(1), p64(0x100)
    blobs = {(oid, tid): b"full-payload-20bytes"}
    storage = FakeStorage(blobs)
    _install_patches(monkeypatch, storage)

    # Pre-seed S3 with a wrong-sized object at the target key.
    s3_client._client.put_object(
        Bucket=s3_client.bucket_name,
        Key=_expected_key(oid, tid),
        Body=b"short",
    )

    counts = migrate.run_migration(
        storage, s3_client, workers=1, verify_size=True, tmp_dir=str(tmp_path)
    )

    assert counts["size_mismatch_reuploaded"] == 1
    assert counts["skipped"] == 0
    head = s3_client.head_object(_expected_key(oid, tid))
    assert head is not None
    assert head["ContentLength"] == len(blobs[(oid, tid)])


def test_stale_blob_skipped(s3_client, tmp_path, monkeypatch):
    """A blob that vanished between enumeration and materialisation is
    counted as stale, not failed.
    """
    oid, tid = p64(1), p64(0x100)
    storage = FakeStorage({(oid, tid): b"whatever"})
    _install_patches(monkeypatch, storage)

    def vanishing_materialize(*_a, **_kw):
        raise LookupError("blob_chunk rows vanished")

    monkeypatch.setattr(migrate, "_materialize_blob", vanishing_materialize)

    counts = migrate.run_migration(
        storage, s3_client, workers=1, tmp_dir=str(tmp_path)
    )

    assert counts["stale"] == 1
    assert counts["uploaded"] == 0
    assert counts["failed"] == 0


def test_temp_file_removed_after_upload(s3_client, tmp_path, monkeypatch):
    oid, tid = p64(1), p64(0x100)
    storage = FakeStorage({(oid, tid): b"payload"})
    _install_patches(monkeypatch, storage)

    migrate.run_migration(
        storage, s3_client, workers=1, tmp_dir=str(tmp_path)
    )

    # No leftover .blob.tmp files in the tmp dir.
    remaining = [
        f for f in os.listdir(str(tmp_path)) if f.endswith(".blob.tmp")
    ]
    assert remaining == []


def test_key_format_regression(tmp_path):
    """Pins the S3 key format so _oid_hex / _tid_hex can't drift silently."""
    assert migrate.s3_key_for(p64(1), p64(0x40875bfa5cccf11)) == (
        "blobs/1/40875bfa5cccf11.blob"
    )
    assert migrate.s3_key_for(p64(0), p64(0)) == "blobs/0/0.blob"
    assert migrate.s3_key_for(p64(0x94E4), p64(0xDEADBEEF)) == (
        "blobs/94e4/deadbeef.blob"
    )


def test_failed_upload_raises(s3_client, tmp_path, monkeypatch):
    blobs = {(p64(1), p64(0x100)): b"payload"}
    storage = FakeStorage(blobs)
    _install_patches(monkeypatch, storage)

    def boom(*_a, **_kw):
        raise RuntimeError("simulated upload failure")

    monkeypatch.setattr(s3_client, "upload_file", boom)

    with pytest.raises(RuntimeError, match="migration had 1 failures"):
        migrate.run_migration(
            storage, s3_client, workers=1, tmp_dir=str(tmp_path)
        )
