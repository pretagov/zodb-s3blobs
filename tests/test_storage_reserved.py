"""Reserved blobs: committed without data, uploaded to their final key after."""

from moto import mock_aws
from persistent.mapping import PersistentMapping
from ZODB.MappingStorage import MappingStorage
from zodb_s3blobs.cache import S3BlobCache
from zodb_s3blobs.s3client import S3Client
from zodb_s3blobs.storage import blob_key
from zodb_s3blobs.storage import committed_blob_key
from zodb_s3blobs.storage import reserve_blob
from zodb_s3blobs.storage import S3BlobStorage

import boto3
import logging
import pytest
import time
import transaction
import ZODB
import ZODB.blob
import ZODB.POSException


PART = 5 * 1024 * 1024  # S3's minimum size for all but the last part
DATA = b"a" * PART + b"tail"


@pytest.fixture
def s3_client():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        yield S3Client(bucket_name="test-bucket", region_name="us-east-1")


def _relstorage(keep_history, tmp_path):
    pytest.importorskip("relstorage")
    from relstorage.adapters.sqlite.adapter import Sqlite3Adapter
    from relstorage.options import Options
    from relstorage.storage import RelStorage

    options = Options(keep_history=keep_history, pack_gc=True)
    adapter = Sqlite3Adapter(data_dir=str(tmp_path / "db"), pragmas={}, options=options)
    return RelStorage(adapter, options=options)


@pytest.fixture(params=["mapping", "relstorage-hf", "relstorage-hp"])
def db(request, s3_client, tmp_path):
    base = (
        MappingStorage()
        if request.param == "mapping"
        else _relstorage(request.param == "relstorage-hp", tmp_path)
    )
    storage = S3BlobStorage(
        base,
        s3_client,
        S3BlobCache(str(tmp_path / "cache"), max_size=50 * 1024 * 1024),
        temp_dir=str(tmp_path / "staging"),
    )
    database = ZODB.DB(storage)
    yield database
    database.close()


def _open(db):
    tm = transaction.TransactionManager()
    return db.open(tm), tm


def _reserve(db):
    """Commit a reserved blob in an upload record; return its oid and key."""
    conn, tm = _open(db)
    blob = reserve_blob()
    conn.root()["upload"] = PersistentMapping(blob=blob)
    tm.commit()
    key = committed_blob_key(blob)
    oid = blob._p_oid
    conn.close()
    return oid, key


def _start_upload(s3_client, key, data=DATA):
    upload_id = s3_client.create_multipart_upload(key)
    parts = [
        {
            "PartNumber": n,
            "ETag": s3_client.upload_part(key, upload_id, n, data[i : i + PART]),
        }
        for n, i in enumerate(range(0, len(data), PART), start=1)
    ]
    return upload_id, parts


def _upload(s3_client, key, data=DATA):
    upload_id, parts = _start_upload(s3_client, key, data)
    s3_client.complete_multipart_upload(key, upload_id, parts)


def _link(db):
    """Move the reserved blob from the upload record to a file, as the end of
    an upload does."""
    conn, tm = _open(db)
    root = conn.root()
    root["file"] = PersistentMapping(blob=root["upload"]["blob"])
    del root["upload"]
    tm.commit()
    conn.close()


def _read(db, name):
    conn, tm = _open(db)
    try:
        with conn.root()[name]["blob"].open("r") as f:
            return f.read()
    finally:
        tm.abort()
        conn.close()


def _pack(db):
    # Pack right up to now: the shortest possible pack time.
    time.sleep(0.01)
    db.pack(time.time())


def _blob_keys(s3_client):
    return sorted(s3_client.list_objects("blobs/"))


def _uploads(s3_client):
    return list(s3_client.list_multipart_uploads("blobs/"))


class TestReserve:
    def test_commit_uploads_nothing(self, db, s3_client):
        oid, key = _reserve(db)
        assert _blob_keys(s3_client) == []
        assert key == blob_key(oid, db.lastTransaction())

    def test_blob_has_no_data_until_uploaded(self, db, s3_client):
        _reserve(db)
        with pytest.raises(ZODB.POSException.POSKeyError):
            _read(db, "upload")

    def test_data_uploaded_to_key_is_the_blob(self, db, s3_client):
        _oid, key = _reserve(db)
        _upload(s3_client, key)
        assert _read(db, "upload") == DATA

    def test_linking_stores_and_copies_nothing(self, db, s3_client, monkeypatch):
        _oid, key = _reserve(db)
        _upload(s3_client, key)
        calls = []
        monkeypatch.setattr(s3_client, "upload_file", lambda *a: calls.append(a))
        monkeypatch.setattr(s3_client, "copy_object", lambda *a: calls.append(a))
        _link(db)
        assert calls == []
        assert _blob_keys(s3_client) == [key]
        assert _read(db, "file") == DATA

    def test_committed_blob_key_keeps_blob_a_ghost(self, db):
        conn, tm = _open(db)
        blob = reserve_blob()
        conn.root()["upload"] = PersistentMapping(blob=blob)
        tm.commit()
        committed_blob_key(blob)
        assert blob._p_changed is None  # still a ghost: nothing was loaded
        conn.close()

    def test_committed_blob_key_needs_a_commit(self):
        with pytest.raises(ValueError, match="not been committed"):
            committed_blob_key(reserve_blob())

    @pytest.mark.parametrize(
        "content",
        [b"S3BLOB-RESERVED\n\n", b"S3BLOB-RESERVED", b"x S3BLOB-RESERVED\n"],
        ids=["trailing", "truncated", "leading"],
    )
    def test_only_the_exact_marker_is_a_reservation(self, db, s3_client, content):
        conn, tm = _open(db)
        blob = ZODB.blob.Blob(content)
        conn.root()["file"] = PersistentMapping(blob=blob)
        tm.commit()
        conn.close()
        assert len(_blob_keys(s3_client)) == 1
        assert _read(db, "file") == content


class TestPack:
    def test_keeps_reachable_reservation_with_upload_in_progress(self, db, s3_client):
        _oid, key = _reserve(db)
        upload_id, parts = _start_upload(s3_client, key)
        _pack(db)
        assert _uploads(s3_client) == [(key, upload_id)]
        s3_client.complete_multipart_upload(key, upload_id, parts)
        _link(db)
        assert _read(db, "file") == DATA

    def test_keeps_linked_blob(self, db, s3_client):
        _oid, key = _reserve(db)
        _upload(s3_client, key)
        _link(db)
        _pack(db)
        assert _blob_keys(s3_client) == [key]
        assert _read(db, "file") == DATA

    def test_removes_abandoned_reservation_and_its_data(self, db, s3_client):
        _oid, key = _reserve(db)
        _upload(s3_client, key)
        _abandon(db)
        _pack(db)
        assert _blob_keys(s3_client) == []

    def test_aborts_upload_to_abandoned_reservation(self, db, s3_client):
        _oid, key = _reserve(db)
        _start_upload(s3_client, key)
        _abandon(db)
        _pack(db)
        assert _uploads(s3_client) == []

    def test_leaves_uploads_to_keys_newer_than_last_commit(self, db, s3_client):
        key = "blobs/1/7fffffffffffffff.blob"
        upload_id = s3_client.create_multipart_upload(key)
        _pack(db)
        assert _uploads(s3_client) == [(key, upload_id)]

    def test_leaves_uploads_outside_blob_keys(self, db, s3_client):
        upload_id = s3_client.create_multipart_upload("blobs/not-a-blob")
        _pack(db)
        assert _uploads(s3_client) == [("blobs/not-a-blob", upload_id)]

    def test_listing_failure_does_not_fail_pack(
        self, db, s3_client, monkeypatch, caplog
    ):
        def fail(prefix):
            raise RuntimeError("not supported")

        monkeypatch.setattr(s3_client, "list_multipart_uploads", fail)
        with caplog.at_level(logging.WARNING, logger="zodb_s3blobs.storage"):
            _pack(db)
        assert "failed to list multipart uploads" in caplog.text

    def test_abort_failure_does_not_fail_pack(self, db, s3_client, monkeypatch, caplog):
        _oid, key = _reserve(db)
        _start_upload(s3_client, key)
        _abandon(db)

        def fail(key, upload_id):
            raise RuntimeError("boom")

        monkeypatch.setattr(s3_client, "abort_multipart_upload", fail)
        with caplog.at_level(logging.WARNING, logger="zodb_s3blobs.storage"):
            _pack(db)
        assert "failed to abort multipart upload" in caplog.text


def _abandon(db):
    """Drop the upload record without linking the blob, as when an unfinished
    upload expires."""
    conn, tm = _open(db)
    del conn.root()["upload"]
    tm.commit()
    conn.close()


class TestListMultipartUploads:
    def test_keys_are_relative_to_the_client_prefix(self, s3_client):
        prefixed = S3Client(
            bucket_name="test-bucket", region_name="us-east-1", prefix="site"
        )
        upload_id = prefixed.create_multipart_upload("blobs/1/2.blob")
        assert list(prefixed.list_multipart_uploads("blobs/")) == [
            ("blobs/1/2.blob", upload_id)
        ]
        assert list(s3_client.list_multipart_uploads("blobs/")) == []
