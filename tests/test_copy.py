"""copyTransactionsFrom (zodbconvert): FileStorage source -> S3BlobStorage
wrapping RelStorage (SQLite)."""

from moto import mock_aws
from ZODB.blob import Blob
from ZODB.blob import BlobStorage
from ZODB.FileStorage import FileStorage
from ZODB.MappingStorage import MappingStorage
from zodb_s3blobs.cache import S3BlobCache
from zodb_s3blobs.s3client import S3Client
from zodb_s3blobs.storage import _oid_hex
from zodb_s3blobs.storage import _tid_hex
from zodb_s3blobs.storage import S3BlobStorage

import boto3
import pytest
import transaction
import ZODB


pytest.importorskip("relstorage")

from relstorage.adapters.sqlite.adapter import Sqlite3Adapter
from relstorage.options import Options
from relstorage.storage import RelStorage


@pytest.fixture
def s3_client():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        yield S3Client(bucket_name="test-bucket", region_name="us-east-1")


@pytest.fixture
def blob_cache(tmp_path):
    return S3BlobCache(str(tmp_path / "cache"), max_size=10 * 1024 * 1024)


@pytest.fixture
def source(tmp_path):
    """A FileStorage source with three blobs, one per transaction."""
    fs_path = str(tmp_path / "source.fs")
    blob_dir = str(tmp_path / "source-blobs")
    storage = BlobStorage(blob_dir, FileStorage(fs_path, create=True))
    db = ZODB.DB(storage)
    conn = db.open()
    oids = {}
    for name, content in [("a", b"alpha"), ("b", b"bravo"), ("c", b"charlie")]:
        blob = Blob()
        with blob.open("w") as f:
            f.write(content)
        conn.root()[name] = blob
        transaction.commit()
        oids[name] = (blob._p_oid, storage.lastTransaction())
    conn.close()
    db.close()

    storage = BlobStorage(blob_dir, FileStorage(fs_path, read_only=True))
    yield storage, oids
    storage.close()


@pytest.fixture(params=[True, False], ids=["history-preserving", "history-free"])
def dest(request, s3_client, blob_cache, tmp_path):
    options = Options(keep_history=request.param)
    adapter = Sqlite3Adapter(
        data_dir=str(tmp_path / "dest.sqlite"), pragmas={}, options=options
    )
    storage = S3BlobStorage(
        RelStorage(adapter, name="dest", options=options),
        s3_client,
        blob_cache,
        temp_dir=str(tmp_path / "dest-staging"),
    )
    yield storage
    storage.close()


def test_copy_uploads_blobs_with_source_tids(source, dest, s3_client):
    source_storage, oids = source

    dest.copyTransactionsFrom(source_storage)

    assert set(s3_client.list_objects("blobs/")) == {
        f"blobs/{_oid_hex(oid)}/{_tid_hex(tid)}.blob" for oid, tid in oids.values()
    }


def test_copy_result_is_readable(source, dest):
    source_storage, _ = source

    dest.copyTransactionsFrom(source_storage)

    db = ZODB.DB(dest)
    conn = db.open()
    for name, expected in [("a", b"alpha"), ("b", b"bravo"), ("c", b"charlie")]:
        with conn.root()[name].open("r") as f:
            assert f.read() == expected
    conn.close()


def test_copy_does_not_store_blobs_in_base(source, dest):
    source_storage, _ = source

    dest.copyTransactionsFrom(source_storage)

    conn, cursor = dest._adapter.connmanager.open_for_load()
    try:
        cursor.execute("SELECT COUNT(*) FROM blob_chunk")
        assert cursor.fetchone()[0] == 0
    finally:
        dest._adapter.connmanager.close(conn, cursor)


def test_copy_requires_relstorage_base(source, s3_client, blob_cache, tmp_path):
    source_storage, _ = source
    storage = S3BlobStorage(
        MappingStorage(), s3_client, blob_cache, temp_dir=str(tmp_path / "staging")
    )

    with pytest.raises(NotImplementedError):
        storage.copyTransactionsFrom(source_storage)
