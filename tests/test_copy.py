"""zodbconvert (Copy) integration: source FileStorage -> dest S3BlobStorage."""

from moto import mock_aws
from relstorage.adapters.sqlite.adapter import Sqlite3Adapter
from relstorage.options import Options
from relstorage.storage import RelStorage
from ZODB.blob import Blob, BlobStorage
from ZODB.FileStorage import FileStorage
from zodb_s3blobs.cache import S3BlobCache
from zodb_s3blobs.s3client import S3Client
from zodb_s3blobs.storage import S3BlobStorage

import boto3
import pytest
import transaction
import ZODB


@pytest.fixture
def s3_env():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        yield


@pytest.fixture
def s3_client(s3_env):
    return S3Client(bucket_name="test-bucket", region_name="us-east-1")


@pytest.fixture
def blob_cache(tmp_path):
    return S3BlobCache(str(tmp_path / "cache"), max_size=10 * 1024 * 1024)


def _populate_source(tmp_path, blobs):
    """Build a FileStorage+BlobStorage source with the given blobs committed.

    Returns the (still open, read-only) source storage and the list of TIDs
    written, in commit order.
    """
    fs_path = str(tmp_path / "source.fs")
    blob_dir = str(tmp_path / "source-blobs")

    fs = FileStorage(fs_path, create=True)
    storage = BlobStorage(blob_dir, fs)
    db = ZODB.DB(storage)
    conn = db.open()
    root = conn.root()
    tids = []
    for name, content in blobs.items():
        b = Blob()
        with b.open("w") as f:
            f.write(content)
        root[name] = b
        transaction.commit()
        tids.append(storage.lastTransaction())
    conn.close()
    db.close()

    fs = FileStorage(fs_path, read_only=True)
    return BlobStorage(blob_dir, fs), tids


def _make_dest(s3_client, blob_cache, tmp_path):
    """Build a RelStorage(SQLite)-backed S3BlobStorage destination.

    Mirrors production where the inner storage is RelStorage. SQLite avoids
    needing an external database in tests.
    """
    db_path = str(tmp_path / "dest.sqlite")
    blob_dir = str(tmp_path / "dest-blobs")
    options = Options(
        keep_history=True,
        blob_dir=blob_dir,
        shared_blob_dir=False,
    )
    adapter = Sqlite3Adapter(data_dir=db_path, pragmas={}, options=options)
    inner = RelStorage(adapter, name="dest", options=options)
    return S3BlobStorage(
        inner, s3_client, blob_cache, temp_dir=str(tmp_path / "dest-staging")
    )


class TestCopyTransactionsFrom:
    def test_roundtrip(self, s3_client, blob_cache, tmp_path):
        source, _ = _populate_source(
            tmp_path,
            {"a": b"alpha", "b": b"bravo", "c": b"charlie"},
        )
        dest = _make_dest(s3_client, blob_cache, tmp_path)

        dest.copyTransactionsFrom(source)

        keys = list(s3_client.list_objects("blobs/"))
        assert len(keys) == 3

        db = ZODB.DB(dest)
        conn = db.open()
        root = conn.root()
        for name, expected in [("a", b"alpha"), ("b", b"bravo"), ("c", b"charlie")]:
            with root[name].open("r") as f:
                assert f.read() == expected
        conn.close()
        db.close()
        source.close()

    def test_tid_preservation(self, s3_client, blob_cache, tmp_path):
        """S3 keys must carry the source's TIDs so loadBlob finds them."""
        source, tids = _populate_source(tmp_path, {"only": b"data"})
        dest = _make_dest(s3_client, blob_cache, tmp_path)

        dest.copyTransactionsFrom(source)

        keys = list(s3_client.list_objects("blobs/"))
        assert len(keys) == 1
        # Key format: blobs/{oid_hex}/{tid_hex}.blob
        # The blob was committed in the last transaction in `tids`.
        # Key format uses the same _tid_hex helper (strips leading zeros).
        last_tid_hex = tids[-1].hex().lstrip("0") or "0"
        assert keys[0].endswith(f"/{last_tid_hex}.blob"), (
            f"expected key to end with /{last_tid_hex}.blob, got {keys[0]}"
        )

        source.close()
        dest.close()
