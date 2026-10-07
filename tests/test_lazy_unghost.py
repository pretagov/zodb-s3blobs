"""Unghosting a Blob doesn't download it from S3."""

from moto import mock_aws
from persistent.mapping import PersistentMapping
from ZODB.blob import Blob
from ZODB.interfaces import IBlob
from ZODB.tests.MVCCMappingStorage import MVCCMappingStorage
from zodb_s3blobs.cache import S3BlobCache
from zodb_s3blobs.s3client import S3Client
from zodb_s3blobs.storage import S3BlobStorage

import boto3
import os
import pytest
import transaction
import ZODB


@pytest.fixture
def s3_client():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        yield S3Client(bucket_name="test-bucket", region_name="us-east-1")


@pytest.fixture
def reader(s3_client, tmp_path):
    """A connection, with a cold cache, to a database holding root["f"]["blob"]."""
    shared = MVCCMappingStorage()

    def make_db(name):
        return ZODB.DB(
            S3BlobStorage(
                shared.new_instance(),
                s3_client,
                S3BlobCache(str(tmp_path / f"{name}-cache"), max_size=10**7),
                temp_dir=str(tmp_path / f"{name}-staging"),
            )
        )

    writer_db = make_db("writer")
    tm = transaction.TransactionManager()
    conn = writer_db.open(tm)
    conn.root()["f"] = PersistentMapping(blob=Blob(b"blob data"))
    tm.commit()
    conn.close()
    writer_db.close()

    db = make_db("reader")
    tm = transaction.TransactionManager()
    conn = db.open(tm)
    downloads = []
    download_file = s3_client.download_file
    s3_client.download_file = lambda *args: (
        downloads.append(args),
        download_file(*args),
    )
    yield conn, downloads
    tm.abort()
    conn.close()
    db.close()


def test_unghost_does_not_download(reader):
    conn, downloads = reader
    blob = conn.root()["f"]["blob"]

    assert IBlob.providedBy(blob)  # unghosts it
    assert blob._p_serial == conn._storage.load(blob._p_oid)[1]
    assert not os.path.exists(blob._p_blob_committed)
    assert downloads == []


def test_open_downloads(reader):
    conn, downloads = reader
    blob = conn.root()["f"]["blob"]

    with blob.open("r") as f:
        assert f.read() == b"blob data"
    assert len(downloads) == 1


def test_committed_downloads(reader):
    conn, downloads = reader
    blob = conn.root()["f"]["blob"]

    path = blob.committed()
    with open(path, "rb") as f:
        assert f.read() == b"blob data"
    assert path == blob._p_blob_committed
    assert len(downloads) == 1


def test_flag_cleared_after_unghost(reader):
    conn, _ = reader
    blob = conn.root()["f"]["blob"]
    blob._p_activate()

    assert conn._storage._unghosting is False


def test_patch_is_applied_once():
    from zodb_s3blobs.storage import _patch_connection_setstate

    import ZODB.Connection

    patched = ZODB.Connection.Connection.setstate
    _patch_connection_setstate()
    assert ZODB.Connection.Connection.setstate is patched
