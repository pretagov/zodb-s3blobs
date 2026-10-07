"""Presigned download URLs for committed blobs."""

from moto import mock_aws
from urllib.parse import parse_qs
from urllib.parse import urlparse
from ZODB.MappingStorage import MappingStorage
from ZODB.utils import p64
from ZODB.utils import z64
from zodb_s3blobs.cache import S3BlobCache
from zodb_s3blobs.s3client import S3Client
from zodb_s3blobs.s3client import S3OperationError
from zodb_s3blobs.storage import S3BlobStorage

import base64
import boto3
import pytest
import transaction


@pytest.fixture
def s3_env():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        yield


def _storage(s3_client, tmp_path):
    return S3BlobStorage(
        MappingStorage(),
        s3_client,
        S3BlobCache(str(tmp_path / "cache"), max_size=10**7),
        temp_dir=str(tmp_path / "staging"),
    )


@pytest.fixture
def storage(s3_env, tmp_path):
    return _storage(
        S3Client(bucket_name="test-bucket", region_name="us-east-1"), tmp_path
    )


def _commit_blob(storage, tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"data")
    txn = transaction.get()
    storage.tpc_begin(txn)
    storage.storeBlob(p64(1), z64, b"pickle", str(path), "", txn)
    storage.tpc_vote(txn)
    return storage.tpc_finish(txn)


def test_url_for_committed_blob(storage, tmp_path):
    tid = _commit_blob(storage, tmp_path)

    url = urlparse(storage.presigned_url(p64(1), tid))
    query = parse_qs(url.query)

    assert url.path.endswith("/" + storage._s3_key(p64(1), tid))
    assert "Signature" in query or "X-Amz-Signature" in query
    assert "response-content-type" not in query
    assert "response-content-disposition" not in query


def test_url_sets_response_headers(storage, tmp_path):
    tid = _commit_blob(storage, tmp_path)

    url = storage.presigned_url(
        p64(1),
        tid,
        content_type="application/pdf",
        filename='Résumé "final"\r\n.pdf',
        disposition="attachment",
    )
    query = parse_qs(urlparse(url).query)

    assert query["response-content-type"] == ["application/pdf"]
    assert query["response-content-disposition"] == [
        "attachment; filename*=UTF-8''R%C3%A9sum%C3%A9%20%22final%22%0D%0A.pdf"
    ]


@pytest.mark.parametrize("oid, serial", [(None, p64(1)), (p64(1), None), (p64(1), z64)])
def test_none_without_committed_revision(storage, oid, serial):
    assert storage.presigned_url(oid, serial) is None


def test_none_for_blob_of_current_transaction(storage, tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"data")
    txn = transaction.get()
    storage.tpc_begin(txn)
    storage.storeBlob(p64(1), z64, b"pickle", str(path), "", txn)

    assert storage.presigned_url(p64(1), p64(2)) is None
    storage.tpc_abort(txn)


def test_none_with_sse_c(s3_env, tmp_path):
    key = base64.b64encode(b"k" * 32).decode()
    s3_client = S3Client(
        bucket_name="test-bucket", region_name="us-east-1", sse_customer_key=key
    )

    assert _storage(s3_client, tmp_path).presigned_url(p64(1), p64(2)) is None


def test_none_and_logged_if_signing_fails(storage, caplog):
    def fail(*args, **kwargs):
        raise S3OperationError("boom")

    storage._s3_client.generate_get_presigned_url = fail

    assert storage.presigned_url(p64(1), p64(2)) is None
    assert "Failed to presign S3 URL for oid=1 tid=2" in caplog.text
