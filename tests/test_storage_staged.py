from moto import mock_aws
from ZODB.MappingStorage import MappingStorage
from ZODB.utils import p64
from zodb_s3blobs.cache import S3BlobCache
from zodb_s3blobs.s3client import S3Client
from zodb_s3blobs.storage import S3BlobStorage
from zodb_s3blobs.storage import write_staged_marker

import boto3
import os
import pytest
import shutil
import transaction


@pytest.fixture
def s3_env():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        yield


@pytest.fixture
def s3_client(s3_env):
    return S3Client(bucket_name="test-bucket", region_name="us-east-1")


@pytest.fixture
def storage(s3_client, tmp_path):
    cache = S3BlobCache(str(tmp_path / "cache"), max_size=10 * 1024 * 1024)
    return S3BlobStorage(
        MappingStorage(), s3_client, cache, temp_dir=str(tmp_path / "staging")
    )


def _seed_staging_object(s3_client, key, body):
    s3_client._client.put_object(
        Bucket=s3_client.bucket_name, Key=s3_client._full_key(key), Body=body
    )


def _make_marker(tmp_path, staging_key, size, name="marker.bin"):
    p = str(tmp_path / name)
    write_staged_marker(p, staging_key, size)
    return p


class TestStagedUpload:
    def test_copy_object_on_vote(self, storage, s3_client, tmp_path):
        staging_key = "tus-staging/abc123"
        _seed_staging_object(s3_client, staging_key, b"uploaded via multipart")

        marker_path = _make_marker(tmp_path, staging_key, 22)

        oid = p64(1)
        txn = transaction.get()
        storage.tpc_begin(txn)
        storage.storeBlob(oid, p64(0), b"pickle", marker_path, "", txn)
        storage.tpc_vote(txn)

        # Final blob should now exist in S3 as blobs/{oid}/{tid}.blob
        keys = sorted(s3_client.list_objects("blobs/"))
        assert len(keys) == 1
        assert keys[0].endswith(".blob")

        storage.tpc_finish(txn)

        # After commit, the staging key is deleted, and the final key remains
        remaining = sorted(s3_client.list_objects(""))
        assert any(k.startswith("blobs/") for k in remaining)
        assert not any(k == staging_key for k in remaining)

    def test_marker_bypasses_local_move(self, storage, s3_client, tmp_path):
        """storeBlob doesn't stage the marker file locally."""
        staging_key = "tus-staging/xyz"
        _seed_staging_object(s3_client, staging_key, b"data")

        marker_path = _make_marker(tmp_path, staging_key, 4)

        calls = []
        orig_move = shutil.move
        shutil.move = lambda *a, **kw: calls.append(a) or orig_move(*a, **kw)
        try:
            oid = p64(7)
            txn = transaction.get()
            storage.tpc_begin(txn)
            storage.storeBlob(oid, p64(0), b"pickle", marker_path, "", txn)
        finally:
            shutil.move = orig_move

        assert calls == []  # shutil.move was not invoked
        assert oid in storage._pending_blobs_s3
        assert oid not in storage._pending_blobs
        # Marker file was removed
        assert not os.path.exists(marker_path)

        storage.tpc_vote(txn)
        storage.tpc_finish(txn)

    def test_abort_deletes_staging_and_final(self, storage, s3_client, tmp_path):
        staging_key = "tus-staging/doomed"
        _seed_staging_object(s3_client, staging_key, b"will be aborted")

        marker_path = _make_marker(tmp_path, staging_key, 15)

        oid = p64(2)
        txn = transaction.get()
        storage.tpc_begin(txn)
        storage.storeBlob(oid, p64(0), b"pickle", marker_path, "", txn)
        storage.tpc_vote(txn)

        # Final key exists after vote
        assert len(list(s3_client.list_objects("blobs/"))) == 1
        # Staging key still there (will be cleaned in finish OR abort)
        assert len(list(s3_client.list_objects("tus-staging/"))) == 1

        storage.tpc_abort(txn)

        # Both gone after abort
        assert list(s3_client.list_objects("blobs/")) == []
        assert list(s3_client.list_objects("tus-staging/")) == []
        assert storage._pending_blobs_s3 == {}
        assert storage._pending_staging_keys == []

    def test_finish_deletes_staging_but_keeps_final(self, storage, s3_client, tmp_path):
        staging_key = "tus-staging/keep-final"
        _seed_staging_object(s3_client, staging_key, b"kept bytes")

        marker_path = _make_marker(tmp_path, staging_key, 10)

        oid = p64(3)
        txn = transaction.get()
        storage.tpc_begin(txn)
        storage.storeBlob(oid, p64(0), b"pickle", marker_path, "", txn)
        storage.tpc_vote(txn)
        storage.tpc_finish(txn)

        # Final blob remains; staging gone
        assert len(list(s3_client.list_objects("blobs/"))) == 1
        assert list(s3_client.list_objects("tus-staging/")) == []

    def test_staged_blob_not_added_to_cache(self, storage, s3_client, tmp_path):
        """S3-staged blobs are NOT pre-populated into the local cache."""
        staging_key = "tus-staging/nocache"
        _seed_staging_object(s3_client, staging_key, b"content")

        marker_path = _make_marker(tmp_path, staging_key, 7)

        oid = p64(4)
        txn = transaction.get()
        storage.tpc_begin(txn)
        storage.storeBlob(oid, p64(0), b"pickle", marker_path, "", txn)
        storage.tpc_vote(txn)
        tid = storage.tpc_finish(txn)

        # Cache has no entry for this oid/tid — loadBlob will fetch from S3
        assert storage._cache.get(oid, tid) is None

        # loadBlob still works (downloads on demand)
        path = storage.loadBlob(oid, tid)
        with open(path, "rb") as f:
            assert f.read() == b"content"

    def test_mixed_local_and_s3_staged_in_same_txn(self, storage, s3_client, tmp_path):
        """A transaction with both a locally-staged and S3-staged blob."""
        # Local blob
        local_path = tmp_path / "local.bin"
        local_path.write_bytes(b"local blob bytes")

        # S3-staged blob
        staging_key = "tus-staging/mixed"
        _seed_staging_object(s3_client, staging_key, b"staged in s3")
        marker_path = _make_marker(tmp_path, staging_key, 12, name="mixed_marker.bin")

        txn = transaction.get()
        storage.tpc_begin(txn)
        storage.storeBlob(p64(10), p64(0), b"pickle1", str(local_path), "", txn)
        storage.storeBlob(p64(11), p64(0), b"pickle2", marker_path, "", txn)
        storage.tpc_vote(txn)

        # Both blobs in S3 under blobs/
        assert len(list(s3_client.list_objects("blobs/"))) == 2

        storage.tpc_finish(txn)
        assert list(s3_client.list_objects("tus-staging/")) == []


class TestMarker:
    def _commit(self, storage, oid, path):
        txn = transaction.get()
        storage.tpc_begin(txn)
        storage.storeBlob(oid, p64(0), b"pickle", path, "", txn)
        storage.tpc_vote(txn)
        return storage.tpc_finish(txn)

    def _read(self, s3_client, key, tmp_path):
        path = str(tmp_path / "download")
        s3_client.download_file(key, path)
        with open(path, "rb") as f:
            return f.read()

    def test_marker_outside_staging_prefix_is_stored_as_data(
        self, storage, s3_client, tmp_path
    ):
        """A file that looks like a marker can't copy, or delete, another key."""
        victim = tmp_path / "victim.bin"
        victim.write_bytes(b"private")
        victim_tid = self._commit(storage, p64(1), str(victim))
        victim_key = storage._s3_key(p64(1), victim_tid)

        forged = tmp_path / "forged.bin"
        content = b"S3BLOB-STAGED\n" + victim_key.encode() + b"\n7\n"
        forged.write_bytes(content)
        tid = self._commit(storage, p64(2), str(forged))

        assert self._read(s3_client, victim_key, tmp_path) == b"private"
        assert self._read(s3_client, storage._s3_key(p64(2), tid), tmp_path) == content

    @pytest.mark.parametrize(
        "content",
        [
            b"S3BLOB-STAGED\ntus-staging/x\n",
            b"S3BLOB-STAGED\ntus-staging/x\nnot-a-size\n",
            b"S3BLOB-STAGED\n\xff\n1\n",
            b"S3BLOB-STAGED\ntus-staging/x\n1\n" + b"-" * 4096,
        ],
        ids=["missing-size", "bad-size", "bad-key", "too-big"],
    )
    def test_malformed_marker_is_stored_as_data(
        self, storage, s3_client, tmp_path, content
    ):
        path = tmp_path / "blob.bin"
        path.write_bytes(content)
        tid = self._commit(storage, p64(1), str(path))

        assert self._read(s3_client, storage._s3_key(p64(1), tid), tmp_path) == content

    def test_write_staged_marker_rejects_key_outside_prefix(self, tmp_path):
        with pytest.raises(ValueError):
            write_staged_marker(str(tmp_path / "m"), "blobs/1/2.blob", 1)

    def test_failed_staging_key_delete_is_logged(
        self, storage, s3_client, tmp_path, caplog
    ):
        staging_key = "tus-staging/undeletable"
        _seed_staging_object(s3_client, staging_key, b"data")
        marker_path = _make_marker(tmp_path, staging_key, 4)

        def fail(key):
            raise RuntimeError("boom")

        s3_client.delete_object = fail
        self._commit(storage, p64(1), marker_path)

        assert "Failed to delete S3 staging key tus-staging/undeletable" in caplog.text
