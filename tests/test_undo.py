"""Transactional undo of blob revisions.

A history-preserving base storage undoes a transaction by writing the
previous state of each object again, under the undo's own transaction id.
For a blob, that state needs the blob of the revision it restores under
the new key ``blobs/<oid>/<undo tid>.blob``.
"""

from moto import mock_aws
from ZODB.blob import Blob
from ZODB.FileStorage import FileStorage
from ZODB.POSException import UndoError
from zodb_s3blobs.cache import S3BlobCache
from zodb_s3blobs.s3client import S3Client
from zodb_s3blobs.storage import _oid_hex
from zodb_s3blobs.storage import _tid_hex
from zodb_s3blobs.storage import S3BlobStorage

import boto3
import pytest
import time
import transaction
import ZODB


@pytest.fixture
def s3_client():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        yield S3Client(bucket_name="test-bucket", region_name="us-east-1")


def _file_storage(s3_client, tmp_path):
    return S3BlobStorage(
        FileStorage(str(tmp_path / "Data.fs")),
        s3_client,
        S3BlobCache(str(tmp_path / "cache"), max_size=10 * 1024 * 1024),
        temp_dir=str(tmp_path / "staging"),
    )


def _relstorage(s3_client, tmp_path):
    pytest.importorskip("relstorage")
    from relstorage.adapters.sqlite.adapter import Sqlite3Adapter
    from relstorage.options import Options
    from relstorage.storage import RelStorage

    options = Options(keep_history=True)
    adapter = Sqlite3Adapter(data_dir=str(tmp_path / "db"), pragmas={}, options=options)
    return S3BlobStorage(
        RelStorage(adapter, options=options),
        s3_client,
        S3BlobCache(str(tmp_path / "cache"), max_size=10 * 1024 * 1024),
        temp_dir=str(tmp_path / "staging"),
    )


@pytest.fixture(params=[_file_storage, _relstorage], ids=["FileStorage", "RelStorage"])
def db(request, s3_client, tmp_path):
    database = ZODB.DB(request.param(s3_client, tmp_path))
    yield database
    database.close()


def _commit(note):
    transaction.get().note(note)
    transaction.commit()
    # Distinct transaction ids and undo log timestamps.
    time.sleep(0.01)


def _undo_last(db):
    db.undo(db.undoLog(0, 1)[0]["id"])
    _commit("undo")


def _read(db, name):
    conn = db.open()
    try:
        with conn.root()[name].open() as f:
            return f.read()
    finally:
        conn.close()


def _write(db, name, data, note):
    conn = db.open()
    try:
        root = conn.root()
        if name in root:
            with root[name].open("w") as f:
                f.write(data)
        else:
            root[name] = Blob(data)
        _commit(note)
    finally:
        conn.close()


def test_undo_restores_previous_blob_revision(db):
    _write(db, "b", b"one", "create")
    _write(db, "b", b"two", "replace")

    _undo_last(db)

    assert _read(db, "b") == b"one"


def test_undo_copies_the_restored_blob_to_the_undo_tid(db, s3_client):
    _write(db, "b", b"one", "create")
    _write(db, "b", b"two", "replace")
    conn = db.open()
    oid = conn.root()["b"]._p_oid
    conn.close()

    _undo_last(db)

    undo_tid = db.lastTransaction()
    assert f"blobs/{_oid_hex(oid)}/{_tid_hex(undo_tid)}.blob" in set(
        s3_client.list_objects("blobs/")
    )


def test_undo_of_an_undo_restores_the_undone_blob(db):
    _write(db, "b", b"one", "create")
    _write(db, "b", b"two", "replace")
    _undo_last(db)

    _undo_last(db)

    assert _read(db, "b") == b"two"


def test_undo_of_a_blob_creation_copies_nothing(db, s3_client):
    conn = db.open()
    conn.root()["other"] = 1
    _commit("unrelated")
    conn.close()
    _write(db, "b", b"one", "create")
    keys = set(s3_client.list_objects("blobs/"))

    _undo_last(db)

    conn = db.open()
    assert "b" not in conn.root()
    conn.close()
    assert set(s3_client.list_objects("blobs/")) == keys


def test_undo_fails_when_the_restored_blob_is_missing(db, s3_client):
    _write(db, "b", b"one", "create")
    conn = db.open()
    blob = conn.root()["b"]
    blob._p_activate()
    oid, serial = blob._p_oid, blob._p_serial
    conn.close()
    _write(db, "b", b"two", "replace")
    s3_client.delete_object(f"blobs/{_oid_hex(oid)}/{_tid_hex(serial)}.blob")
    keys = set(s3_client.list_objects("blobs/"))

    db.undo(db.undoLog(0, 1)[0]["id"])
    with pytest.raises(UndoError, match="missing from S3"):
        _commit("undo")
    transaction.abort()

    assert set(s3_client.list_objects("blobs/")) == keys
    assert _read(db, "b") == b"two"


def test_undo_of_a_later_transaction_leaves_the_blob(db):
    _write(db, "b", b"one", "create")
    _write(db, "b", b"two", "replace")
    conn = db.open()
    conn.root()["other"] = 1
    _commit("unrelated")
    conn.close()

    _undo_last(db)

    assert _read(db, "b") == b"two"


def test_undo_of_several_transactions_at_once(db):
    _write(db, "a", b"a1", "create a")
    _write(db, "b", b"b1", "create b")
    _write(db, "a", b"a2", "replace a")
    _write(db, "b", b"b2", "replace b")

    log = db.undoLog(0, 2)
    db.undoMultiple([entry["id"] for entry in log])
    _commit("undo both")

    assert _read(db, "a") == b"a1"
    assert _read(db, "b") == b"b1"


def test_aborted_undo_removes_the_copied_key(db, s3_client):
    _write(db, "b", b"one", "create")
    _write(db, "b", b"two", "replace")
    keys = set(s3_client.list_objects("blobs/"))

    db.undo(db.undoLog(0, 1)[0]["id"])
    txn = transaction.get()
    txn.note("undo, then fail")

    def fail_after_vote(*args):
        raise RuntimeError("fail after the vote")

    # A second data manager that fails its vote, after the undo's own vote
    # has copied the key.
    class Failing:
        def abort(self, txn):
            pass

        tpc_begin = commit = tpc_finish = tpc_abort = abort

        def tpc_vote(self, txn):
            fail_after_vote()

        def sortKey(self):
            return "~"  # after the storage's data manager

    txn.join(Failing())
    with pytest.raises(RuntimeError):
        transaction.commit()
    transaction.abort()

    assert set(s3_client.list_objects("blobs/")) == keys
    assert _read(db, "b") == b"two"
