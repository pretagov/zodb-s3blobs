"""The migration CLI's Postgres side and command line, against fakes.

``test_migrate.py`` replaces the database access with in-memory functions.
These tests run the real ``iter_blob_pairs``, ``_materialize_blob`` and
``blob_size_in_db`` against a fake RelStorage adapter whose cursor answers
the queries they make and whose connection serves large objects.
"""

from moto import mock_aws
from types import ModuleType
from types import SimpleNamespace
from ZODB.utils import p64
from zodb_s3blobs import migrate
from zodb_s3blobs.s3client import S3Client

import boto3
import os
import pytest
import sys


@pytest.fixture
def s3_client():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        yield S3Client(bucket_name="test-bucket", region_name="us-east-1")


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    for name in (
        "RELSTORAGE_DSN",
        "S3_BLOB_BUCKET",
        "S3_BLOB_REGION",
        "S3_BLOB_PREFIX",
    ):
        monkeypatch.delenv(name, raising=False)


class FakeLob:
    def __init__(self, data, fail_after=None):
        self.data = data
        self.pos = 0
        self.fail_after = fail_after
        self.closed = False

    def read(self, size):
        if self.fail_after is not None and self.pos >= self.fail_after:
            raise OSError("connection lost")
        chunk = self.data[self.pos : self.pos + size]
        self.pos += len(chunk)
        return chunk

    def seek(self, offset, whence):
        assert whence == 2
        self.pos = len(self.data) + offset

    def tell(self):
        return self.pos

    def close(self):
        self.closed = True


class FakeDB:
    """``blob_chunk`` rows and the large objects they point to."""

    def __init__(self):
        self.rows = []  # (zoid, tid, chunk_num, loid)
        self.lobs = {}  # loid -> bytes
        self.fail_after = None
        self.opened = 0
        self.closed = 0

    def add_blob(self, zoid, tid, *chunks):
        for num, data in enumerate(chunks):
            loid = 1000 + len(self.lobs)
            self.lobs[loid] = data
            # Insert out of order so the ORDER BY matters.
            self.rows.insert(0, (zoid, tid, num, loid))


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self.result = []

    def execute(self, sql, params=None):
        if sql == "SELECT DISTINCT zoid, tid FROM blob_chunk":
            self.result = sorted({(z, t) for z, t, _, _ in self.db.rows})
        elif sql.startswith(
            "SELECT chunk FROM blob_chunk WHERE zoid = %s AND tid = %s"
        ):
            zoid, tid = params
            rows = sorted(
                (n, loid) for z, t, n, loid in self.db.rows if (z, t) == (zoid, tid)
            )
            self.result = [(loid,) for _, loid in rows]
        else:  # pragma: no cover
            raise AssertionError(f"unexpected query {sql!r}")

    def fetchall(self):
        return list(self.result)


class FakeConn:
    def __init__(self, db):
        self.db = db
        self.lobs = []

    def lobject(self, loid, mode):
        assert mode == "rb"
        lob = FakeLob(self.db.lobs[loid], fail_after=self.db.fail_after)
        self.lobs.append(lob)
        return lob


class FakeConnManager:
    def __init__(self, db):
        self.db = db

    def open_for_load(self):
        self.db.opened += 1
        return FakeConn(self.db), FakeCursor(self.db)

    def close(self, conn, cursor):
        self.db.closed += 1


def fake_storage(db):
    storage = SimpleNamespace(
        _adapter=SimpleNamespace(connmanager=FakeConnManager(db)), closed=False
    )

    def close():
        storage.closed = True

    storage.close = close
    return storage


@pytest.fixture
def db():
    db = FakeDB()
    db.add_blob(1, 0x100, b"hello ", b"world")
    db.add_blob(2, 0x200, b"x" * 100)
    return db


def test_iter_blob_pairs(db):
    pairs = list(migrate.iter_blob_pairs(fake_storage(db)))
    assert pairs == [(p64(1), p64(0x100)), (p64(2), p64(0x200))]
    assert db.opened == db.closed == 1


def test_materialize_joins_the_chunks_in_order(db, tmp_path, monkeypatch):
    monkeypatch.setattr(migrate, "_LOB_READ_CHUNK_BYTES", 4)
    path, size = migrate._materialize_blob(
        fake_storage(db), p64(1), p64(0x100), str(tmp_path)
    )
    with open(path, "rb") as f:
        assert f.read() == b"hello world"
    assert size == 11
    assert db.opened == db.closed == 1


def test_materialize_missing_blob(db, tmp_path):
    with pytest.raises(LookupError, match="vanished"):
        migrate._materialize_blob(fake_storage(db), p64(9), p64(9), str(tmp_path))
    assert os.listdir(tmp_path) == []
    assert db.opened == db.closed == 1


def test_materialize_removes_the_temp_file_on_error(db, tmp_path):
    db.fail_after = 0
    with pytest.raises(OSError, match="connection lost"):
        migrate._materialize_blob(fake_storage(db), p64(2), p64(0x200), str(tmp_path))
    assert os.listdir(tmp_path) == []
    assert db.opened == db.closed == 1


def test_blob_size_in_db(db):
    assert migrate.blob_size_in_db(fake_storage(db), p64(1), p64(0x100)) == 11
    with pytest.raises(LookupError, match="vanished"):
        migrate.blob_size_in_db(fake_storage(db), p64(9), p64(9))
    assert db.opened == db.closed == 2


def test_run_migration_copies_the_bytes(db, s3_client):
    counts = migrate.run_migration(fake_storage(db), s3_client, workers=2)
    assert counts["uploaded"] == 2
    assert counts["bytes_uploaded"] == 111
    body = s3_client._client.get_object(
        Bucket="test-bucket", Key=migrate.s3_key_for(p64(1), p64(0x100))
    )["Body"].read()
    assert body == b"hello world"


def test_run_migration_with_more_blobs_than_workers(s3_client):
    db = FakeDB()
    for zoid in range(1, 13):
        db.add_blob(zoid, 0x100, b"blob %d" % zoid)
    counts = migrate.run_migration(
        fake_storage(db), s3_client, workers=1, progress_interval=1
    )
    assert counts["scanned"] == counts["uploaded"] == 12
    assert migrate.count_s3_keys(s3_client) == 12


def test_verify_size_skips_a_matching_key(db, s3_client):
    storage = fake_storage(db)
    migrate.run_migration(storage, s3_client, workers=1)
    counts = migrate.run_migration(storage, s3_client, workers=1, verify_size=True)
    assert counts["skipped"] == 2
    assert counts["uploaded"] == 0


def test_verify_size_keeps_a_key_whose_rows_are_gone(db, s3_client):
    key = migrate.s3_key_for(p64(9), p64(9))
    s3_client._client.put_object(Bucket="test-bucket", Key=key, Body=b"abc")
    status = migrate.upload_one(
        s3_client,
        fake_storage(db),
        p64(9),
        p64(9),
        overwrite=False,
        dry_run=False,
        verify_size=True,
        tmp_dir=None,
    )
    assert status == ("stale", 3)


def test_dry_run_with_verify_size_reports_the_size(db, s3_client):
    counts = migrate.run_migration(
        fake_storage(db), s3_client, dry_run=True, verify_size=True
    )
    assert counts["would_upload"] == 2
    assert migrate.count_s3_keys(s3_client) == 0


def test_dry_run_stale(db, s3_client, monkeypatch):
    """Rows listed by the scan but gone (a pack) before the size check."""
    list_pairs = FakeCursor.execute

    def execute(self, sql, params=None):
        list_pairs(self, sql, params)
        if sql.startswith("SELECT DISTINCT"):
            db.rows.clear()

    monkeypatch.setattr(FakeCursor, "execute", execute)
    counts = migrate.run_migration(
        fake_storage(db), s3_client, dry_run=True, verify_size=True
    )
    assert counts["stale"] == 2


def test_open_relstorage(monkeypatch):
    calls = {}

    class Options:
        def __init__(self, **kw):
            calls["options"] = kw

    class PostgreSQLAdapter:
        def __init__(self, dsn, options):
            calls["dsn"] = dsn

    class RelStorage:
        def __init__(self, adapter, name, options):
            calls["name"] = name

    for name, attrs in (
        ("relstorage", {}),
        ("relstorage.adapters", {}),
        ("relstorage.adapters.postgresql", {"PostgreSQLAdapter": PostgreSQLAdapter}),
        ("relstorage.options", {"Options": Options}),
        ("relstorage.storage", {"RelStorage": RelStorage}),
    ):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    storage = migrate.open_relstorage("host=db", "/cache", 1024, keep_history=True)
    assert isinstance(storage, RelStorage)
    assert calls["dsn"] == "host=db"
    assert calls["name"] == "zodb-s3blobs-migrate"
    assert calls["options"] == {
        "keep_history": True,
        "blob_dir": "/cache",
        "blob_cache_size": 1024,
        "shared_blob_dir": False,
    }


KEEPALIVES = {"keepalives": 1, "idle": 30, "interval": 10, "count": 3}


def test_keepalives_added_to_a_key_value_dsn():
    assert migrate._augment_dsn_with_keepalives(
        "host=db keepalives_idle=5", **KEEPALIVES
    ) == (
        "host=db keepalives_idle=5 keepalives=1 keepalives_interval=10 keepalives_count=3"
    )


def test_keepalives_already_in_the_dsn():
    dsn = "host=db keepalives=0 keepalives_idle=1 keepalives_interval=2 keepalives_count=3"
    assert migrate._augment_dsn_with_keepalives(dsn, **KEEPALIVES) == dsn


def test_keepalives_empty_dsn():
    assert migrate._augment_dsn_with_keepalives("", **KEEPALIVES) == (
        "keepalives=1 keepalives_idle=30 keepalives_interval=10 keepalives_count=3"
    )


def test_keepalives_added_to_a_url_dsn():
    dsn = migrate._augment_dsn_with_keepalives(
        "postgresql://u:p@db:5432/zodb?sslmode=require&keepalives=0", **KEEPALIVES
    )
    assert dsn == (
        "postgresql://u:p@db:5432/zodb?sslmode=require&keepalives=0"
        "&keepalives_idle=30&keepalives_interval=10&keepalives_count=3"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("1GB", 1024**3),
        ("512mb", 512 * 1024**2),
        ("1.5K", 1536),
        ("2g", 2 * 1024**3),
        ("3M", 3 * 1024**2),
        ("4KB", 4096),
        ("1048576", 1048576),
    ],
)
def test_parse_size(value, expected):
    assert migrate._parse_size(value) == expected


def test_main_needs_a_bucket(capsys):
    assert migrate.main(["--blob-cache-dir", "/tmp"]) == 2
    assert "--bucket" in capsys.readouterr().err


def test_main_needs_a_dsn(s3_client, capsys):
    argv = [
        "--blob-cache-dir",
        "/tmp",
        "--bucket",
        "test-bucket",
        "--region",
        "us-east-1",
    ]
    assert migrate.main(argv) == 2
    assert "--dsn" in capsys.readouterr().err


def test_main_count_only(s3_client, capsys):
    s3_client._client.put_object(
        Bucket="test-bucket", Key="blobs/1/100.blob", Body=b"x"
    )
    argv = [
        "--blob-cache-dir",
        "/tmp",
        "--bucket",
        "test-bucket",
        "--region",
        "us-east-1",
        "--count-only",
    ]
    assert migrate.main(argv) == 0
    assert "key count: 1" in capsys.readouterr().out


def test_main_runs_the_migration(db, s3_client, monkeypatch):
    storage = fake_storage(db)
    opened = {}

    def open_relstorage(**kw):
        opened.update(kw)
        return storage

    monkeypatch.setattr(migrate, "open_relstorage", open_relstorage)
    monkeypatch.setenv("RELSTORAGE_DSN", "host=db")
    argv = [
        "--blob-cache-dir",
        "/cache",
        "--blob-cache-size",
        "512MB",
        "--bucket",
        "test-bucket",
        "--region",
        "us-east-1",
        "--workers",
        "2",
        "-v",
    ]
    assert migrate.main(argv) == 0
    assert opened["dsn"].startswith("host=db keepalives=1 ")
    assert opened["blob_cache_size"] == 512 * 1024**2
    assert opened["keep_history"] is False
    assert storage.closed
    assert migrate.count_s3_keys(s3_client) == 2
