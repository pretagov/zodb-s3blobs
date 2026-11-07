"""The compare CLI, against fake Postgres connections and moto S3."""

from moto import mock_aws
from types import ModuleType
from ZODB.utils import p64
from zodb_s3blobs import compare
from zodb_s3blobs.migrate import s3_key_for
from zodb_s3blobs.s3client import S3Client

import boto3
import json
import pytest
import sys


@pytest.fixture
def s3_client():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        yield S3Client(bucket_name="test-bucket", region_name="us-east-1")


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    for name in ("S3_BLOB_BUCKET", "S3_BLOB_REGION", "S3_BLOB_PREFIX"):
        monkeypatch.delenv(name, raising=False)


class FakeCursor:
    """Answers the queries ``compare`` makes from ``(zoid, tid)`` tables."""

    def __init__(self, tables):
        self.tables = tables
        self.rows = []
        self.itersize = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def execute(self, sql, params=None):
        if "information_schema.tables" in sql:
            self.rows = [(1,)] if params[0] in self.tables else []
        elif sql.startswith("SELECT count(*) FROM "):
            self.rows = [(len(self.tables[sql.rsplit(" ", 1)[1]]),)]
        elif sql == "SELECT count(DISTINCT zoid) FROM object_state":
            self.rows = [(len({z for z, _ in self.tables["object_state"]}),)]
        elif sql == "SELECT min(tid), max(tid) FROM object_state":
            tids = [t for _, t in self.tables["object_state"]]
            self.rows = [(min(tids), max(tids)) if tids else (None, None)]
        elif sql == "SELECT count(DISTINCT (zoid, tid)) FROM blob_chunk":
            self.rows = [(len(set(self.tables["blob_chunk"])),)]
        elif sql == "SELECT zoid, tid FROM object_state":
            self.rows = list(self.tables["object_state"])
        elif sql == "SELECT DISTINCT zoid, tid FROM blob_chunk":
            self.rows = sorted(set(self.tables["blob_chunk"]))
        else:  # pragma: no cover
            raise AssertionError(f"unexpected query {sql!r}")


class FakeConn:
    def __init__(self, **tables):
        self.tables = tables
        self.closed = False

    def cursor(self, name=None):
        return FakeCursor(self.tables)

    def close(self):
        self.closed = True


def put_blob(s3_client, zoid, tid):
    s3_client._client.put_object(
        Bucket="test-bucket", Key=s3_key_for(p64(zoid), p64(tid)), Body=b"x"
    )


def full_db():
    return FakeConn(
        object_state=[(1, 0x100), (2, 0x200), (3, 0x200)],
        current_object=[(1, 0x100), (2, 0x200), (3, 0x200)],
        transaction=[(0x100,), (0x200,)],
        blob_chunk=[(2, 0x200), (2, 0x200), (3, 0x200)],
    )


def test_collect_table_counts():
    counts = compare.collect_table_counts(full_db())
    assert counts == {
        "object_state": 3,
        "current_object": 3,
        "transaction": 2,
        "blob_chunk": 3,
        "object_state_distinct_zoid": 3,
        "object_state_tid_min": 0x100,
        "object_state_tid_max": 0x200,
        "blob_chunk_distinct_pairs": 2,
    }


def test_collect_table_counts_missing_tables():
    counts = compare.collect_table_counts(FakeConn(object_state=[]))
    assert counts == {
        "object_state": 0,
        "current_object": None,
        "transaction": None,
        "blob_chunk": None,
        "object_state_distinct_zoid": 0,
        "object_state_tid_min": None,
        "object_state_tid_max": None,
    }


def test_fetch_s3_blob_pairs_ignores_other_keys(s3_client):
    put_blob(s3_client, 2, 0x200)
    s3_client._client.put_object(
        Bucket="test-bucket", Key="blobs/2/notes.txt", Body=b"x"
    )
    assert compare.fetch_s3_blob_pairs(s3_client) == {(2, 0x200): "blobs/2/200.blob"}


def test_open_dsn(monkeypatch):
    psycopg2 = ModuleType("psycopg2")
    psycopg2.connect = lambda dsn: ("connected", dsn)
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2)
    assert compare._open_dsn("host=db") == ("connected", "host=db")


def run_main(monkeypatch, conns, *args):
    monkeypatch.setattr(compare, "_open_dsn", conns.__getitem__)
    argv = [
        "--bucket",
        "test-bucket",
        "--region",
        "us-east-1",
        "--source-dsn",
        "source",
        *args,
    ]
    return compare.main(argv)


def test_main_no_differences(s3_client, monkeypatch, tmp_path, capsys):
    put_blob(s3_client, 2, 0x200)
    put_blob(s3_client, 3, 0x200)
    conns = {"source": full_db(), "dest": full_db()}
    report_path = tmp_path / "report.json"

    result = run_main(
        monkeypatch,
        conns,
        "--dest-dsn",
        "dest",
        "--json-report",
        str(report_path),
        "-v",
    )

    assert result == 0
    out = capsys.readouterr().out
    assert "OK: no differences detected" in out
    assert "s3_blob_keys" in out
    assert conns["source"].closed and conns["dest"].closed
    report = json.loads(report_path.read_text())
    assert report["s3_blob_keys"] == 2
    assert report["blob_diff"]["count_only_in_source"] == 0
    assert report["object_state_diff"]["count_dest"] == 3


def test_main_reports_differences(s3_client, monkeypatch, capsys):
    # Blob 3 never reached S3; 7, 8 and 9 are in S3 but not the source.
    put_blob(s3_client, 2, 0x200)
    for zoid in (7, 8, 9):
        put_blob(s3_client, zoid, 0x300)
    source = full_db()
    source.tables["blob_chunk"].append((4, 0x400))
    dest = FakeConn(object_state=[(1, 0x100), (5, 0x500), (6, 0x500)])
    conns = {"source": source, "dest": dest}

    result = run_main(monkeypatch, conns, "--dest-dsn", "dest", "--diff-limit", "1")

    assert result == 1
    out = capsys.readouterr().out
    assert "FAIL: 9 differing rows across all checks" in out
    assert "only in source.blob_chunk: 2" in out
    assert "only in dest.s3_keys: 3" in out
    assert "only in source.object_state: 2" in out
    assert "only in dest.object_state: 2" in out
    assert "zoid=3 tid=200" in out
    assert "... and 2 more" in out
    assert "... and 1 more" in out


def test_main_without_a_dest_db(s3_client, monkeypatch, capsys):
    put_blob(s3_client, 2, 0x200)
    put_blob(s3_client, 3, 0x200)

    result = run_main(monkeypatch, {"source": full_db()})

    assert result == 0
    out = capsys.readouterr().out
    assert "object_state ↔ dest object_state" not in out
    assert "OK: no differences detected" in out


def test_main_needs_a_bucket(capsys):
    assert compare.main(["--source-dsn", "source"]) == 2
    assert "--bucket" in capsys.readouterr().err
