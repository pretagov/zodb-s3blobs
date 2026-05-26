"""CLI for migrating existing RelStorage blobs into S3.

The typical deployment runs RelStorage with ``blob-cache-size`` set and no 
``shared-blob-dir true``, which stores blob bytes in the Postgres 
``blob_chunk`` table and uses the local blob-dir only as an on-disk LRU cache. 
To migrate such a deployment to ``S3BlobStorage``, every ``(zoid, tid)`` pair 
in ``blob_chunk`` must be copied into S3 under the key scheme 
``blobs/{_oid_hex(oid)}/{_tid_hex(tid)}.blob`` that ``S3BlobStorage.loadBlob`` 
reads. This module does exactly that.

The CLI opens a RelStorage using the same options the container uses, lists
blob pairs via a SELECT over ``blob_chunk``, materialises each blob by
reading the underlying Postgres large objects directly, and uploads each
to S3 via ``S3Client``. Re-runs are idempotent: keys already present in
S3 are skipped, optionally with a size check.

Going around RelStorage's own ``loadBlob`` is deliberate. Two reasons:

1. RelStorage 3+ assumes every ``(zoid, tid)`` pair has exactly one row in
   ``blob_chunk`` (the "we no longer chunk blobs" assertion in the Postgres
   mover). Older installs may still have multi-chunk legacy data that
   triggers ``AssertionError`` in that path.
2. Using ``loadBlob`` routes through the shared on-disk blob cache, whose
   background cleanup thread can unlink a file while ``upload_file`` is
   still trying to open it. Reading the LOB bytes into a per-task temp
   file sidesteps that race entirely.
"""

from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from concurrent.futures import wait
from zodb_s3blobs.s3client import S3Client
from zodb_s3blobs.storage import _oid_hex
from zodb_s3blobs.storage import _tid_hex

import argparse
import contextlib
import logging
import os
import sys
import tempfile
import ZODB.utils


_LOB_READ_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MiB


logger = logging.getLogger("zodb_s3blobs.migrate")


def s3_key_for(oid, tid):
    """Return the S3 key ``S3BlobStorage`` uses for a given (oid, tid).

    Kept as a top-level helper so the migration and storage cannot diverge:
    any future change to ``_oid_hex`` / ``_tid_hex`` is picked up here
    automatically.
    """
    return f"blobs/{_oid_hex(oid)}/{_tid_hex(tid)}.blob"


def iter_blob_pairs(storage):
    """Yield ``(oid_bytes, tid_bytes, None)`` for every row in ``blob_chunk``.

    Uses the storage's adapter cursor so we inherit its DSN, isolation, and
    driver choice. The query works on both history-free and history-preserving
    schemas (both carry a ``tid`` column; see relstorage/adapters/schema.py).

    The third tuple element is always ``None`` here — bytes have to be
    materialised from the DB on demand. The shape matches
    ``iter_blob_pairs_from_disk`` so ``run_migration`` can consume either.
    """
    conn, cursor = storage._adapter.connmanager.open_for_load()
    try:
        cursor.execute("SELECT DISTINCT zoid, tid FROM blob_chunk")
        rows = cursor.fetchall()
    finally:
        storage._adapter.connmanager.close(conn, cursor)
    for zoid, tid in rows:
        yield ZODB.utils.p64(int(zoid)), ZODB.utils.p64(int(tid)), None


def iter_blob_pairs_from_disk(blob_dir):
    """Yield ``(oid_bytes, tid_bytes, local_path)`` for every blob file on disk.

    Assumes ZODB's ``BushyLayout``: each blob lives at
    ``<blob_dir>/0xNN/0xNN/0xNN/0xNN/0xNN/0xNN/0xNN/0xNN/0x<tid_hex>.blob``
    — eight directory levels, one per byte of the 64-bit oid, followed
    by a file whose stem is the tid in lowercase hex (with optional
    ``0x`` prefix). Anything that doesn't match that shape is skipped
    with a debug log; that covers the ``.layout`` marker, the ``tmp/``
    workspace RelStorage uses for in-flight uploads, and any stray
    files left from earlier crashes.
    """
    blob_dir = os.path.abspath(blob_dir)
    for dirpath, _dirnames, filenames in os.walk(blob_dir):
        rel = os.path.relpath(dirpath, blob_dir)
        if rel == ".":
            continue
        parts = rel.split(os.sep)
        if len(parts) != 8 or not all(
            p.startswith("0x") and len(p) == 4 for p in parts
        ):
            continue
        try:
            oid = bytes(int(p[2:], 16) for p in parts)
        except ValueError:
            logger.debug("skipping unparseable oid path %r", rel)
            continue

        for name in filenames:
            if not name.endswith(".blob"):
                continue
            stem = name[: -len(".blob")]
            if stem.startswith("0x"):
                stem = stem[2:]
            try:
                tid = bytes.fromhex(stem)
            except ValueError:
                logger.debug("skipping unparseable tid filename %r", name)
                continue
            if len(tid) != 8:
                logger.debug("skipping non-8-byte tid filename %r", name)
                continue
            yield oid, tid, os.path.join(dirpath, name)


def _fetch_chunk_loids(cursor, oid, tid):
    """Return the list of LOB OIDs for ``(oid, tid)`` in chunk order.

    Postgres RelStorage stores ``blob_chunk.chunk`` as an ``OID`` column —
    each row references a server-side large object holding that chunk's
    bytes. For recent RelStorage writes there is only one chunk per
    (zoid, tid); older installs may have multiple.
    """
    cursor.execute(
        "SELECT chunk FROM blob_chunk "
        "WHERE zoid = %s AND tid = %s ORDER BY chunk_num",
        (ZODB.utils.u64(oid), ZODB.utils.u64(tid)),
    )
    return [row[0] for row in cursor.fetchall()]


def _materialize_blob(storage, oid, tid, tmp_dir):
    """Stream all blob chunks for (oid, tid) into a fresh temp file.

    Returns ``(local_path, size_bytes)``. The caller is responsible for
    unlinking the file. Raises ``LookupError`` if the blob has no rows in
    ``blob_chunk`` (already deleted between the enumeration SELECT and
    this call, e.g. during a concurrent pack).
    """
    conn, cursor = storage._adapter.connmanager.open_for_load()
    try:
        loids = _fetch_chunk_loids(cursor, oid, tid)
        if not loids:
            raise LookupError(
                f"blob_chunk rows vanished for oid={_oid_hex(oid)} "
                f"tid={_tid_hex(tid)}"
            )
        fd, path = tempfile.mkstemp(
            suffix=".blob.tmp", prefix="zs3b-migrate-", dir=tmp_dir
        )
        total = 0
        try:
            with os.fdopen(fd, "wb") as out:
                for loid in loids:
                    lob = conn.lobject(loid, "rb")
                    try:
                        while True:
                            data = lob.read(_LOB_READ_CHUNK_BYTES)
                            if not data:
                                break
                            out.write(data)
                            total += len(data)
                    finally:
                        lob.close()
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(path)
            raise
    finally:
        storage._adapter.connmanager.close(conn, cursor)
    return path, total


def blob_size_in_db(storage, oid, tid):
    """Return the sum of LOB lengths for (oid, tid) in ``blob_chunk``.

    Used by ``--verify-size`` to catch prior partial uploads whose S3
    ContentLength does not match the canonical size.
    """
    conn, cursor = storage._adapter.connmanager.open_for_load()
    try:
        loids = _fetch_chunk_loids(cursor, oid, tid)
        total = 0
        for loid in loids:
            lob = conn.lobject(loid, "rb")
            try:
                lob.seek(0, 2)  # SEEK_END
                total += lob.tell()
            finally:
                lob.close()
    finally:
        storage._adapter.connmanager.close(conn, cursor)
    return int(total)


def upload_one(
    s3_client,
    storage,
    oid,
    tid,
    *,
    overwrite,
    dry_run,
    verify_size,
    tmp_dir,
    local_path=None,
):
    """Migrate one (oid, tid) blob into S3. Returns (status, size_bytes).

    status ∈ {"uploaded", "skipped", "would_upload",
              "size_mismatch_reuploaded", "stale"}.

    When ``local_path`` is provided, the blob is read straight from that
    path (shared-blob-dir migrations). Otherwise blob bytes are
    materialised from Postgres via ``_materialize_blob``. In both modes
    the resulting upload uses the same S3 key.

    Any other exception bubbles up so the caller can record a per-task failure.
    """
    key = s3_key_for(oid, tid)
    oid_hex = _oid_hex(oid)
    tid_hex = _tid_hex(tid)

    def _local_size():
        """Size of the canonical blob bytes (disk file or DB LOBs)."""
        if local_path is not None:
            return os.path.getsize(local_path)
        return blob_size_in_db(storage, oid, tid)

    size_mismatch = False
    if not overwrite:
        meta = s3_client.head_object(key)
        if meta is not None:
            remote_size = int(meta.get("ContentLength", -1))
            if verify_size:
                try:
                    local_size = _local_size()
                except (LookupError, FileNotFoundError):
                    logger.warning(
                        "stale source oid=%s tid=%s — keeping existing S3 key",
                        oid_hex,
                        tid_hex,
                    )
                    return ("stale", remote_size if remote_size >= 0 else 0)
                if remote_size != local_size:
                    logger.warning(
                        "size mismatch oid=%s tid=%s remote=%d source=%d — re-uploading",
                        oid_hex,
                        tid_hex,
                        remote_size,
                        local_size,
                    )
                    size_mismatch = True
                    # fall through to re-upload
                else:
                    return ("skipped", remote_size)
            else:
                return ("skipped", remote_size if remote_size >= 0 else 0)

    if dry_run:
        try:
            size = _local_size() if verify_size else 0
        except (LookupError, FileNotFoundError):
            return ("stale", 0)
        logger.debug(
            "would upload oid=%s tid=%s key=%s size=%d", oid_hex, tid_hex, key, size
        )
        return ("would_upload", size)

    if local_path is not None:
        # Disk-backed source: upload the canonical file directly. Don't
        # delete it — the on-disk directory is the authoritative store
        # until the operator decides to remove it.
        try:
            size = os.path.getsize(local_path)
        except FileNotFoundError:
            logger.warning("disk blob vanished before upload: %s", local_path)
            return ("stale", 0)
        s3_client.upload_file(local_path, key)
    else:
        try:
            mat_path, size = _materialize_blob(storage, oid, tid, tmp_dir)
        except LookupError as exc:
            logger.warning("skipping vanished blob: %s", exc)
            return ("stale", 0)
        try:
            s3_client.upload_file(mat_path, key)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(mat_path)

    status = "size_mismatch_reuploaded" if size_mismatch else "uploaded"
    logger.debug("%s oid=%s tid=%s key=%s size=%d", status, oid_hex, tid_hex, key, size)
    return (status, size)


def run_migration(
    storage,
    s3_client,
    *,
    source_iter=None,
    workers=8,
    overwrite=False,
    dry_run=False,
    verify_size=False,
    progress_interval=500,
    tmp_dir=None,
):
    """Walk every (oid, tid) blob pair from ``source_iter`` into S3.

    ``source_iter`` must yield ``(oid, tid, local_path_or_None)`` triples.
    Defaults to ``iter_blob_pairs(storage)`` for the DB-backed mode.
    Pass ``iter_blob_pairs_from_disk(blob_dir)`` for shared-blob-dir
    migrations; in that case ``storage`` may be ``None`` (no DB access
    is needed once the iterator yields canonical disk paths).

    Returns a dict of counters. Raises at the end if any task failed.
    """
    counts = {
        "scanned": 0,
        "uploaded": 0,
        "skipped": 0,
        "would_upload": 0,
        "size_mismatch_reuploaded": 0,
        "stale": 0,
        "failed": 0,
        "bytes_uploaded": 0,
    }
    failures = []

    if source_iter is None:
        source_iter = iter_blob_pairs(storage)

    # Each materialisation writes a throwaway temp file; using a dedicated
    # directory makes it easy to pre-size the disk and clean up after a
    # crash (``rm -rf``). Defaults to the system temp dir. Disk-backed
    # uploads don't use it, but creating it is cheap.
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="zs3b-migrate-")
        _owns_tmp = True
    else:
        os.makedirs(tmp_dir, exist_ok=True)
        _owns_tmp = False

    def _submit(ex, oid, tid, local_path):
        return ex.submit(
            upload_one,
            s3_client,
            storage,
            oid,
            tid,
            overwrite=overwrite,
            dry_run=dry_run,
            verify_size=verify_size,
            tmp_dir=tmp_dir,
            local_path=local_path,
        )

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            # Feed lazily: walk the source iterator, keep at most
            # ``workers * 4`` in flight so memory / temp-file disk use stays
            # bounded on large dumps.
            in_flight = set()
            max_in_flight = max(workers * 4, workers + 1)

            for oid, tid, local_path in source_iter:
                counts["scanned"] += 1
                while len(in_flight) >= max_in_flight:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        _record(fut, counts, failures)
                    if counts["scanned"] % progress_interval == 0:
                        _log_progress(counts)
                in_flight.add(_submit(ex, oid, tid, local_path))

            for fut in as_completed(in_flight):
                _record(fut, counts, failures)
    finally:
        if _owns_tmp:
            with contextlib.suppress(OSError):
                os.rmdir(tmp_dir)  # non-empty only if something leaked

    _log_progress(counts, final=True)

    if failures:
        logger.error("migration completed with %d failures", len(failures))
        for oid_hex, tid_hex, exc in failures[:20]:
            logger.error("  failed oid=%s tid=%s: %s", oid_hex, tid_hex, exc)
        raise RuntimeError(f"migration had {len(failures)} failures")

    return counts


def _record(fut, counts, failures):
    try:
        status, size = fut.result()
    except Exception as exc:  # noqa: BLE001 — we want to record any failure
        counts["failed"] += 1
        # We don't have the (oid, tid) handy here; rely on upload_one's
        # DEBUG log plus the exception text for correlation.
        failures.append(("?", "?", exc))
        logger.exception("upload task failed: %s", exc)
        return
    counts[status] = counts.get(status, 0) + 1
    if status in ("uploaded", "size_mismatch_reuploaded"):
        counts["bytes_uploaded"] += size


def _log_progress(counts, final=False):
    level = logging.INFO
    logger.log(
        level,
        "%s scanned=%d uploaded=%d skipped=%d would_upload=%d "
        "size_mismatch_reuploaded=%d stale=%d failed=%d bytes_uploaded=%d",
        "DONE" if final else "progress",
        counts["scanned"],
        counts["uploaded"],
        counts["skipped"],
        counts["would_upload"],
        counts["size_mismatch_reuploaded"],
        counts["stale"],
        counts["failed"],
        counts["bytes_uploaded"],
    )


def count_s3_keys(s3_client):
    """Return the number of S3 keys under ``blobs/``."""
    return sum(1 for _ in s3_client.list_objects("blobs/"))


def open_relstorage(dsn, blob_cache_dir, blob_cache_size, keep_history=False):
    """Open a RelStorage against Postgres using the given DSN.

    Kept small on purpose: full ZConfig parsing is avoided; the operator
    passes the same DSN and cache directory the container uses.
    """
    from relstorage.adapters.postgresql import PostgreSQLAdapter
    from relstorage.options import Options
    from relstorage.storage import RelStorage

    options = Options(
        keep_history=keep_history,
        blob_dir=blob_cache_dir,
        blob_cache_size=blob_cache_size,
        shared_blob_dir=False,
    )
    adapter = PostgreSQLAdapter(dsn=dsn, options=options)
    return RelStorage(adapter, name="zodb-s3blobs-migrate", options=options)


# ---- argparse / main -------------------------------------------------------


def _env(name, default=None):
    return os.environ.get(name, default)


def _augment_dsn_with_keepalives(dsn, *, keepalives, idle, interval, count):
    """Append libpq tcp keepalive params to ``dsn`` and return the result.

    Handles both DSN forms libpq accepts: space-separated ``key=value`` and
    URL (``postgres://`` / ``postgresql://``). Existing keepalive params in
    the DSN are left untouched on the assumption that an explicit DSN value
    overrides the CLI default.
    """
    params = {
        "keepalives": str(keepalives),
        "keepalives_idle": str(idle),
        "keepalives_interval": str(interval),
        "keepalives_count": str(count),
    }

    if dsn.startswith(("postgres://", "postgresql://")):
        from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

        parsed = urlparse(dsn)
        existing = dict(parse_qsl(parsed.query, keep_blank_values=True))
        for k, v in params.items():
            existing.setdefault(k, v)
        return urlunparse(parsed._replace(query=urlencode(existing)))

    existing_keys = {
        token.split("=", 1)[0] for token in dsn.split() if "=" in token
    }
    suffix = " ".join(
        f"{k}={v}" for k, v in params.items() if k not in existing_keys
    )
    if not suffix:
        return dsn
    sep = "" if dsn.endswith(" ") or not dsn else " "
    return f"{dsn}{sep}{suffix}"


def _parse_size(s):
    """Parse "1GB" / "512MB" / "1048576" → bytes. Accepts common suffixes."""
    if s is None:
        return None
    s = str(s).strip().upper()
    for suffix, mult in (
        ("GB", 1024**3),
        ("MB", 1024**2),
        ("KB", 1024),
        ("G", 1024**3),
        ("M", 1024**2),
        ("K", 1024),
    ):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


def build_argparser():
    p = argparse.ArgumentParser(
        prog="zodb-s3blobs-migrate",
        description="Copy blobs from a RelStorage Postgres DB into S3.",
    )

    p.add_argument(
        "--dsn",
        default=_env("RELSTORAGE_DSN"),
        help="Postgres DSN for the source RelStorage (env RELSTORAGE_DSN). "
        "Example: \"host=... port=... dbname=... user=... password=...\"",
    )
    p.add_argument(
        "--blob-cache-dir",
        default=None,
        help="Local blob-cache directory the source RelStorage uses. "
        "Required for DB-backed migrations (no --from-disk); ignored when "
        "--from-disk is set.",
    )
    p.add_argument(
        "--from-disk",
        default=None,
        metavar="BLOB_DIR",
        help="Migrate blobs directly from a shared-blob-dir on disk "
        "instead of materialising them out of Postgres LOBs. The "
        "directory is expected to use ZODB's BushyLayout (the default "
        "for RelStorage with shared-blob-dir true). Source files are "
        "read but not modified or deleted.",
    )
    p.add_argument(
        "--blob-cache-size",
        default="1GB",
        help="Size of the on-disk cache RelStorage is allowed to use during the "
        "migration. Default: 1GB. Accepts 1GB / 512MB / bytes.",
    )
    p.add_argument(
        "--keep-history",
        action="store_true",
        help="Set if the source RelStorage runs in history-preserving mode. "
        "Default is history-free.",
    )

    p.add_argument("--bucket", default=_env("S3_BLOB_BUCKET"))
    p.add_argument("--endpoint-url", default=_env("S3_BLOB_ENDPOINT_URL"))
    p.add_argument("--region", default=_env("S3_BLOB_REGION"))
    p.add_argument("--prefix", default=_env("S3_BLOB_PREFIX", ""))
    p.add_argument("--access-key", default=_env("S3_ACCESS_KEY"))
    p.add_argument("--secret-key", default=_env("S3_SECRET_KEY"))
    p.add_argument("--sse-customer-key", default=_env("S3_BLOB_SSE_CUSTOMER_KEY"))
    p.add_argument(
        "--no-ssl", action="store_true", help="Disable TLS for the S3 endpoint."
    )
    p.add_argument(
        "--addressing-style",
        default="auto",
        choices=("auto", "path", "virtual"),
    )

    p.add_argument(
        "--keepalives",
        type=int,
        default=1,
        choices=(0, 1),
        help="libpq tcp keepalives flag. Default 1 (enabled). Set 0 to disable.",
    )
    p.add_argument(
        "--keepalives-idle",
        type=int,
        default=30,
        help="Seconds of inactivity before sending a keepalive. Default 30.",
    )
    p.add_argument(
        "--keepalives-interval",
        type=int,
        default=10,
        help="Seconds between keepalive retransmits. Default 10.",
    )
    p.add_argument(
        "--keepalives-count",
        type=int,
        default=3,
        help="Failed keepalives before the connection is dropped. Default 3.",
    )

    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-upload even if the key is already present in S3.",
    )
    p.add_argument(
        "--verify-size",
        action="store_true",
        help="When a key is already in S3, compare its ContentLength to the "
        "sum of blob_chunk lengths. Mismatches trigger a re-upload.",
    )
    p.add_argument(
        "--tmp-dir",
        default=None,
        help="Directory for per-task blob temp files. Defaults to a system "
        "tmp dir. Pre-size the filesystem for roughly workers * max_blob_size.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--count-only",
        action="store_true",
        help="Print the number of objects under blobs/ in S3 and exit.",
    )
    p.add_argument("--progress-interval", type=int, default=500)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.bucket:
        print("error: --bucket or S3_BLOB_BUCKET is required", file=sys.stderr)
        return 2

    s3 = S3Client(
        bucket_name=args.bucket,
        prefix=args.prefix or "",
        endpoint_url=args.endpoint_url,
        region_name=args.region,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
        use_ssl=not args.no_ssl,
        addressing_style=args.addressing_style,
        sse_customer_key=args.sse_customer_key,
    )

    if args.count_only:
        n = count_s3_keys(s3)
        print(f"s3 blobs/ key count: {n}")
        return 0

    if args.from_disk:
        if not os.path.isdir(args.from_disk):
            print(
                f"error: --from-disk path {args.from_disk!r} is not a directory",
                file=sys.stderr,
            )
            return 2
        # No DB access needed for shared-blob-dir migrations.
        run_migration(
            None,
            s3,
            source_iter=iter_blob_pairs_from_disk(args.from_disk),
            workers=args.workers,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            verify_size=args.verify_size,
            progress_interval=args.progress_interval,
            tmp_dir=args.tmp_dir,
        )
        return 0

    if not args.dsn:
        print("error: --dsn or RELSTORAGE_DSN is required", file=sys.stderr)
        return 2
    if not args.blob_cache_dir:
        print("error: --blob-cache-dir is required for DB-backed migration", file=sys.stderr)
        return 2

    dsn = _augment_dsn_with_keepalives(
        args.dsn,
        keepalives=args.keepalives,
        idle=args.keepalives_idle,
        interval=args.keepalives_interval,
        count=args.keepalives_count,
    )

    storage = open_relstorage(
        dsn=dsn,
        blob_cache_dir=args.blob_cache_dir,
        blob_cache_size=_parse_size(args.blob_cache_size),
        keep_history=args.keep_history,
    )
    try:
        run_migration(
            storage,
            s3,
            workers=args.workers,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            verify_size=args.verify_size,
            progress_interval=args.progress_interval,
            tmp_dir=args.tmp_dir,
        )
    finally:
        storage.close()

    return 0


# ---- purge -----------------------------------------------------------------
#
# After a migration has been verified, the blob bytes still in the source
# RelStorage Postgres DB can be reclaimed. Two storage locations carry
# blob data and both must be cleaned to avoid orphaned LOBs:
#
#   1. ``blob_chunk`` — one row per ``(zoid, tid, chunk_num)``; the
#      ``chunk`` column is a Postgres ``OID`` pointing at a server-side
#      large object.
#   2. ``pg_largeobject`` — holds the actual bytes for each LOB.
#      ``DELETE FROM blob_chunk`` alone leaves these orphaned; they have
#      to be released with ``lo_unlink``.
#
# Object pickles in ``object_state`` reference blobs only by
# ``(zoid, tid)`` — the same key the S3 storage uses — so they need no
# modification when the DB-side bytes are dropped.


def assert_temp_blob_chunk_empty(storage):
    """Refuse to purge if ``temp_blob_chunk`` is non-empty.

    Rows in ``temp_blob_chunk`` indicate an in-flight or interrupted
    commit. Dropping them would destroy state the server may still use
    for recovery. If the table doesn't exist (some schemas), treat as
    empty.
    """
    conn, cursor = storage._adapter.connmanager.open_for_load()
    try:
        try:
            cursor.execute("SELECT count(*) FROM temp_blob_chunk")
        except Exception as exc:
            # Table may not exist on all schemas; the connmanager's
            # cursor likely needs a rollback before reuse.
            with contextlib.suppress(Exception):
                conn.rollback()
            logger.debug("temp_blob_chunk probe skipped: %s", exc)
            return
        (count,) = cursor.fetchone()
    finally:
        storage._adapter.connmanager.close(conn, cursor)
    if count:
        raise RuntimeError(
            f"temp_blob_chunk has {count} row(s) — refusing to purge. "
            "This usually means an in-flight or interrupted commit. "
            "Stop the application, let any pending pack/commit finish, "
            "and re-run."
        )


def _verify_one(s3_client, storage, oid, tid, *, verify_size):
    """Return ``("ok"|"missing"|"size_mismatch", oid_hex, tid_hex)``."""
    oid_hex = _oid_hex(oid)
    tid_hex = _tid_hex(tid)
    meta = s3_client.head_object(s3_key_for(oid, tid))
    if meta is None:
        return ("missing", oid_hex, tid_hex)
    if verify_size:
        try:
            local_size = blob_size_in_db(storage, oid, tid)
        except LookupError:
            return ("ok", oid_hex, tid_hex)
        remote_size = int(meta.get("ContentLength", -1))
        if remote_size != local_size:
            return ("size_mismatch", oid_hex, tid_hex)
    return ("ok", oid_hex, tid_hex)


def verify_all_blobs_in_s3(
    storage,
    s3_client,
    *,
    verify_size=False,
    workers=8,
    progress_interval=500,
    sample_limit=20,
):
    """Pre-flight check that every blob in ``blob_chunk`` exists in S3.

    Returns ``(counts, problems)`` where ``problems`` is a list of
    ``(status, oid_hex, tid_hex)`` capped at ``sample_limit`` entries.
    The caller decides whether to abort.
    """
    counts = {"checked": 0, "ok": 0, "missing": 0, "size_mismatch": 0}
    problems = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        in_flight = set()
        max_in_flight = max(workers * 4, workers + 1)

        def _drain(done):
            for fut in done:
                status, oid_hex, tid_hex = fut.result()
                counts["checked"] += 1
                counts[status] = counts.get(status, 0) + 1
                if status != "ok" and len(problems) < sample_limit:
                    problems.append((status, oid_hex, tid_hex))
                if counts["checked"] % progress_interval == 0:
                    logger.info(
                        "verify progress checked=%d ok=%d missing=%d size_mismatch=%d",
                        counts["checked"],
                        counts["ok"],
                        counts["missing"],
                        counts["size_mismatch"],
                    )

        for oid, tid in iter_blob_pairs(storage):
            while len(in_flight) >= max_in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                _drain(done)
            in_flight.add(
                ex.submit(
                    _verify_one,
                    s3_client,
                    storage,
                    oid,
                    tid,
                    verify_size=verify_size,
                )
            )

        _drain(as_completed(in_flight))

    logger.info(
        "verify done checked=%d ok=%d missing=%d size_mismatch=%d",
        counts["checked"],
        counts["ok"],
        counts["missing"],
        counts["size_mismatch"],
    )
    return counts, problems


def run_purge(
    storage,
    s3_client,
    *,
    dsn,
    dry_run=False,
    verify_size=False,
    workers=8,
    batch_size=1000,
    progress_interval=500,
):
    """Verify and (optionally) delete blob data from the source DB.

    Raises ``RuntimeError`` if pre-flight checks fail. On success returns
    a counters dict.
    """
    logger.info("purge: checking temp_blob_chunk is empty")
    assert_temp_blob_chunk_empty(storage)

    logger.info(
        "purge: verifying every blob_chunk pair exists in S3 (verify_size=%s)",
        verify_size,
    )
    vcounts, problems = verify_all_blobs_in_s3(
        storage,
        s3_client,
        verify_size=verify_size,
        workers=workers,
        progress_interval=progress_interval,
    )
    if vcounts["missing"] or vcounts["size_mismatch"]:
        logger.error(
            "purge aborted: %d missing, %d size-mismatched. Examples:",
            vcounts["missing"],
            vcounts["size_mismatch"],
        )
        for status, oid_hex, tid_hex in problems:
            logger.error("  %-13s oid=%s tid=%s", status, oid_hex, tid_hex)
        raise RuntimeError(
            "purge aborted: not every source blob is present in S3"
        )

    counts = {
        "rows_deleted": 0,
        "lobs_unlinked": 0,
        "batches": 0,
    }

    if dry_run:
        # No deletes — we already know from verify how many distinct
        # (zoid, tid) pairs would be affected. Count rows + LOBs.
        import psycopg2

        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM blob_chunk")
                (rows,) = cur.fetchone()
                cur.execute("SELECT count(DISTINCT chunk) FROM blob_chunk")
                (lobs,) = cur.fetchone()
        logger.info(
            "DRY-RUN would delete blob_chunk rows=%d distinct LOBs=%d",
            rows,
            lobs,
        )
        counts["rows_deleted"] = rows
        counts["lobs_unlinked"] = lobs
        return counts

    logger.info("purge: deleting blob_chunk rows + lo_unlink in batches of %d", batch_size)
    import psycopg2

    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = False
        while True:
            with conn.cursor() as cur:
                # Pull a batch off the head of blob_chunk and remove it
                # in one transaction. Postgres doesn't allow LIMIT on a
                # DELETE directly; use a CTE that selects the batch
                # first, then delete by primary-key tuple, returning the
                # LOB oids so we can unlink them in the same txn.
                cur.execute(
                    "WITH del AS ("
                    "  SELECT zoid, tid, chunk_num, chunk "
                    "  FROM blob_chunk "
                    "  LIMIT %s "
                    ") "
                    "DELETE FROM blob_chunk b "
                    "USING del "
                    "WHERE b.zoid = del.zoid "
                    "  AND b.tid = del.tid "
                    "  AND b.chunk_num = del.chunk_num "
                    "RETURNING del.chunk",
                    (batch_size,),
                )
                lob_oids = [row[0] for row in cur.fetchall()]
                if not lob_oids:
                    conn.commit()
                    break
                # Unlink each LOB inside the same transaction. Duplicate
                # OIDs (shouldn't happen in practice but defend against
                # it) would cause errors on the second unlink, so dedupe.
                seen = set()
                unlinked_this_batch = 0
                for oid in lob_oids:
                    if oid in seen:
                        continue
                    seen.add(oid)
                    cur.execute("SELECT lo_unlink(%s)", (oid,))
                    unlinked_this_batch += 1
            conn.commit()
            counts["batches"] += 1
            counts["rows_deleted"] += len(lob_oids)
            counts["lobs_unlinked"] += unlinked_this_batch
            if counts["batches"] % max(1, progress_interval // batch_size) == 0:
                logger.info(
                    "purge progress batches=%d rows_deleted=%d lobs_unlinked=%d",
                    counts["batches"],
                    counts["rows_deleted"],
                    counts["lobs_unlinked"],
                )
    except BaseException:
        with contextlib.suppress(Exception):
            conn.rollback()
        raise
    finally:
        with contextlib.suppress(Exception):
            conn.close()

    logger.info(
        "purge done batches=%d rows_deleted=%d lobs_unlinked=%d",
        counts["batches"],
        counts["rows_deleted"],
        counts["lobs_unlinked"],
    )
    logger.info(
        "purge: run `VACUUM (ANALYZE) blob_chunk;` to update planner stats. "
        "To reclaim disk from pg_largeobject, run `vacuumlo` (contrib) or "
        "`VACUUM FULL pg_largeobject` during a maintenance window — both "
        "lock heavily so they are not done here."
    )
    return counts


def build_purge_argparser():
    p = argparse.ArgumentParser(
        prog="zodb-s3blobs-purge",
        description=(
            "Delete blob bytes from a RelStorage Postgres DB after they "
            "have been migrated to S3. Verifies every blob_chunk pair "
            "exists in S3 first; refuses to run otherwise. Destructive "
            "and irreversible — keep a backup."
        ),
    )

    p.add_argument("--dsn", default=_env("RELSTORAGE_DSN"))
    p.add_argument(
        "--blob-cache-dir",
        required=True,
        help="Local blob-cache directory the source RelStorage uses. "
        "Required only so the RelStorage adapter can be opened; the "
        "purger does not read blob bytes from disk.",
    )
    p.add_argument("--blob-cache-size", default="1GB")
    p.add_argument("--keep-history", action="store_true")

    p.add_argument("--bucket", default=_env("S3_BLOB_BUCKET"))
    p.add_argument("--endpoint-url", default=_env("S3_BLOB_ENDPOINT_URL"))
    p.add_argument("--region", default=_env("S3_BLOB_REGION"))
    p.add_argument("--prefix", default=_env("S3_BLOB_PREFIX", ""))
    p.add_argument("--access-key", default=_env("S3_ACCESS_KEY"))
    p.add_argument("--secret-key", default=_env("S3_SECRET_KEY"))
    p.add_argument("--sse-customer-key", default=_env("S3_BLOB_SSE_CUSTOMER_KEY"))
    p.add_argument("--no-ssl", action="store_true")
    p.add_argument(
        "--addressing-style",
        default="auto",
        choices=("auto", "path", "virtual"),
    )

    p.add_argument("--keepalives", type=int, default=1, choices=(0, 1))
    p.add_argument("--keepalives-idle", type=int, default=30)
    p.add_argument("--keepalives-interval", type=int, default=10)
    p.add_argument("--keepalives-count", type=int, default=3)

    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--verify-size", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pre-flight verification and report what would be "
        "deleted; do not modify the database.",
    )
    p.add_argument("--progress-interval", type=int, default=500)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def purge_main(argv=None):
    args = build_purge_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.bucket:
        print("error: --bucket or S3_BLOB_BUCKET is required", file=sys.stderr)
        return 2
    if not args.dsn:
        print("error: --dsn or RELSTORAGE_DSN is required", file=sys.stderr)
        return 2

    dsn = _augment_dsn_with_keepalives(
        args.dsn,
        keepalives=args.keepalives,
        idle=args.keepalives_idle,
        interval=args.keepalives_interval,
        count=args.keepalives_count,
    )

    s3 = S3Client(
        bucket_name=args.bucket,
        prefix=args.prefix or "",
        endpoint_url=args.endpoint_url,
        region_name=args.region,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
        use_ssl=not args.no_ssl,
        addressing_style=args.addressing_style,
        sse_customer_key=args.sse_customer_key,
    )

    storage = open_relstorage(
        dsn=dsn,
        blob_cache_dir=args.blob_cache_dir,
        blob_cache_size=_parse_size(args.blob_cache_size),
        keep_history=args.keep_history,
    )
    try:
        run_purge(
            storage,
            s3,
            dsn=dsn,
            dry_run=args.dry_run,
            verify_size=args.verify_size,
            workers=args.workers,
            batch_size=args.batch_size,
            progress_interval=args.progress_interval,
        )
    finally:
        storage.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
