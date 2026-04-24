"""CLI for migrating existing RelStorage blobs into S3.

The typical `plone6-nsw-dds`-shaped deployment runs RelStorage with
``blob-cache-size`` set and no ``shared-blob-dir true``, which stores blob
bytes in the Postgres ``blob_chunk`` table and uses the local blob-dir only
as an on-disk LRU cache. To migrate such a deployment to ``S3BlobStorage``,
every ``(zoid, tid)`` pair in ``blob_chunk`` must be copied into S3 under
the key scheme ``blobs/{_oid_hex(oid)}/{_tid_hex(tid)}.blob`` that
``S3BlobStorage.loadBlob`` reads. This module does exactly that.

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
    """Yield ``(oid_bytes, tid_bytes)`` for every row in ``blob_chunk``.

    Uses the storage's adapter cursor so we inherit its DSN, isolation, and
    driver choice. The query works on both history-free and history-preserving
    schemas (both carry a ``tid`` column; see relstorage/adapters/schema.py).
    """
    conn, cursor = storage._adapter.connmanager.open_for_load()
    try:
        cursor.execute("SELECT DISTINCT zoid, tid FROM blob_chunk")
        rows = cursor.fetchall()
    finally:
        storage._adapter.connmanager.close(conn, cursor)
    for zoid, tid in rows:
        yield ZODB.utils.p64(int(zoid)), ZODB.utils.p64(int(tid))


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
):
    """Migrate one (oid, tid) blob into S3. Returns (status, size_bytes).

    status ∈ {"uploaded", "skipped", "would_upload",
              "size_mismatch_reuploaded", "stale"}.

    Any other exception bubbles up so the caller can record a per-task failure.
    """
    key = s3_key_for(oid, tid)
    oid_hex = _oid_hex(oid)
    tid_hex = _tid_hex(tid)

    size_mismatch = False
    if not overwrite:
        meta = s3_client.head_object(key)
        if meta is not None:
            remote_size = int(meta.get("ContentLength", -1))
            if verify_size:
                try:
                    local_size = blob_size_in_db(storage, oid, tid)
                except LookupError:
                    logger.warning(
                        "stale blob_chunk row oid=%s tid=%s — keeping existing S3 key",
                        oid_hex,
                        tid_hex,
                    )
                    return ("stale", remote_size if remote_size >= 0 else 0)
                if remote_size != local_size:
                    logger.warning(
                        "size mismatch oid=%s tid=%s remote=%d db=%d — re-uploading",
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
            size = blob_size_in_db(storage, oid, tid) if verify_size else 0
        except LookupError:
            return ("stale", 0)
        logger.debug(
            "would upload oid=%s tid=%s key=%s size=%d", oid_hex, tid_hex, key, size
        )
        return ("would_upload", size)

    try:
        local_path, size = _materialize_blob(storage, oid, tid, tmp_dir)
    except LookupError as exc:
        logger.warning("skipping vanished blob: %s", exc)
        return ("stale", 0)

    try:
        s3_client.upload_file(local_path, key)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(local_path)

    status = "size_mismatch_reuploaded" if size_mismatch else "uploaded"
    logger.debug("%s oid=%s tid=%s key=%s size=%d", status, oid_hex, tid_hex, key, size)
    return (status, size)


def run_migration(
    storage,
    s3_client,
    *,
    workers=8,
    overwrite=False,
    dry_run=False,
    verify_size=False,
    progress_interval=500,
    tmp_dir=None,
):
    """Walk every (oid, tid) blob pair and migrate each into S3.

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

    # Each materialisation writes a throwaway temp file; using a dedicated
    # directory makes it easy to pre-size the disk and clean up after a
    # crash (``rm -rf``). Defaults to the system temp dir.
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="zs3b-migrate-")
        _owns_tmp = True
    else:
        os.makedirs(tmp_dir, exist_ok=True)
        _owns_tmp = False

    def _submit(ex, oid, tid):
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
        )

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            # Feed lazily: walk the blob_chunk rows, keep at most
            # ``workers * 4`` in flight so memory / temp-file disk use stays
            # bounded on large dumps.
            in_flight = set()
            max_in_flight = max(workers * 4, workers + 1)

            for oid, tid in iter_blob_pairs(storage):
                counts["scanned"] += 1
                while len(in_flight) >= max_in_flight:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        _record(fut, counts, failures)
                    if counts["scanned"] % progress_interval == 0:
                        _log_progress(counts)
                in_flight.add(_submit(ex, oid, tid))

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
        required=True,
        help="Local blob-cache directory the source RelStorage uses. "
        "Used transiently to materialise blobs before upload; --evict-after-upload "
        "keeps this dir small.",
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

    if not args.dsn:
        print("error: --dsn or RELSTORAGE_DSN is required", file=sys.stderr)
        return 2

    storage = open_relstorage(
        dsn=args.dsn,
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


if __name__ == "__main__":
    sys.exit(main())
