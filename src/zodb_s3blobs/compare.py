"""CLI for comparing two RelStorage Postgres databases at the schema level.

Use case: an "original" RelStorage DB (with blobs in ``blob_chunk``) and a
"migrated" DB produced by ``zodbconvert`` whose dest storage is wrapped by
``S3BlobStorage`` (so blobs live in S3 instead). The two DBs should be
logically equivalent — every ``(zoid, tid)`` row in the source's
``object_state`` should appear in the dest's, and every blob in the source's
``blob_chunk`` should have a corresponding S3 key.

This script reports the differences. It does **not** verify blob bytes — only
existence and inventory counts. The aim is to surface anything ``zodbconvert``
does that a hand-rolled migration (e.g. ``zodb_s3blobs.migrate``) might miss,
not to be a content-integrity audit.

Output:
- Section 1: row-count comparison across major RelStorage tables + S3 key count.
- Section 2: ``object_state`` ``(zoid, tid)`` symmetric diff (sampled).
- Section 3: blob ``(zoid, tid)`` set diff against S3 keys.
- Optional ``--json-report``: machine-readable copy of the same data.

Each section prints its own counts and (subject to ``--diff-limit``) example
rows. Exit code is non-zero if any diff is non-empty so it composes with CI.
"""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from zodb_s3blobs.s3client import S3Client
from zodb_s3blobs.storage import _oid_hex
from zodb_s3blobs.storage import _tid_hex

import argparse
import json
import logging
import os
import re
import sys
import ZODB.utils


logger = logging.getLogger("zodb_s3blobs.compare")


_BLOB_KEY_RE = re.compile(r"^blobs/([0-9a-f]+)/([0-9a-f]+)\.blob$")


# ---- DB queries ----------------------------------------------------------


def _open_dsn(dsn):
    """Open a raw psycopg2 connection. We bypass RelStorage entirely here
    because we only need read-only SQL — no blob-cache-dir, no schema
    validation, no keep_history flag.
    """
    import psycopg2

    return psycopg2.connect(dsn)


def _table_exists(cursor, name):
    cursor.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_name = %s AND table_schema = current_schema()",
        (name,),
    )
    return cursor.fetchone() is not None


def collect_table_counts(conn):
    """Return a dict of row counts and aggregate stats for the major
    RelStorage tables present in this DB. Missing tables show as None.
    """
    out = {}
    with conn.cursor() as cur:
        for table in ("object_state", "current_object", "transaction", "blob_chunk"):
            if not _table_exists(cur, table):
                out[table] = None
                continue
            cur.execute(f"SELECT count(*) FROM {table}")
            out[table] = int(cur.fetchone()[0])

        if out.get("object_state") is not None:
            cur.execute("SELECT count(DISTINCT zoid) FROM object_state")
            out["object_state_distinct_zoid"] = int(cur.fetchone()[0])
            cur.execute("SELECT min(tid), max(tid) FROM object_state")
            row = cur.fetchone()
            out["object_state_tid_min"] = int(row[0]) if row[0] is not None else None
            out["object_state_tid_max"] = int(row[1]) if row[1] is not None else None

        if out.get("blob_chunk") is not None:
            cur.execute("SELECT count(DISTINCT (zoid, tid)) FROM blob_chunk")
            out["blob_chunk_distinct_pairs"] = int(cur.fetchone()[0])
    return out


def fetch_object_state_pairs(conn):
    """Return the set of ``(zoid_int, tid_int)`` pairs in ``object_state``.

    For databases over a few million rows this materialises a large set in
    memory; consider streaming or sampling at that point. For the sizes we
    actually run against (low millions) this is fine and dramatically
    simpler than batched diff.
    """
    pairs = set()
    with conn.cursor(name="cmp_object_state") as cur:  # server-side cursor
        cur.itersize = 50000
        cur.execute("SELECT zoid, tid FROM object_state")
        for zoid, tid in cur:
            pairs.add((int(zoid), int(tid)))
    return pairs


def fetch_blob_chunk_pairs(conn):
    pairs = set()
    with conn.cursor(name="cmp_blob_chunk") as cur:
        cur.itersize = 50000
        cur.execute("SELECT DISTINCT zoid, tid FROM blob_chunk")
        for zoid, tid in cur:
            pairs.add((int(zoid), int(tid)))
    return pairs


# ---- S3 inventory --------------------------------------------------------


def fetch_s3_blob_pairs(s3_client):
    """Return ``{(zoid_int, tid_int): key}`` for every parseable key under
    ``blobs/``. Unparseable keys are logged at DEBUG and ignored — the same
    treatment they get from ``S3BlobStorage.pack``.
    """
    out = {}
    bad = 0
    for key in s3_client.list_objects("blobs/"):
        m = _BLOB_KEY_RE.match(key)
        if m is None:
            bad += 1
            logger.debug("compare: skipping unparseable S3 key %s", key)
            continue
        try:
            zoid = int(m.group(1), 16)
            tid = int(m.group(2), 16)
        except (ValueError, OverflowError):
            bad += 1
            continue
        out[(zoid, tid)] = key
    if bad:
        logger.warning("compare: ignored %d unparseable S3 keys", bad)
    return out


# ---- Comparison ----------------------------------------------------------


def diff_pairs(source, dest, *, label_source, label_dest, limit):
    """Return ``{only_in_source: [...], only_in_dest: [...], counts: {...}}``."""
    only_src = source - dest
    only_dst = dest - source
    return {
        "label_source": label_source,
        "label_dest": label_dest,
        "count_source": len(source),
        "count_dest": len(dest),
        "count_only_in_source": len(only_src),
        "count_only_in_dest": len(only_dst),
        "samples_only_in_source": _format_samples(only_src, limit),
        "samples_only_in_dest": _format_samples(only_dst, limit),
    }


def _format_samples(pairs, limit):
    out = []
    for zoid, tid in sorted(pairs)[:limit]:
        out.append(
            {
                "zoid": zoid,
                "tid": tid,
                "oid_hex": _oid_hex(ZODB.utils.p64(zoid)),
                "tid_hex": _tid_hex(ZODB.utils.p64(tid)),
            }
        )
    return out


def _print_diff_section(title, diff):
    print(f"\n--- {title} ---")
    print(
        f"  {diff['label_source']}: {diff['count_source']:>10}   "
        f"{diff['label_dest']}: {diff['count_dest']:>10}"
    )
    print(f"  only in {diff['label_source']}: {diff['count_only_in_source']}")
    if diff["samples_only_in_source"]:
        for s in diff["samples_only_in_source"]:
            print(f"    zoid={s['oid_hex']} tid={s['tid_hex']}")
        if diff["count_only_in_source"] > len(diff["samples_only_in_source"]):
            print(
                f"    ... and {diff['count_only_in_source'] - len(diff['samples_only_in_source'])} more"
            )
    print(f"  only in {diff['label_dest']}: {diff['count_only_in_dest']}")
    if diff["samples_only_in_dest"]:
        for s in diff["samples_only_in_dest"]:
            print(f"    zoid={s['oid_hex']} tid={s['tid_hex']}")
        if diff["count_only_in_dest"] > len(diff["samples_only_in_dest"]):
            print(
                f"    ... and {diff['count_only_in_dest'] - len(diff['samples_only_in_dest'])} more"
            )


def _print_counts_section(source_counts, dest_counts, s3_key_count):
    print("\n--- Inventory ---")
    print(f"  {'metric':<35} {'source':>14} {'dest':>14}")
    keys = sorted(set(source_counts) | set(dest_counts))
    for k in keys:
        s = source_counts.get(k)
        d = dest_counts.get(k)
        sd = "—" if s is None else f"{s:>14}"
        dd = "—" if d is None else f"{d:>14}"
        print(f"  {k:<35} {sd} {dd}")
    print(f"  {'s3_blob_keys':<35} {'—':>14} {s3_key_count:>14}")


# ---- Main ----------------------------------------------------------------


def _env(name, default=None):
    return os.environ.get(name, default)


def build_argparser():
    p = argparse.ArgumentParser(
        prog="zodb-s3blobs-compare",
        description=(
            "Compare an original RelStorage DB against a zodbconvert-produced "
            "DB whose blobs live in S3. Reports inventory and (zoid, tid) "
            "diffs to surface migration gaps."
        ),
    )
    p.add_argument(
        "--source-dsn", required=True, help="Postgres DSN for the original RelStorage DB."
    )
    p.add_argument(
        "--dest-dsn",
        default=None,
        help="Postgres DSN for the migrated RelStorage DB. If omitted, "
        "object_state comparison is skipped and only the blob → S3 check runs.",
    )

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

    p.add_argument(
        "--diff-limit",
        type=int,
        default=20,
        help="Max number of example rows shown per diff section (default 20).",
    )
    p.add_argument(
        "--json-report",
        default=None,
        help="If given, write the full report (incl. all diff samples up to "
        "--diff-limit) to this file as JSON.",
    )
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

    report = {}

    logger.info("opening source DB")
    source_conn = _open_dsn(args.source_dsn)
    try:
        logger.info("collecting source counts")
        source_counts = collect_table_counts(source_conn)
        logger.info("loading source blob_chunk pairs")
        source_blob_pairs = fetch_blob_chunk_pairs(source_conn)
        if args.dest_dsn:
            logger.info("loading source object_state pairs")
            source_obj_pairs = fetch_object_state_pairs(source_conn)
        else:
            source_obj_pairs = None
    finally:
        source_conn.close()

    if args.dest_dsn:
        logger.info("opening dest DB")
        dest_conn = _open_dsn(args.dest_dsn)
        try:
            logger.info("collecting dest counts")
            dest_counts = collect_table_counts(dest_conn)
            logger.info("loading dest object_state pairs")
            dest_obj_pairs = fetch_object_state_pairs(dest_conn)
        finally:
            dest_conn.close()
    else:
        dest_counts = {}
        dest_obj_pairs = None

    logger.info("listing S3 blob keys")
    s3_pairs = fetch_s3_blob_pairs(s3)

    # --- Section 1: counts ---
    _print_counts_section(source_counts, dest_counts, len(s3_pairs))
    report["source_counts"] = source_counts
    report["dest_counts"] = dest_counts
    report["s3_blob_keys"] = len(s3_pairs)

    nonzero_diffs = 0

    # --- Section 2: blob_chunk vs S3 keys ---
    blob_diff = diff_pairs(
        source_blob_pairs,
        set(s3_pairs.keys()),
        label_source="source.blob_chunk",
        label_dest="dest.s3_keys",
        limit=args.diff_limit,
    )
    _print_diff_section("Blob (zoid, tid) presence: source blob_chunk ↔ S3", blob_diff)
    report["blob_diff"] = blob_diff
    nonzero_diffs += blob_diff["count_only_in_source"]
    nonzero_diffs += blob_diff["count_only_in_dest"]

    # --- Section 3: object_state vs object_state (if dest provided) ---
    if args.dest_dsn:
        obj_diff = diff_pairs(
            source_obj_pairs,
            dest_obj_pairs,
            label_source="source.object_state",
            label_dest="dest.object_state",
            limit=args.diff_limit,
        )
        _print_diff_section(
            "Object (zoid, tid) presence: source object_state ↔ dest object_state",
            obj_diff,
        )
        report["object_state_diff"] = obj_diff
        nonzero_diffs += obj_diff["count_only_in_source"]
        nonzero_diffs += obj_diff["count_only_in_dest"]

    if args.json_report:
        with open(args.json_report, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        logger.info("wrote JSON report to %s", args.json_report)

    print()
    if nonzero_diffs:
        print(f"FAIL: {nonzero_diffs} differing rows across all checks")
        return 1
    print("OK: no differences detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
