import contextlib
import logging
import os
import re
import shutil
import tempfile
import ZODB.blob
import ZODB.interfaces
import ZODB.POSException
import ZODB.utils
import zope.interface


logger = logging.getLogger(__name__)

_relstorage_lock_early_applied = False


def _ensure_relstorage_lock_early():
    """Force RelStorage to allocate TID during tpc_vote (LOCK_EARLY).

    RelStorage 3.x defers TID allocation to tpc_finish for shorter lock
    hold times. S3BlobStorage needs the TID in tpc_vote to construct the
    S3 key before uploading. Without LOCK_EARLY, the TID is not available.

    This monkey-patches the module-level LOCK_EARLY flag in
    relstorage.storage.tpc.vote. The flag is read by AbstractVote._vote()
    to decide whether to call _lock_and_move(vote_only=True).

    Safe to call multiple times (idempotent).
    """
    global _relstorage_lock_early_applied
    if _relstorage_lock_early_applied:
        return
    try:
        import relstorage.storage.tpc.vote as vote_mod
    except ImportError:
        return
    if not vote_mod.LOCK_EARLY:
        vote_mod.LOCK_EARLY = True
        logger.info(
            "Forced RELSTORAGE_LOCK_EARLY=True for S3BlobStorage "
            "TID availability during tpc_vote"
        )
    _relstorage_lock_early_applied = True


_BLOB_KEY_RE = re.compile(r"^blobs/([0-9a-f]+)/[0-9a-f]+\.blob$")

# A blob file holding a marker stands for an object already uploaded to S3,
# e.g. by a TUS upload handler, which is copied to the blob's key at commit:
#
#     S3BLOB-STAGED\n<staging_key>\n<size>\n
#
# The marker is recognised by content because savepoints move blob files.
# Only keys under STAGING_PREFIX are accepted: commit deletes the staging
# key, and anyone able to store a blob can store a file that looks like a
# marker.
STAGING_PREFIX = "tus-staging/"
_STAGED_MAGIC = b"S3BLOB-STAGED\n"
_STAGED_MARKER_MAX_BYTES = 4096  # markers are ~70 bytes; don't read real blobs


def write_staged_marker(path, staging_key, size):
    """Write a marker to ``path`` for an object uploaded to ``staging_key``."""
    if not staging_key.startswith(STAGING_PREFIX):
        raise ValueError(f"staging key must start with {STAGING_PREFIX!r}")
    with open(path, "wb") as f:
        f.write(_STAGED_MAGIC + f"{staging_key}\n{size}\n".encode())


def _parse_staged_marker(path):
    """Return the staging key if ``path`` holds a valid marker, else None."""
    try:
        if os.path.getsize(path) > _STAGED_MARKER_MAX_BYTES:
            return None
        with open(path, "rb") as f:
            head = f.read(_STAGED_MARKER_MAX_BYTES)
    except OSError:
        return None
    if not head.startswith(_STAGED_MAGIC):
        return None
    try:
        _magic, staging_key, size, _rest = head.split(b"\n", 3)
        staging_key = staging_key.decode("utf-8")
        int(size)
    except ValueError:  # includes UnicodeDecodeError
        return None
    if not staging_key.startswith(STAGING_PREFIX):
        logger.warning(
            "Storing blob %s as data: it looks like a staged-upload marker "
            "but its key %r is not under %r",
            path,
            staging_key,
            STAGING_PREFIX,
        )
        return None
    return staging_key


@zope.interface.implementer(ZODB.interfaces.IBlobStorage)
class S3BlobStorage:
    """ZODB storage wrapper that redirects blob operations to S3.

    Wraps any base storage via __getattr__ proxy pattern.
    All blob methods are explicitly defined to shadow the base
    storage's methods (if any).
    """

    def __init__(self, base_storage, s3_client, cache, temp_dir=None):
        self.__storage = base_storage
        self._s3_client = s3_client
        self._cache = cache
        self._pending_blobs = {}  # {oid: staged_path}
        self._pending_blobs_s3 = {}  # {oid: staging_key}
        self._uploaded_keys = []  # [(oid, tid, s3_key)]
        self._pending_staging_keys = []  # deleted after commit or abort
        self._temp_dir = temp_dir or tempfile.mkdtemp()
        os.makedirs(self._temp_dir, exist_ok=True, mode=0o700)

        # Force LOCK_EARLY if RelStorage is involved so TID is
        # available during tpc_vote for S3 key construction.
        if hasattr(base_storage, "_tpc_phase"):
            _ensure_relstorage_lock_early()

    def __getattr__(self, name):
        return getattr(self.__storage, name)

    def __len__(self):
        return len(self.__storage)

    def __repr__(self):
        return f"<S3BlobStorage proxy for {self.__storage!r}>"

    # -- Blob methods (explicitly defined, shadow base storage) --

    def storeBlob(self, oid, oldserial, data, blobfilename, version, transaction):
        # Store object data (pickle) in base storage
        self.__storage.store(oid, oldserial, data, "", transaction)
        staging_key = _parse_staged_marker(blobfilename)
        if staging_key is not None:
            self._pending_blobs_s3[oid] = staging_key
            with contextlib.suppress(OSError):
                os.remove(blobfilename)
            return
        # Stage blob locally
        oid_hex = _oid_hex(oid)
        staged_path = os.path.join(self._temp_dir, f"{oid_hex}.blob")
        shutil.move(blobfilename, staged_path)
        self._pending_blobs[oid] = staged_path

    def loadBlob(self, oid, serial):
        # Check pending blobs first (stored in current txn, not yet in S3)
        pending = self._pending_blobs.get(oid)
        if pending is not None and os.path.exists(pending):
            return pending

        # Check cache
        cached = self._cache.get(oid, serial)
        if cached is not None:
            return cached

        # Download from S3
        key = self._s3_key(oid, serial)
        meta = self._s3_client.head_object(key)
        if meta is None:
            raise ZODB.POSException.POSKeyError(oid, serial)

        # Download to temp, put in cache
        tmp_download = os.path.join(self._temp_dir, f"dl_{_oid_hex(oid)}.tmp")
        self._s3_client.download_file(key, tmp_download)
        path = self._cache.put(oid, serial, tmp_download)
        # Clean up temp download
        with contextlib.suppress(OSError):
            os.remove(tmp_download)
        return path

    def openCommittedBlobFile(self, oid, serial, blob=None):
        filename = self.loadBlob(oid, serial)
        if blob is None:
            return open(filename, "rb")
        return ZODB.blob.BlobFile(filename, "r", blob)

    def temporaryDirectory(self):
        return self._temp_dir

    # -- 2PC hooks --

    def tpc_vote(self, transaction):
        self.__storage.tpc_vote(transaction)
        if self._pending_blobs or self._pending_blobs_s3:
            tid = self._extract_base_tid()
            for oid, staged_path in self._pending_blobs.items():
                key = self._s3_key(oid, tid)
                self._s3_client.upload_file(staged_path, key)
                self._uploaded_keys.append((oid, tid, key))
            for oid, staging_key in self._pending_blobs_s3.items():
                final_key = self._s3_key(oid, tid)
                self._s3_client.copy_object(staging_key, final_key)
                self._uploaded_keys.append((oid, tid, final_key))
                self._pending_staging_keys.append(staging_key)

    def tpc_finish(self, transaction, func=lambda tid: None):
        tid = self.__storage.tpc_finish(transaction, func)
        # Move staged files into cache (must not fail)
        for oid, staged_path in self._pending_blobs.items():
            try:
                self._cache.put(oid, tid, staged_path)
            except Exception:
                logger.warning(
                    "Failed to cache blob for oid=%s tid=%s",
                    _oid_hex(oid),
                    _tid_hex(tid),
                    exc_info=True,
                )
            # Clean staged file
            with contextlib.suppress(OSError):
                os.remove(staged_path)
        # Blobs copied from a staging key aren't cached until first loaded.
        self._delete_staging_keys()
        self._pending_blobs = {}
        self._pending_blobs_s3 = {}
        self._uploaded_keys = []
        return tid

    def tpc_abort(self, transaction):
        self.__storage.tpc_abort(transaction)
        # Delete uploaded S3 keys (best-effort)
        for _oid, _tid, key in self._uploaded_keys:
            try:
                self._s3_client.delete_object(key)
            except Exception:
                logger.warning(
                    "Failed to delete S3 key %s during abort", key, exc_info=True
                )
        self._delete_staging_keys()
        # Clean staged files
        for _oid, staged_path in self._pending_blobs.items():
            with contextlib.suppress(OSError):
                os.remove(staged_path)
        self._pending_blobs = {}
        self._pending_blobs_s3 = {}
        self._uploaded_keys = []

    def _delete_staging_keys(self):
        """Delete the staging keys copied in tpc_vote (best-effort)."""
        for staging_key in self._pending_staging_keys:
            try:
                self._s3_client.delete_object(staging_key)
            except Exception:
                logger.warning(
                    "Failed to delete S3 staging key %s", staging_key, exc_info=True
                )
        self._pending_staging_keys = []

    # -- MVCC --

    def new_instance(self):
        new_instance = getattr(self.__storage, "new_instance", None)
        base = new_instance() if new_instance is not None else self.__storage
        # Each MVCC instance gets its own temp dir to avoid file name collisions
        instance_temp = tempfile.mkdtemp(dir=self._temp_dir)
        return S3BlobStorage(base, self._s3_client, self._cache, instance_temp)

    def close(self):
        self.__storage.close()
        close_cache = getattr(self._cache, "close", None)
        if close_cache is not None:
            close_cache()
        with contextlib.suppress(OSError):
            shutil.rmtree(self._temp_dir)

    # -- Pack / GC --

    def pack(self, pack_time, referencesf):
        # Pack the base storage first
        self.__storage.pack(pack_time, referencesf)
        # GC: remove S3 keys for unreachable OIDs
        for key in self._s3_client.list_objects("blobs/"):
            oid = self._oid_from_key(key)
            if oid is None:
                continue
            try:
                self.__storage.load(oid)
            except ZODB.POSException.POSKeyError:
                logger.info("GC: removing orphaned S3 key %s", key)
                try:
                    self._s3_client.delete_object(key)
                except Exception:
                    logger.warning("GC: failed to delete S3 key %s", key, exc_info=True)

    @staticmethod
    def _oid_from_key(key):
        """Extract oid bytes from S3 key like 'blobs/{oid_hex}/{tid_hex}.blob'."""
        m = _BLOB_KEY_RE.match(key)
        if m is None:
            return None
        try:
            return ZODB.utils.p64(int(m.group(1), 16))
        except (ValueError, OverflowError):
            return None

    # -- Helpers --

    def _extract_base_tid(self):
        """Extract the current transaction TID from the base storage.

        Supports two storage backends:
        - BaseStorage subclasses (MappingStorage, FileStorage): ``_tid`` attribute
        - RelStorage: ``_tpc_phase.committing_tid_lock.tid``
          (requires LOCK_EARLY so TID is allocated during tpc_vote)
        """
        # BaseStorage path (MappingStorage, FileStorage)
        tid = getattr(self.__storage, "_tid", None)
        if tid is not None:
            return tid
        # RelStorage path: TID lives in the TPC phase's lock object
        phase = getattr(self.__storage, "_tpc_phase", None)
        if phase is not None:
            lock = getattr(phase, "committing_tid_lock", None)
            if lock is not None:
                tid = getattr(lock, "tid", None)
                if tid is not None:
                    return tid
        raise RuntimeError(
            "Cannot determine TID after tpc_vote. "
            "The base storage must expose _tid (BaseStorage) "
            "or _tpc_phase.committing_tid_lock.tid (RelStorage with LOCK_EARLY)."
        )

    def _s3_key(self, oid, tid):
        return f"blobs/{_oid_hex(oid)}/{_tid_hex(tid)}.blob"


def _oid_hex(oid):
    """Convert oid bytes to hex string."""
    return ZODB.utils.oid_repr(oid).removeprefix("0x").lstrip("0") or "0"


def _tid_hex(tid):
    """Convert tid bytes to hex string."""
    return ZODB.utils.tid_repr(tid).removeprefix("0x").lstrip("0") or "0"
