from zope.interface import Interface


class IS3Client(Interface):
    """Abstraction over S3-compatible object storage."""

    def upload_file(local_path, s3_key):
        """Upload a local file to S3."""

    def download_file(s3_key, local_path):
        """Download an S3 object to a local file (atomic via temp+rename)."""

    def delete_object(s3_key):
        """Delete an S3 object."""

    def head_object(s3_key):
        """Return metadata dict for an S3 object, or None if not found."""

    def list_objects(prefix):
        """Yield S3 keys matching the given prefix."""

    def create_multipart_upload(s3_key):
        """Initiate a multipart upload; return the UploadId."""

    def upload_part(s3_key, upload_id, part_number, body):
        """Upload a single part; return the ETag."""

    def complete_multipart_upload(s3_key, upload_id, parts):
        """Complete a multipart upload. parts is a list of {PartNumber, ETag}."""

    def abort_multipart_upload(s3_key, upload_id):
        """Abort a multipart upload and discard any uploaded parts."""

    def copy_object(src_key, dst_key):
        """Server-side copy of one S3 object to another key in the same bucket."""


class IS3BlobCache(Interface):
    """Local filesystem LRU cache for S3 blobs."""

    def get(oid, tid):
        """Return cached file path or None."""

    def put(oid, tid, source_path):
        """Copy source file into cache."""

    def notify_loaded(byte_count):
        """Track loaded bytes and trigger cleanup if threshold exceeded."""


class IS3BlobStorage(Interface):
    """Marker for S3-backed blob storage."""
