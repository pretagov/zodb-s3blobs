"""S3Client multipart upload, copy and lifecycle helpers."""

from botocore.exceptions import ClientError
from moto import mock_aws
from urllib.parse import parse_qs
from urllib.parse import urlparse
from zodb_s3blobs.s3client import S3Client
from zodb_s3blobs.s3client import S3OperationError

import boto3
import pytest


PART = b"x" * (5 * 1024 * 1024)  # S3's minimum size for all but the last part


@pytest.fixture
def s3_client():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        yield S3Client(bucket_name="test-bucket", region_name="us-east-1", prefix="pfx")


def _read(s3_client, key, tmp_path):
    path = str(tmp_path / "download")
    s3_client.download_file(key, path)
    with open(path, "rb") as f:
        return f.read()


def _client_error(code="AccessDenied"):
    return ClientError({"Error": {"Code": code, "Message": ""}}, "op")


class TestMultipartUpload:
    def test_upload_list_and_complete(self, s3_client, tmp_path):
        upload_id = s3_client.create_multipart_upload("tus-staging/a")
        s3_client.upload_part("tus-staging/a", upload_id, 2, b"tail")
        s3_client.upload_part("tus-staging/a", upload_id, 1, PART)

        parts = s3_client.list_parts("tus-staging/a", upload_id)
        assert [(p["PartNumber"], p["Size"]) for p in parts] == [
            (1, len(PART)),
            (2, 4),
        ]

        s3_client.complete_multipart_upload("tus-staging/a", upload_id, parts)
        assert _read(s3_client, "tus-staging/a", tmp_path) == PART + b"tail"

    def test_abort(self, s3_client):
        upload_id = s3_client.create_multipart_upload("tus-staging/a")
        s3_client.upload_part("tus-staging/a", upload_id, 1, b"data")
        s3_client.abort_multipart_upload("tus-staging/a", upload_id)

        with pytest.raises(S3OperationError):
            s3_client.list_parts("tus-staging/a", upload_id)

    @pytest.mark.parametrize(
        "method, args",
        [
            ("create_multipart_upload", ()),
            ("upload_part", ("id", 1, b"")),
            ("complete_multipart_upload", ("id", [])),
            ("abort_multipart_upload", ("id",)),
            ("list_parts", ("id",)),
        ],
    )
    def test_errors_are_wrapped(self, s3_client, method, args):
        def fail(**kwargs):
            raise _client_error()

        setattr(s3_client._client, method, fail)
        with pytest.raises(S3OperationError):
            getattr(s3_client, method)("tus-staging/a", *args)


class TestListPartsPagination:
    def _pages(self, s3_client, pages):
        calls = []

        def list_parts(**kwargs):
            calls.append(kwargs["PartNumberMarker"])
            return pages[len(calls) - 1]

        s3_client._client.list_parts = list_parts
        return calls

    def _part(self, n):
        return {"PartNumber": n, "ETag": f'"{n}"', "Size": 1}

    def test_uses_next_part_number_marker(self, s3_client):
        calls = self._pages(
            s3_client,
            [
                {
                    "Parts": [self._part(1)],
                    "IsTruncated": True,
                    "NextPartNumberMarker": 1,
                },
                {"Parts": [self._part(2)], "IsTruncated": False},
            ],
        )
        parts = s3_client.list_parts("tus-staging/a", "id")
        assert [p["PartNumber"] for p in parts] == [1, 2]
        assert calls == [0, 1]

    def test_falls_back_to_last_part_without_marker(self, s3_client):
        """Tigris can omit NextPartNumberMarker."""
        calls = self._pages(
            s3_client,
            [
                {"Parts": [self._part(1), self._part(2)], "IsTruncated": True},
                {"Parts": [self._part(3)], "IsTruncated": False},
            ],
        )
        parts = s3_client.list_parts("tus-staging/a", "id")
        assert [p["PartNumber"] for p in parts] == [1, 2, 3]
        assert calls == [0, 2]

    def test_stops_without_marker_or_parts(self, s3_client):
        calls = self._pages(s3_client, [{"Parts": [], "IsTruncated": True}])
        assert s3_client.list_parts("tus-staging/a", "id") == []
        assert calls == [0]


class TestPresignedUploadPart:
    def test_url(self, s3_client):
        url = urlparse(
            s3_client.generate_upload_part_presigned_url("tus-staging/a", "id", 3)
        )
        query = parse_qs(url.query)
        assert url.path.endswith("/pfx/tus-staging/a")
        assert query["uploadId"] == ["id"]
        assert query["partNumber"] == ["3"]

    def test_error_is_wrapped(self, s3_client):
        def fail(*args, **kwargs):
            raise _client_error()

        s3_client._client.generate_presigned_url = fail
        with pytest.raises(S3OperationError):
            s3_client.generate_upload_part_presigned_url("tus-staging/a", "id", 1)


class TestCopyObject:
    def test_copy(self, s3_client, tmp_path):
        source = tmp_path / "source"
        source.write_bytes(b"data")
        s3_client.upload_file(str(source), "tus-staging/a")

        s3_client.copy_object("tus-staging/a", "blobs/1/2.blob")

        assert _read(s3_client, "blobs/1/2.blob", tmp_path) == b"data"

    def test_missing_source(self, s3_client):
        with pytest.raises(S3OperationError):
            s3_client.copy_object("tus-staging/missing", "blobs/1/2.blob")


class TestLifecycleRule:
    def _rules(self, s3_client):
        return s3_client._client.get_bucket_lifecycle_configuration(
            Bucket="test-bucket"
        )["Rules"]

    def test_adds_rule_under_client_prefix(self, s3_client):
        assert s3_client.ensure_abort_multipart_lifecycle_rule("tus", "tus-staging/")

        (rule,) = self._rules(s3_client)
        assert rule["ID"] == "tus"
        assert rule["Filter"]["Prefix"] == "pfx/tus-staging/"
        assert rule["AbortIncompleteMultipartUpload"]["DaysAfterInitiation"] == 7

    def test_keeps_other_rules_and_is_idempotent(self, s3_client):
        s3_client._client.put_bucket_lifecycle_configuration(
            Bucket="test-bucket",
            LifecycleConfiguration={
                "Rules": [
                    {
                        "ID": "other",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "logs/"},
                        "Expiration": {"Days": 30},
                    }
                ]
            },
        )
        assert s3_client.ensure_abort_multipart_lifecycle_rule("tus", "tus-staging/")
        assert s3_client.ensure_abort_multipart_lifecycle_rule("tus", "tus-staging/")

        assert sorted(r["ID"] for r in self._rules(s3_client)) == ["other", "tus"]

    @pytest.mark.parametrize("prefix", ["", "blob", "blobs/", "blobs/1/"])
    def test_refuses_prefix_covering_blobs(self, s3_client, prefix):
        assert not s3_client.ensure_abort_multipart_lifecycle_rule("tus", prefix)

    def test_false_if_config_unreadable(self, s3_client):
        def fail(**kwargs):
            raise _client_error()

        s3_client._client.get_bucket_lifecycle_configuration = fail
        assert not s3_client.ensure_abort_multipart_lifecycle_rule(
            "tus", "tus-staging/"
        )

    def test_false_if_rule_cannot_be_written(self, s3_client):
        def fail(**kwargs):
            raise _client_error()

        s3_client._client.put_bucket_lifecycle_configuration = fail
        assert not s3_client.ensure_abort_multipart_lifecycle_rule(
            "tus", "tus-staging/"
        )
