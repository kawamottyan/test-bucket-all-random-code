import pickle
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union

import boto3
import botocore
import polars as pl
import torch

from src.common.utils.general import setup_logger

logger = setup_logger(__name__)


class FileFormat:
    PARQUET = "parquet"
    CSV = "csv"
    PICKLE = "pickle"
    PT = "pt"


class S3Handler:
    def __init__(
        self,
        aws_region: str,
        bucket_name: str,
        endpoint_url=None,
        verify_ssl=False,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        client_kwargs: Dict[str, Union[str, bool, None, boto3.session.Config]] = {
            "region_name": aws_region,
        }

        if endpoint_url:
            client_kwargs.update(
                {
                    "endpoint_url": endpoint_url,
                    "verify": verify_ssl,
                    "aws_access_key_id": aws_access_key_id,
                    "aws_secret_access_key": aws_secret_access_key,
                }
            )
            client_kwargs["config"] = boto3.session.Config(signature_version="s3v4")

        self.s3_client = boto3.client("s3", **client_kwargs)
        self.bucket_name = bucket_name

    def _bucket_exists(self) -> bool:
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except botocore.exceptions.ClientError as e:
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                return False
            raise

    def _verify_bucket(self) -> None:
        try:
            if not self._bucket_exists():
                logger.error("Bucket does not exist: %s", self.bucket_name)
                raise ValueError(f"Bucket {self.bucket_name} does not exist")
            else:
                logger.info("Bucket exists: %s", self.bucket_name)

        except botocore.exceptions.ClientError as e:
            logger.error("Failed to verify bucket %s: %s", self.bucket_name, e)
            raise

    def put_parquet_object(self, df: pl.DataFrame, s3_key: str) -> None:
        parquet_buffer = BytesIO()
        df.write_parquet(parquet_buffer)
        parquet_buffer.seek(0)

        logger.info("Uploading Parquet data to s3://%s/%s", self.bucket_name, s3_key)
        self.s3_client.put_object(
            Bucket=self.bucket_name, Key=s3_key, Body=parquet_buffer
        )
        logger.info("Uploaded Parquet data to s3://%s/%s", self.bucket_name, s3_key)

    def _get_object(self, s3_key: str):
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return obj
        except self.s3_client.exceptions.NoSuchKey:
            logger.info("No object found at s3://%s/%s", self.bucket_name, s3_key)
            return None

    def _get_file_format(self, s3_key: str) -> str:
        extension = Path(s3_key).suffix.lower()
        format_map = {
            ".parquet": FileFormat.PARQUET,
            ".csv": FileFormat.CSV,
            ".pickle": FileFormat.PICKLE,
            ".pkl": FileFormat.PICKLE,
            ".pt": FileFormat.PT,
        }
        if extension in format_map:
            return format_map[extension]
        raise ValueError(f"Unsupported file extension: {extension}")

    def get_dataframe(
        self,
        s3_key: str,
        columns: Optional[list[str]] = None,
    ) -> Optional[pl.DataFrame]:
        try:
            logger.info("Fetching dataframe from s3://%s/%s", self.bucket_name, s3_key)
            obj = self._get_object(s3_key)
            if not obj:
                return None

            file_format = self._get_file_format(s3_key)

            with BytesIO(obj["Body"].read()) as file_body:
                if file_format == FileFormat.PARQUET:
                    df = pl.read_parquet(file_body)
                elif file_format == FileFormat.CSV:
                    df = pl.read_csv(file_body)
                else:
                    logger.error(
                        "Unsupported file format for dataframe: %s", file_format
                    )
                    return None

                if columns:
                    df = df.select(columns)
                return df

        except Exception as e:
            logger.error("Error retrieving dataframe from S3: %s", e)
            return None

    def get_pickle_object(
        self,
        s3_key: str,
    ) -> Optional[Any]:
        try:
            logger.info(
                "Fetching pickle object from s3://%s/%s", self.bucket_name, s3_key
            )
            obj = self._get_object(s3_key)
            if not obj:
                return None

            with BytesIO(obj["Body"].read()) as file_body:
                return pickle.load(file_body)

        except Exception as e:
            logger.error("Error retrieving pickle object from S3: %s", e)
            return None

    def get_torch_model(self, s3_key: str) -> Optional[torch.jit.ScriptModule]:
        try:
            logger.info(
                "Fetching torch model from s3://%s/%s", self.bucket_name, s3_key
            )
            obj = self._get_object(s3_key)
            if not obj:
                return None

            with BytesIO(obj["Body"].read()) as file_body:
                return torch.jit.load(file_body)

        except Exception as e:
            logger.error("Error retrieving torch model from S3: %s", e)
            return None

    def get_all_activity_logs(
        self,
        log_name: str,
        columns: Optional[list[str]] = None,
    ) -> Optional[pl.DataFrame]:
        try:
            prefix = f"log_storage/{log_name.lower()}/"
            paginator = self.s3_client.get_paginator("list_objects_v2")
            dfs = []

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        if obj["Key"].endswith(".parquet"):
                            df = self.get_dataframe(obj["Key"], columns)
                            if df is not None and not df.is_empty():
                                dfs.append(df)

            return pl.concat(dfs) if dfs else None

        except Exception as e:
            logger.error("Error merging activity logs: %s", e)
            return None

    def upload_parquet(self, df: pl.DataFrame, s3_key: str) -> None:
        parquet_buffer = BytesIO()
        df.write_parquet(parquet_buffer)
        parquet_buffer.seek(0)

        logger.info("Uploading Parquet data to s3://%s/%s", self.bucket_name, s3_key)
        self.s3_client.put_object(
            Bucket=self.bucket_name, Key=s3_key, Body=parquet_buffer
        )
        logger.info("Uploaded Parquet data to s3://%s/%s", self.bucket_name, s3_key)
