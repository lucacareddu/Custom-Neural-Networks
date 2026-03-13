import os

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_DEFAULT_REGION"] = "eu-north-1"

import boto3
import json

import uuid

payload = json.dumps(json.load(open('stream_test'))).encode("utf-8")

boto3.client('kinesis').put_record(
                StreamName="input_stream",
                Data=payload,
                PartitionKey=str(uuid.uuid4())
            )
