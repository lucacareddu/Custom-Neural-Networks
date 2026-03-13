import lambda_function
import base64
import json


def mock_event(num_observations=-1):
    data = json.dumps(json.load(open('stream_test'))[:num_observations]).encode("utf-8")
    encoded_data = base64.b64encode(data).decode("utf-8")

    event = {
        "Records": [
            {
                "kinesis": {
                    "kinesisSchemaVersion": "1.0",
                    "partitionKey": "1",
                    "sequenceNumber": "49630081666084879290581185630324770398608704880802529282",
                    "data": encoded_data,
                    "approximateArrivalTimestamp": 1654161514.132
                },
                "eventSource": "aws:kinesis",
                "eventVersion": "1.0",
                "eventID": "shardId-000000000000:49630081666084879290581185630324770398608704880802529282",
                "eventName": "aws:kinesis:record",
                "invokeIdentityArn": "arn:aws:iam::387546586013:role/lambda-kinesis-role",
                "awsRegion": "eu-north-1",
                "eventSourceARN": "arn:aws:kinesis:eu-west-1:387546586013:stream/ride_events"
            }
        ]
    }

    return event


#------------------------------------- TESTS

def test_full_input():
    event = mock_event()
    result = json.loads(lambda_function.lambda_handler(event, None)['body'])["predictions"]

    assert isinstance(result, list)


def test_singleton_input():
    event = mock_event(num_observations=1)
    try:
        result = json.loads(lambda_function.lambda_handler(event, None)['body'])["predictions"]
    except Exception as e:
        result = None

    assert result is None
    