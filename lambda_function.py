import boto3
import json
import base64

from custom_rul_prediction import infer


kinesis_client = boto3.client('kinesis')


def lambda_handler(event, context):
    predictions_events = []

    for i, record in enumerate(event['Records']):
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')

        pred = infer(decoded_data)
        prediction_event = {record['partitionKey']: pred}

        predictions_events.append(prediction_event)

        kinesis_client.put_record(
                StreamName="output_stream",
                Data=json.dumps(prediction_event),
                PartitionKey=str(i+1)
            )

    return {
        'statusCode': 200,
        'body': json.dumps({"predictions": predictions_events})
    }
