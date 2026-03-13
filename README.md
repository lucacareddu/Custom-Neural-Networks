## Kinesis from the AWS CLI

KINESIS_STREAM_INPUT=input_stream
aws kinesis put-record \
    --stream-name ${KINESIS_STREAM_INPUT} \
    --partition-key 1 \
    --cli-binary-format raw-in-base64-out \
    --data 'Hello, this is a test.'


KINESIS_STREAM_OUTPUT=output_stream
SHARD='shardId-000000000000'

SHARD_ITERATOR=$(aws kinesis \
    get-shard-iterator \
        --shard-id ${SHARD} \
        --shard-iterator-type LATEST \
        --stream-name ${KINESIS_STREAM_OUTPUT} \
        --query 'ShardIterator' \
)

RESULT=$(aws kinesis get-records --shard-iterator $SHARD_ITERATOR)

echo ${RESULT} | jq -r '.Records[0].Data' | base64 --decode


## Docker Container (Lambda Function)

docker build --provenance false -t lambdainfer:v1 .

docker run -it --rm -e AWS_ACCESS_KEY_ID= -e AWS_SECRET_ACCESS_KEY= -e AWS_DEFAULT_REGION=eu-north-1 -p 8080:8080 lambdainfer:v1


## ECR

aws ecr create-repository --repository-name rul_prediction

aws ecr get-login-password --region eu-north-1 \
| docker login --username AWS --password-stdin 023530626203.dkr.ecr.eu-north-1.amazonaws.com


## Docker push in ECR

REMOTE_URI=023530626203.dkr.ecr.eu-north-1.amazonaws.com/rul_prediction
REMOTE_TAG=v1
REMOTE_IMAGE=${REMOTE_URI}:${REMOTE_TAG}

LOCAL_IMAGE=lambdainfer:v1
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE}
docker push ${REMOTE_IMAGE}


## Evidently Monitoring

api_key=