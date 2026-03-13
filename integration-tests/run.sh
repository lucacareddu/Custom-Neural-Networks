#!/usr/bin/env bash

cleanup() {
  docker compose down -v
}

trap cleanup EXIT

# if [[ -z "${GITHUB_ACTIONS}" ]]; then
#   cd "$(dirname "$0")"
# fi

# export PREDICTIONS_STREAM_NAME="ride_predictions"

docker compose up --build -d --wait

# aws --endpoint-url=http://localhost:4566 \
#     kinesis create-stream \
#     --stream-name ${PREDICTIONS_STREAM_NAME} \
#     --shard-count 1

conda run -n e2erul python integration-tests/test_docker.py

EXIT_CODE=$?

if [ ${EXIT_CODE} != 0 ]; then
    echo ERROR: exit code $EXIT_CODE
    docker compose logs
    cleanup
    exit ${EXIT_CODE}
fi


# pipenv run python test_kinesis.py

# EXIT_CODE=$?

# if [ ${EXIT_CODE} != 0 ]; then
#     docker-compose logs
#     docker-compose down
#     exit ${EXIT_CODE}
# fi


cleanup