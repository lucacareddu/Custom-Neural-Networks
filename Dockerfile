FROM public.ecr.aws/lambda/python:3.10.2023.06.28.13-x86_64

RUN pip install -U pip
RUN pip install pipenv 

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "lambda_function.py", "custom_rul_prediction.py", "mlflow.db", "./" ]

CMD [ "lambda_function.lambda_handler" ]