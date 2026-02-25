## BUILD
sudo docker build -t appinfer:v1 -f Dockerfile_app .
sudo docker build -t appprefect:v1 -f Dockerfile_prefect .


## RUN

# W/ INTERNAL NETWORK
docker network create app-network

sudo docker run -it --rm --network app-network --name prefect -e PREFECT_SERVER_API_HOST=0.0.0.0 -e PREFECT_SERVER_API_PORT=4002 -p 4002:4002 appprefect:v1

sudo docker run -it --rm -v /home/luca/customnn/mlruns --network app-network --name app -p 8002:8002 -e PREFECT_API_URL=http://prefect:4002/api appinfer:v1

*(or use docker compose: docker compose up)*

# W/O INTERNAL NETWORK
sudo docker run -it --rm -e PREFECT_SERVER_API_HOST=0.0.0.0 -e PREFECT_SERVER_API_PORT=4002 -p 4002:4002 appprefect:v1

sudo docker run -it --rm -v /home/luca/customnn/mlruns -p 8002:8002 -e PREFECT_API_URL=http://prefect:4002/api appinfer:v1