# Running in a Docker Container
You can run this project inside a docker image.
Steps to run this code in a docker container
# Installation
Go to the project directory
## 1. Build docker container
```$ docker build -t "pyhgt:latest" -f ./Docker/DockerFile .```

## 2. After the successfull build, run the container

```$ docker run --rm -it --init --runtime=nvidia --ipc=host --network=host --volume=$PWD:/home/user/app -e NVIDIA_VISIBLE_DEVICES=0 "pyhgt:latest" /bin/bash```
## 3. Run the code as
