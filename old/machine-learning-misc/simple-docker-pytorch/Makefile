help:
	@cat Makefile
	
build:
	docker build -t pytorch:cuda-9.0 -f Dockerfile .

main:
	docker run --rm -it --init \
	  --runtime=nvidia \
	  --ipc=host \
	  --user="$(id -u):$(id -g)" \
	  --volume=$PWD:/app \
	  -e NVIDIA_VISIBLE_DEVICES=0 \
	  pytorch:cuda-9.0 python3 main.py