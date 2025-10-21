docker run -e XINFERENCE_MODEL_SRC=modelscope -p 9998:9997 --gpus all xprobe/xinference:<your_version> xinference-local -H 0.0.0.0 --log-level debug
docker run \
  -v </your/home/path>/.xinference:/root/.xinference \
  -v </your/home/path>/.cache/huggingface:/root/.cache/huggingface \
  -v </your/home/path>/.cache/modelscope:/root/.cache/modelscope \
  -p 9997:9997 \
  --gpus all \
  xprobe/xinference:v<your_version> \
  xinference-local -H 0.0.0.0