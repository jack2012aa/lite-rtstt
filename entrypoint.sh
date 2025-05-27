#!/bin/sh
set -e
if [ "$API" = "true" ]; then
  echo "INFO: API mode enabled. Running: python3 app.py --api"
  exec python3 app.py --api
else
  if [ "$CUDA" = "true" ]; then
    echo "INFO: CUDA mode enabled. Running: python3 app.py --cuda --port ${PORT} --pool_size ${POOL_SIZE} --model_type ${MODEL_TYPE}"
    exec python3 app.py --cuda --port "${PORT}" --pool_size "${POOL_SIZE}" --model_type "${MODEL_TYPE}"
  else
    echo "INFO: CPU mode enabled. Running: python3 app.py --port ${PORT} --pool_size ${POOL_SIZE} --model_type ${MODEL_TYPE}"
    exec python3 app.py --port "${PORT}" --pool_size "${POOL_SIZE}" --model_type "${MODEL_TYPE}"
  fi
fi