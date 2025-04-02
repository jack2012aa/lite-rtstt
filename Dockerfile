# 使用包含 CUDA 及 cuDNN 的 NVIDIA 映像，根據需要選擇相應版本
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 安裝基礎套件
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 如果想要用 python 指令代替 python3，可以做個連結
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# 將你的程式碼複製進容器
WORKDIR /app
COPY app.py /app
COPY requirements.txt /app
COPY service /app/service

# 安裝 Whisper 及所需的套件
# Whisper 預設會安裝 torch/ffmpeg 等，但你可能需要指定 torch 版本（如要 GPU 加速，需要 CUDA 對應的版本）
RUN sudo apt update && sudo apt install ffmpeg
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 uninstall torch
RUN pip3 cache purge
RUN pip3 install --no-cache-dir torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir openai-whisper

# 例如你有一個 app.py 入口檔
CMD ["python", "app.py"]
