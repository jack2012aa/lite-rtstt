# lite-rtstt

**lite-rtstt** is a lightweight, efficient, real-time Speech-to-Text (STT) server and client application. It leverages a three-layer Voice Activity Detection (VAD) pipeline to optimize resource usage, ensuring that the heavy STT model (OpenAI Whisper) is only triggered when meaningful speech is detected.

## 🚀 Features

* **Three-Layer Architecture**: Combines low-latency WebRTC VAD, AI-based Silero VAD, and OpenAI Whisper.
* **Real-time Streaming**: Built on FastAPI and WebSockets for low-latency audio streaming.
* **Snap Integrated**: Designed to be packaged and deployed easily via Snap.
* **Client Tools**: Includes built-in CLI tools for microphone streaming and file transcription.

## 🏗️ Architecture

The core strength of `lite-rtstt` lies in its **Three-Layer Pipeline** designed to save computational resources while maintaining high accuracy.

### 1. The Gatekeeper: WebRTC VAD (Layer 1)

* **Role**: Instant noise filtering.
* **Mechanism**: A lightweight, non-AI algorithm checks 30ms audio chunks.
* **Function**: If the audio is pure silence or background noise, it is discarded immediately. Only "potentially active" audio passes to the buffer.



### 2. The Judge: Silero VAD (Layer 2)

* **Role**: Accurate speech detection.
* **Mechanism**: Once enough "active" chunks are accumulated (defined by `active_to_detection_ms`), this AI-based VAD analyzes the buffer .
* **Function**: It determines if the sound is human speech. If confirmed, the state switches to `SPEAKING`. If it was just a loud noise, the buffer is cleared.

### 3. The Transcriber: OpenAI Whisper (Layer 3)

* **Role**: Speech-to-Text transcription.
* **Mechanism**: When the user stops speaking (detected by a duration of silence), the accumulated audio buffer is sent to the Whisper model.
* **Function**: Returns the transcribed text to the client via WebSocket.

---

## 🛠️ Installation

### Prerequisites

* Python 3.10+
* **PortAudio** (Required for microphone access in the client)
* *Ubuntu/Debian*: `sudo apt-get install portaudio19-dev`
* *macOS*: `brew install portaudio`



### Install from Source

```bash
git clone https://github.com/jack2012aa/lite-rtstt.git
cd lite-rtstt

# Install in editable mode
pip install -e .

```

---

## 📖 Usage

The package provides a unified command-line interface: `rtstt`.

### 1. Start the Server

Starts the FastAPI server on port 8000 (default host 0.0.0.0).

```bash
# Standard run
lite-rtstt run

# Run with debug logging enabled
lite-rtstt run --debug
```

*The server exposes a WebSocket endpoint at `/rtstt`.*

### 2. Live Microphone Client

Connects to the server and streams audio from your default microphone input.

```bash
# Connect to localhost (default)
lite-rtstt live

# Connect to a specific remote server
lite-rtstt live --url ws://192.168.1.10:8000/rtstt
```

### 3. Transcribe a File

Connects to the server and transcribe an int16 pcm.

```bash
# Connect to localhost (default)
lite-rtstt transcribe --file test/data/42s_i16.pcm

# Connect to a specific remote server
lite-rtstt transcribe --url ws://192.168.1.10:8000/rtstt --file test/data/42s_i16.pcm
```

---

這是在 `readme.md` 中關於 Snap 安裝、權限連接與 Daemon 設定的補充內容。這段說明是基於你的程式碼中對 `SNAP_DATA` 環境變數的讀取邏輯  以及標準 Snap 架構撰寫的。

你可以將以下內容新增至 `readme.md` 的 **Installation** 或 **Usage** 章節之後：

---

## 📦 Snap Configuration & Daemon Management

If you installed `lite-rtstt` via Snap, the application runs in a sandboxed environment. You need to configure permissions and the daemon service manually.

### 1. Connect Interfaces (Permissions)

By default, Snap applications are restricted from accessing hardware. To allow `lite-rtstt` to capture audio (for the `live` command) or bind to network ports, you must connect the specific plugs:

```bash
# Allow access to the microphone (Required for 'live' mode)
sudo snap connect lite-rtstt:audio-record

# Allow network access (Usually auto-connected, but required for 'run' server mode)
sudo snap connect lite-rtstt:network-bind

```

### 2. Managing the Server Daemon

The Snap package includes a background service (daemon) for the STT server. You can manage it using standard snap commands:

```bash
# Check the status of the server
sudo snap services lite-rtstt

# Start/Stop/Restart the service
sudo snap start lite-rtstt.server
sudo snap stop lite-rtstt.server
sudo snap restart lite-rtstt.server

# View live logs from the daemon
sudo snap logs -f lite-rtstt.server

```

### 3. Custom Configuration

When running as a Snap, the application looks for the configuration file in the `SNAP_DATA` directory.

* **Config Path**: `/var/snap/lite-rtstt/current/stt_config.json`

To customize parameters (e.g., changing the Whisper model or VAD sensitivity), create or edit this JSON file:

```json
{
  "vad_threads": 4,
  "whisper_model": "base",
  "sample_rate": 16000,
  "chunk_size_ms": 30,
  "duration_time_ms": 1200,
  "active_to_detection_ms": 900,
  "max_buffered_chunks": 500,
  "aggresiveness": 1
}

```

Note: The available keys correspond to the `STTConfig` class. After editing, restart the service with `sudo snap restart lite-rtstt.server` to apply changes.

## 🧪 Development & Testing

To run the test suite (including unit tests and integration tests):

```bash
# Run all tests
python -m unittest discover test

# Run a specific test file
python -m unittest test/rtstt_test.py
```

## 📄 License
MIT License