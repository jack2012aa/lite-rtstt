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
rtstt run

# Run with debug logging enabled
rtstt run --debug
```

*The server exposes a WebSocket endpoint at `/rtstt`.*

### 2. Live Microphone Client

Connects to the server and streams audio from your default microphone input.

```bash
# Connect to localhost (default)
rtstt live

# Connect to a specific remote server
rtstt live --url ws://192.168.1.10:8000/rtstt
```

### 3. File Transcription

Simulates a real-time stream using a local int16 PCM or WAV file. Useful for testing or transcribing pre-recorded audio.

```bash
rtstt transcribe --file test/data/7s_i16.pcm
```

---

## ⚙️ Configuration

The application uses a configuration dataclass `STTConfig`. When running in a Snap environment, it looks for `stt_config.json` in `SNAP_DATA`. Otherwise, it uses default values.

**Default Configuration:**

* **VAD Threads**: 4 
* **Whisper Model**: `base` 
* **Sample Rate**: 16000 Hz 
* **Chunk Size**: 30 ms 
* **Silence Duration**: 1200 ms (Time to wait before cutting the sentence) 



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