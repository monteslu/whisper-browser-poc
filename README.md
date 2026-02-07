# 🎤 Whisper Browser POC

Browser-based speech recognition using OpenAI's Whisper model via Transformers.js.

## Features

- **Multiple model sizes**: tiny (~40MB) to medium (~750MB)
- **File upload**: Drag & drop audio files
- **Microphone recording**: Record directly in browser
- **Timestamps**: Click to seek to specific parts
- **Offline capable**: Works without internet after model download
- **WebGPU acceleration**: Uses GPU when available

## Quick Start

```bash
npm install
npm run dev
```

Then open http://localhost:5173

## How It Works

1. **Load Model**: Select a Whisper model size and click "Load Model"
2. **Add Audio**: Drop a file or record from microphone
3. **Transcribe**: Click "Transcribe" to run speech recognition

## Model Sizes

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| whisper-tiny | ~40MB | Fastest | Basic |
| whisper-base | ~75MB | Fast | Good |
| whisper-small | ~250MB | Balanced | Better |
| whisper-medium | ~750MB | Slower | Best |

## Tech Stack

- **Transformers.js**: Browser ML runtime
- **ONNX Runtime Web**: Model inference
- **Web Audio API**: Audio processing
- **Vite**: Dev server & bundler

## Use Cases for Loukai

- **Lyrics transcription**: Auto-generate lyrics from songs
- **Karaoke timing**: Get word-level timestamps for sync
- **Language detection**: Identify song language
- **Accessibility**: Add captions to audio

## Notes

- First load downloads the model (cached in browser)
- Whisper expects 16kHz audio (auto-resampled)
- Longer audio is chunked automatically
- WebGPU provides significant speedup on supported browsers

---

*Built with 🧙‍♂️ magic by Radagast*
