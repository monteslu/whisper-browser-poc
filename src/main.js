/**
 * Whisper Browser POC
 * 
 * Speech-to-text using OpenAI's Whisper model via Transformers.js
 * Models are pre-converted to ONNX and available on HuggingFace
 */

import { pipeline, env } from '@huggingface/transformers';

// Configure Transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

// DOM Elements
const modelSelect = document.getElementById('modelSelect');
const loadModelBtn = document.getElementById('loadModelBtn');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const recordBtn = document.getElementById('recordBtn');
const statusContainer = document.getElementById('statusContainer');
const audioContainer = document.getElementById('audioContainer');
const audioPlayer = document.getElementById('audioPlayer');
const transcribeBtn = document.getElementById('transcribeBtn');
const transcriptContainer = document.getElementById('transcriptContainer');
const transcriptDiv = document.getElementById('transcript');
const timestampsDiv = document.getElementById('timestamps');

// State
let transcriber = null;
let currentAudioUrl = null;
let mediaRecorder = null;
let audioChunks = [];

// Status helpers
function showStatus(message, type = 'loading') {
  statusContainer.innerHTML = `
    <div class="status ${type}">
      ${message}
      ${type === 'loading' ? '<div class="progress-bar"><div class="progress-fill" id="progressFill" style="width: 0%"></div></div>' : ''}
    </div>
  `;
}

function updateProgress(percent) {
  const fill = document.getElementById('progressFill');
  if (fill) fill.style.width = `${percent}%`;
}

function clearStatus() {
  statusContainer.innerHTML = '';
}

// Load Whisper model
async function loadModel() {
  const modelId = modelSelect.value;
  
  loadModelBtn.disabled = true;
  loadModelBtn.textContent = 'Loading...';
  
  showStatus(`📥 Loading ${modelId}...`, 'loading');
  
  try {
    transcriber = await pipeline('automatic-speech-recognition', modelId, {
      progress_callback: (progress) => {
        if (progress.status === 'downloading') {
          const percent = progress.progress || 0;
          updateProgress(percent);
          showStatus(`📥 Downloading: ${progress.file} (${percent.toFixed(1)}%)`, 'loading');
        } else if (progress.status === 'loading') {
          showStatus(`🔧 Loading model into memory...`, 'loading');
        }
      }
    });
    
    showStatus(`✅ Model loaded: ${modelId}`, 'success');
    transcribeBtn.disabled = false;
    recordBtn.disabled = false;
    loadModelBtn.textContent = 'Model Loaded ✓';
    
  } catch (error) {
    console.error('Model load error:', error);
    showStatus(`❌ Failed to load model: ${error.message}`, 'error');
    loadModelBtn.disabled = false;
    loadModelBtn.textContent = 'Load Model';
  }
}

// Process audio file
async function processFile(file) {
  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl);
  }
  
  currentAudioUrl = URL.createObjectURL(file);
  audioPlayer.src = currentAudioUrl;
  audioContainer.style.display = 'block';
  transcriptContainer.style.display = 'none';
  
  showStatus(`✅ Loaded: ${file.name}`, 'success');
}

// Transcribe audio
async function transcribe() {
  if (!transcriber || !currentAudioUrl) return;
  
  transcribeBtn.disabled = true;
  showStatus('🎤 Transcribing...', 'loading');
  transcriptContainer.style.display = 'block';
  transcriptDiv.textContent = '';
  transcriptDiv.classList.add('streaming');
  timestampsDiv.innerHTML = '';
  
  try {
    // Fetch audio as array buffer
    const response = await fetch(currentAudioUrl);
    const arrayBuffer = await response.arrayBuffer();
    
    // Decode audio
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    // Convert to mono Float32Array at target sample rate
    const targetSampleRate = 16000; // Whisper expects 16kHz
    const offlineContext = new OfflineAudioContext(1, audioBuffer.duration * targetSampleRate, targetSampleRate);
    const source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineContext.destination);
    source.start();
    
    const resampledBuffer = await offlineContext.startRendering();
    const audioData = resampledBuffer.getChannelData(0);
    
    showStatus('🧠 Running Whisper inference...', 'loading');
    updateProgress(50);
    
    // Run transcription with timestamps
    const result = await transcriber(audioData, {
      chunk_length_s: 30,
      stride_length_s: 5,
      return_timestamps: true,
      language: 'en',
    });
    
    updateProgress(100);
    
    // Display results
    transcriptDiv.classList.remove('streaming');
    
    if (typeof result.text === 'string') {
      transcriptDiv.textContent = result.text;
    }
    
    // Display timestamps if available
    if (result.chunks && result.chunks.length > 0) {
      let timestampHtml = '<strong>Timestamps:</strong><br>';
      result.chunks.forEach(chunk => {
        const start = formatTime(chunk.timestamp[0]);
        const end = formatTime(chunk.timestamp[1]);
        timestampHtml += `
          <div class="timestamp-chunk" data-start="${chunk.timestamp[0]}">
            <span class="timestamp-time">[${start} → ${end}]</span>
            ${chunk.text}
          </div>
        `;
      });
      timestampsDiv.innerHTML = timestampHtml;
      
      // Click to seek
      timestampsDiv.querySelectorAll('.timestamp-chunk').forEach(el => {
        el.addEventListener('click', () => {
          audioPlayer.currentTime = parseFloat(el.dataset.start);
          audioPlayer.play();
        });
      });
    }
    
    showStatus('✅ Transcription complete!', 'success');
    
  } catch (error) {
    console.error('Transcription error:', error);
    showStatus(`❌ Transcription failed: ${error.message}`, 'error');
    transcriptDiv.classList.remove('streaming');
  }
  
  transcribeBtn.disabled = false;
}

// Format time as MM:SS
function formatTime(seconds) {
  if (seconds === null || seconds === undefined) return '--:--';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Microphone recording
async function toggleRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    recordBtn.textContent = '🎙️ Record from Microphone';
    recordBtn.classList.remove('recording');
    return;
  }
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    
    mediaRecorder.ondataavailable = (e) => {
      audioChunks.push(e.data);
    };
    
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
      currentAudioUrl = URL.createObjectURL(audioBlob);
      audioPlayer.src = currentAudioUrl;
      audioContainer.style.display = 'block';
      showStatus('✅ Recording saved', 'success');
      
      // Stop all tracks
      stream.getTracks().forEach(track => track.stop());
    };
    
    mediaRecorder.start();
    recordBtn.textContent = '⏹️ Stop Recording';
    recordBtn.classList.add('recording');
    showStatus('🔴 Recording...', 'loading');
    
  } catch (error) {
    console.error('Recording error:', error);
    showStatus(`❌ Microphone access denied: ${error.message}`, 'error');
  }
}

// Event handlers
loadModelBtn.addEventListener('click', loadModel);
transcribeBtn.addEventListener('click', transcribe);
recordBtn.addEventListener('click', toggleRecording);

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('audio/')) {
    processFile(file);
  }
});
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) processFile(file);
});

// Log info
console.log('Whisper Browser POC loaded');
console.log('Using Transformers.js for speech recognition');
console.log('Models: whisper-tiny, whisper-base, whisper-small, whisper-medium');
