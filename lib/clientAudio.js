// Browser audio helpers for VoiceGuard.
// Microphone capture uses raw PCM -> WAV instead of MediaRecorder/WebM.
// This avoids browser-specific audio decoding problems.

const TARGET_SAMPLE_RATE = 16000;

/* -----------------------------------------------------------
   AudioContext
----------------------------------------------------------- */

function getAudioContext() {
  const Ctx =
    window.AudioContext ||
    window.webkitAudioContext;

  if (!Ctx) {
    throw new Error(
      'Web Audio API is not supported by this browser.'
    );
  }

  return new Ctx();
}

/* -----------------------------------------------------------
   Downmix
----------------------------------------------------------- */

function toMono(audioBuffer) {
  if (audioBuffer.numberOfChannels === 1) {
    return audioBuffer.getChannelData(0);
  }

  const length = audioBuffer.length;
  const out = new Float32Array(length);

  for (
    let channel = 0;
    channel < audioBuffer.numberOfChannels;
    channel++
  ) {
    const data =
      audioBuffer.getChannelData(channel);

    for (let i = 0; i < length; i++) {
      out[i] +=
        data[i] /
        audioBuffer.numberOfChannels;
    }
  }

  return out;
}

/* -----------------------------------------------------------
   Resampling
----------------------------------------------------------- */

function resample(
  samples,
  fromRate,
  toRate
) {
  if (fromRate === toRate) {
    return samples;
  }

  const ratio =
    fromRate / toRate;

  const newLength =
    Math.floor(
      samples.length / ratio
    );

  const output =
    new Float32Array(newLength);

  for (
    let i = 0;
    i < newLength;
    i++
  ) {
    const sourcePosition =
      i * ratio;

    const index0 =
      Math.floor(sourcePosition);

    const index1 =
      Math.min(
        index0 + 1,
        samples.length - 1
      );

    const fraction =
      sourcePosition - index0;

    output[i] =
      samples[index0] *
        (1 - fraction) +
      samples[index1] *
        fraction;
  }

  return output;
}

/* -----------------------------------------------------------
   Decode uploaded files
----------------------------------------------------------- */

async function decodeToMonoPCM(blob) {
  /*
   * STEP 1
   * Try the browser's native decoder first.
   *
   * This is fast for WAV/MP3 files that the browser
   * can decode normally.
   */

  try {
    const arrayBuffer =
      await blob.arrayBuffer();

    const ctx =
      getAudioContext();

    try {
      const audioBuffer =
        await ctx.decodeAudioData(
          arrayBuffer.slice(0)
        );

      const mono =
        toMono(audioBuffer);

      const resampled =
        resample(
          mono,
          audioBuffer.sampleRate,
          TARGET_SAMPLE_RATE
        );

      return {
        samples: resampled,
        sampleRate: TARGET_SAMPLE_RATE,
        originalSampleRate:
          audioBuffer.sampleRate,
        durationSec:
          audioBuffer.duration,
      };
    } finally {
      await ctx.close();
    }

  } catch (nativeError) {

    /*
     * Browser failed to decode the file.
     *
     * Example:
     * EncodingError: Unable to decode audio data
     *
     * Fall back to FFmpeg.
     */

    console.warn(
      '[VoiceGuard] Browser audio decoder failed. Falling back to FFmpeg:',
      nativeError
    );
  }

  /*
   * STEP 2
   * Send the original audio file to our Next.js API.
   *
   * /api/convert-audio uses FFmpeg to convert:
   *
   * MP3 / M4A / other supported format
   *          ↓
   * 16 kHz
   * Mono
   * PCM WAV
   */

  const formData =
    new FormData();

  formData.append(
    'file',
    blob,
    blob.name || 'audio'
  );

  console.log(
    '[VoiceGuard] Sending audio to FFmpeg converter...'
  );

  const response =
    await fetch(
      '/api/convert-audio',
      {
        method: 'POST',
        body: formData,
      }
    );

  /*
   * Handle FFmpeg/API errors.
   */

  if (!response.ok) {
    let message =
      'FFmpeg audio conversion failed.';

    try {
      const body =
        await response.json();

      if (body?.error) {
        message = body.error;
      }
    } catch {
      // Response was not JSON.
    }

    throw new Error(message);
  }

  /*
   * STEP 3
   * Receive the converted WAV.
   */

  const convertedBlob =
    await response.blob();

  console.log(
    `[VoiceGuard] FFmpeg conversion complete: ${(
      convertedBlob.size / 1024 / 1024
    ).toFixed(2)} MB`
  );

  /*
   * STEP 4
   * Decode the WAV returned by FFmpeg.
   *
   * WAV is much more reliable for
   * decodeAudioData().
   */

  const convertedArrayBuffer =
    await convertedBlob.arrayBuffer();

  const ctx =
    getAudioContext();

  try {
    const audioBuffer =
      await ctx.decodeAudioData(
        convertedArrayBuffer.slice(0)
      );

    const mono =
      toMono(audioBuffer);

    const resampled =
      resample(
        mono,
        audioBuffer.sampleRate,
        TARGET_SAMPLE_RATE
      );

    return {
      samples: resampled,
      sampleRate: TARGET_SAMPLE_RATE,
      originalSampleRate:
        audioBuffer.sampleRate,
      durationSec:
        audioBuffer.duration,
    };

  } finally {
    await ctx.close();
  }
}

/* -----------------------------------------------------------
   Float32 PCM -> WAV
----------------------------------------------------------- */

function encodeWav(
  samples,
  sampleRate
) {
  const bytesPerSample = 2;
  const numChannels = 1;

  const dataSize =
    samples.length *
    bytesPerSample;

  const buffer =
    new ArrayBuffer(
      44 + dataSize
    );

  const view =
    new DataView(buffer);

  function writeString(
    offset,
    text
  ) {
    for (
      let i = 0;
      i < text.length;
      i++
    ) {
      view.setUint8(
        offset + i,
        text.charCodeAt(i)
      );
    }
  }

  // RIFF header
  writeString(0, 'RIFF');

  view.setUint32(
    4,
    36 + dataSize,
    true
  );

  writeString(8, 'WAVE');

  // fmt chunk
  writeString(12, 'fmt ');

  view.setUint32(
    16,
    16,
    true
  );

  // PCM format
  view.setUint16(
    20,
    1,
    true
  );

  view.setUint16(
    22,
    numChannels,
    true
  );

  view.setUint32(
    24,
    sampleRate,
    true
  );

  view.setUint32(
    28,
    sampleRate *
      numChannels *
      bytesPerSample,
    true
  );

  view.setUint16(
    32,
    numChannels *
      bytesPerSample,
    true
  );

  view.setUint16(
    34,
    16,
    true
  );

  // data chunk
  writeString(36, 'data');

  view.setUint32(
    40,
    dataSize,
    true
  );

  // Convert Float32 -> signed 16-bit PCM
  let offset = 44;

  for (
    let i = 0;
    i < samples.length;
    i++
  ) {
    let sample =
      Math.max(
        -1,
        Math.min(1, samples[i])
      );

    const value =
      sample < 0
        ? sample * 32768
        : sample * 32767;

    view.setInt16(
      offset,
      value,
      true
    );

    offset += 2;
  }

  return new Blob(
    [buffer],
    {
      type: 'audio/wav',
    }
  );
}

/* -----------------------------------------------------------
   Microphone recorder
----------------------------------------------------------- */

function createRecorder({
  onStop,
  onError,
  onSamples,
}) {
  let mediaStream = null;

  let audioContext = null;
  let source = null;
  let analyser = null;
  let processor = null;

  let animationFrame = null;

  let recording = false;

  // Raw PCM chunks
  const pcmChunks = [];

  async function start() {
    try {
      if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
      ) {
        throw new Error(
          'Microphone API is unavailable. Use Chrome/Edge on localhost or HTTPS.'
        );
      }

      console.log(
        '[VoiceGuard] Requesting microphone...'
      );

      mediaStream =
        await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });

      console.log(
        '[VoiceGuard] Microphone permission granted.'
      );

      audioContext =
        getAudioContext();

      if (
        audioContext.state ===
        'suspended'
      ) {
        await audioContext.resume();
      }

      console.log(
        `[VoiceGuard] Microphone sample rate: ${audioContext.sampleRate} Hz`
      );

      source =
        audioContext.createMediaStreamSource(
          mediaStream
        );

      // ---------------------------------------
      // LIVE WAVEFORM
      // ---------------------------------------

      analyser =
        audioContext.createAnalyser();

      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.75;

      source.connect(analyser);

      const waveformBuffer =
        new Float32Array(
          analyser.fftSize
        );

      function updateWaveform() {
        if (!recording) {
          return;
        }

        analyser.getFloatTimeDomainData(
          waveformBuffer
        );

        onSamples?.(
          new Float32Array(
            waveformBuffer
          )
        );

        animationFrame =
          requestAnimationFrame(
            updateWaveform
          );
      }

      // ---------------------------------------
      // RAW PCM CAPTURE
      // ---------------------------------------

      // ScriptProcessorNode is deprecated in modern
      // browsers but remains widely supported and is
      // very convenient for this prototype.

      processor =
        audioContext.createScriptProcessor(
          4096,
          1,
          1
        );

      processor.onaudioprocess =
        (event) => {
          if (!recording) {
            return;
          }

          const input =
            event.inputBuffer
              .getChannelData(0);

          // Copy the buffer because the browser
          // reuses the original AudioBuffer.
          pcmChunks.push(
            new Float32Array(input)
          );
        };

      source.connect(processor);

      // Keep processor alive.
      processor.connect(
        audioContext.destination
      );

      recording = true;

      updateWaveform();

      console.log(
        '[VoiceGuard] Recording started.'
      );

    } catch (err) {
      console.error(
        '[VoiceGuard] Microphone start error:',
        err
      );

      cleanup();

      throw err;
    }
  }

  function stop() {
    if (!recording) {
      return;
    }

    console.log(
      '[VoiceGuard] Stopping recording...'
    );

    recording = false;

    if (animationFrame) {
      cancelAnimationFrame(
        animationFrame
      );

      animationFrame = null;
    }

    try {
      // ---------------------------------------
      // Combine PCM chunks
      // ---------------------------------------

      let totalLength = 0;

      for (
        const chunk of pcmChunks
      ) {
        totalLength +=
          chunk.length;
      }

      if (totalLength === 0) {
        throw new Error(
          'No microphone audio was captured.'
        );
      }

      const combined =
        new Float32Array(
          totalLength
        );

      let offset = 0;

      for (
        const chunk of pcmChunks
      ) {
        combined.set(
          chunk,
          offset
        );

        offset +=
          chunk.length;
      }

      // ---------------------------------------
      // Resample to 16 kHz
      // ---------------------------------------

      const inputRate =
        audioContext.sampleRate;

      const samples =
        resample(
          combined,
          inputRate,
          TARGET_SAMPLE_RATE
        );

      console.log(
        `[VoiceGuard] Captured ${samples.length} samples at ${TARGET_SAMPLE_RATE} Hz`
      );

      // ---------------------------------------
      // Create WAV
      // ---------------------------------------

      const wavBlob =
        encodeWav(
          samples,
          TARGET_SAMPLE_RATE
        );

      console.log(
        `[VoiceGuard] WAV created: ${wavBlob.size} bytes`
      );

      cleanup();

      onStop?.(wavBlob);

    } catch (err) {
      console.error(
        '[VoiceGuard] Recording processing error:',
        err
      );

      cleanup();

      onError?.(err);
    }
  }

  function cleanup() {
    recording = false;

    if (animationFrame) {
      cancelAnimationFrame(
        animationFrame
      );

      animationFrame = null;
    }

    try {
      processor?.disconnect();
    } catch {}

    try {
      source?.disconnect();
    } catch {}

    try {
      analyser?.disconnect();
    } catch {}

    processor = null;
    source = null;
    analyser = null;

    if (mediaStream) {
      mediaStream
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      mediaStream = null;
    }

    if (audioContext) {
      audioContext
        .close()
        .catch(() => {});

      audioContext = null;
    }

    pcmChunks.length = 0;
  }

  return {
    start,
    stop,
  };
}

/* -----------------------------------------------------------
   Exports
----------------------------------------------------------- */

export {
  TARGET_SAMPLE_RATE,
  decodeToMonoPCM,
  createRecorder,
};