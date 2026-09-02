'use client';

import { useEffect, useRef, useState } from 'react';
import Waveform from '../components/Waveform';
import RiskGauge from '../components/RiskGauge';
import FeaturePanel from '../components/FeaturePanel';
import AlertBanner from '../components/AlertBanner';
import CallHistoryTable from '../components/CallHistoryTable';
import modelWeights from '../lib/model_weights.json';

export default function Dashboard() {
  const [samples, setSamples] = useState(null);
  const [sampleRate, setSampleRate] = useState(16000);
  const [status, setStatus] = useState('idle');
  const [errorMsg, setErrorMsg] = useState('');
  const [result, setResult] = useState(null);
  const [calls, setCalls] = useState([]);

  const [callerLabel, setCallerLabel] = useState('Inbound caller');
  const [knownContact, setKnownContact] = useState(true);
  const [highValueTransaction, setHighValueTransaction] = useState(false);
  const [callChannel, setCallChannel] = useState('voip');

  const recorderRef = useRef(null);
  const fileInputRef = useRef(null);

  // Load call history when dashboard starts.
  useEffect(() => {
    refreshHistory();
  }, []);

  // Cleanup microphone when leaving the page.
  useEffect(() => {
    return () => {
      try {
        recorderRef.current?.stop();
      } catch {
        // Ignore cleanup errors.
      }
    };
  }, []);

  async function refreshHistory() {
    try {
      const res = await fetch('/api/history');

      if (!res.ok) {
        throw new Error(`History request failed: HTTP ${res.status}`);
      }

      const data = await res.json();

      if (data.ok) {
        setCalls(data.calls || []);
      }
    } catch (err) {
      console.warn('[VoiceGuard] History error:', err);
    }
  }

  // ------------------------------------------------------------
  // Upload audio file
  // ------------------------------------------------------------
  async function handleFile(file) {
    if (!file) return;

    setErrorMsg('');
    setResult(null);
    setStatus('processing');

    try {
      console.log('[VoiceGuard] Processing uploaded file:', file.name);

      const { decodeToMonoPCM } =
        await import('../lib/clientAudio');

      const {
        samples: pcm,
        sampleRate: sr,
      } = await decodeToMonoPCM(file);

      console.log(
        `[VoiceGuard] Decoded ${pcm.length} samples at ${sr} Hz`
      );

      setSamples(pcm);
      setSampleRate(sr);

      await analyze(pcm, sr);
    } catch (err) {
      console.error(
        '[VoiceGuard] Upload processing error:',
        err
      );

      setStatus('error');

      setErrorMsg(
        err?.message ||
          'Failed to decode audio.'
      );
    }
  }

  // ------------------------------------------------------------
  // Start microphone recording
  // ------------------------------------------------------------
  async function startRecording() {
    setErrorMsg('');
    setResult(null);

    // Clear old waveform before starting.
    setSamples(null);
    setStatus('processing');

    try {
      console.log(
        '[VoiceGuard] Starting microphone capture...'
      );

      // Browser capability check.
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          'Microphone API is unavailable. Please use Chrome/Edge on localhost or HTTPS.'
        );
      }

      const { createRecorder } =
        await import('../lib/clientAudio');

      recorderRef.current = createRecorder({
        // ------------------------------------------------------
        // LIVE waveform data
        // ------------------------------------------------------
        onSamples: (liveSamples) => {
          setSamples(liveSamples);
          setSampleRate(16000);
        },

        // ------------------------------------------------------
        // Recording stopped
        // ------------------------------------------------------
        onStop: async (blob) => {
          console.log(
            `[VoiceGuard] Recording stopped. Blob size: ${blob.size} bytes`
          );

          setStatus('processing');

          try {
            const { decodeToMonoPCM } =
              await import('../lib/clientAudio');

            const {
              samples: pcm,
              sampleRate: sr,
            } = await decodeToMonoPCM(blob);

            console.log(
              `[VoiceGuard] Recording decoded: ${pcm.length} samples at ${sr} Hz`
            );

            setSamples(pcm);
            setSampleRate(sr);

            await analyze(pcm, sr);
          } catch (err) {
            console.error(
              '[VoiceGuard] Recording processing error:',
              err
            );

            setStatus('error');

            setErrorMsg(
              err?.message ||
                'Failed to process recording.'
            );
          }
        },

        // ------------------------------------------------------
        // Recorder error
        // ------------------------------------------------------
        onError: (err) => {
          console.error(
            '[VoiceGuard] MediaRecorder error:',
            err
          );

          setStatus('error');

          setErrorMsg(
            err?.message ||
              'Recording error.'
          );
        },
      });

      await recorderRef.current.start();

      setStatus('recording');

      console.log(
        '[VoiceGuard] Microphone recording active.'
      );
    } catch (err) {
      console.error(
        '[VoiceGuard] Microphone error:',
        err
      );

      setStatus('error');

      // Give the actual browser error instead of hiding it.
      let message = 'Microphone access denied or unavailable.';

      if (err?.name === 'NotAllowedError') {
        message =
          'Microphone permission was denied. Allow microphone access for localhost and try again.';
      } else if (err?.name === 'NotFoundError') {
        message =
          'No microphone was found. Connect a microphone and try again.';
      } else if (err?.name === 'NotReadableError') {
        message =
          'The microphone is already being used by another application.';
      } else if (err?.name === 'SecurityError') {
        message =
          'Microphone access is blocked by the browser security settings.';
      } else if (err?.message) {
        message = err.message;
      }

      setErrorMsg(message);
    }
  }

  // ------------------------------------------------------------
  // Stop recording
  // ------------------------------------------------------------
  function stopRecording() {
    console.log(
      '[VoiceGuard] Stop button pressed.'
    );

    recorderRef.current?.stop();
  }

  // ------------------------------------------------------------
  // Send extracted features to backend
  // ------------------------------------------------------------
  async function analyze(pcm, sr) {
    try {
      console.log(
        '[VoiceGuard] Extracting acoustic features...'
      );

      const { extractFeatures } =
        await import('../lib/audioFeatures');

      const {
        vector,
        featureMap,
        meta,
      } = extractFeatures(pcm, sr);

      console.log(
        `[VoiceGuard] Extracted ${vector.length}-dimensional feature vector.`
      );

      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          featureVector: vector,
          featureMap,
          meta,
          callerLabel,
          context: {
            knownContact,
            highValueTransaction,
            callChannel,
          },
        }),
      });

      const text = await res.text();

      let data;

      try {
        data = JSON.parse(text);
      } catch {
        console.error(
          '[VoiceGuard] Backend returned non-JSON:',
          text.slice(0, 500)
        );

        throw new Error(
          `Backend returned an invalid response (HTTP ${res.status}).`
        );
      }

      if (!res.ok || !data.ok) {
        throw new Error(
          data.error ||
            `Analysis failed with HTTP ${res.status}.`
        );
      }

      console.log(
        '[VoiceGuard] Analysis successful:',
        data.record
      );

      setResult(data.record);
      setStatus('done');

      await refreshHistory();
    } catch (err) {
      console.error(
        '[VoiceGuard] Analysis error:',
        err
      );

      setStatus('error');

      setErrorMsg(
        err?.message ||
          'Analysis failed.'
      );
    }
  }

  const risk = result?.risk;

  return (
    <main className="max-w-6xl mx-auto px-6 py-10">

      {/* =====================================================
          HEADER
      ====================================================== */}
      <header className="mb-8 flex items-start justify-between gap-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-50">
            VoiceGuard
          </h1>

          <p className="text-sm text-slate-400 mt-1 max-w-xl">
            Real-time voice cloning &amp; synthetic-speech detection.
            Record or upload a call clip to compute an impersonation
            risk score before any sensitive action is approved.
          </p>
        </div>

        <div className="hidden sm:flex flex-col items-end text-xs text-slate-500 font-mono">
          <span>PS-ID 26104</span>
          <span>
            classifier {modelWeights.version}
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ===================================================
            CAPTURE + WAVEFORM
        ==================================================== */}
        <section className="lg:col-span-2 card p-5">

          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-medium text-slate-200">
              Live capture
            </h2>

            <span className="text-xs font-mono text-slate-500">
              {status === 'recording'
                ? '● RECORDING'
                : status === 'processing'
                  ? 'ANALYZING…'
                  : status.toUpperCase()}
            </span>
          </div>

          {/* LIVE WAVEFORM */}
          <Waveform
            samples={samples}
            sampleRate={sampleRate}
          />

          {/* CONTROLS */}
          <div className="flex flex-wrap gap-2 mt-4">

            {status !== 'recording' ? (
              <button
                onClick={startRecording}
                disabled={status === 'processing'}
                className="px-3 py-1.5 text-sm rounded-md bg-accent/15 text-accent border border-accent/30 hover:bg-accent/25 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                ● Record from mic
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="px-3 py-1.5 text-sm rounded-md bg-danger/15 text-danger border border-danger/30 hover:bg-danger/25 transition-colors"
              >
                ■ Stop &amp; analyze
              </button>
            )}

            <button
              onClick={() =>
                fileInputRef.current?.click()
              }
              disabled={status === 'processing'}
              className="px-3 py-1.5 text-sm rounded-md bg-panel2 text-slate-200 border border-line hover:border-slate-500 transition-colors disabled:opacity-50"
            >
              Upload audio clip
            </button>

            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];

                if (file) {
                  handleFile(file);
                }

                // Allow selecting the same file again.
                e.target.value = '';
              }}
            />
          </div>

          {/* ERROR */}
          {errorMsg && (
            <div className="mt-3 p-3 rounded-md border border-danger/30 bg-danger/10">
              <p className="text-xs text-danger">
                {errorMsg}
              </p>
            </div>
          )}

          {/* =================================================
              CALL CONTEXT
          ================================================== */}
          <div className="mt-5 pt-4 border-t border-line grid grid-cols-1 sm:grid-cols-2 gap-3">

            <label className="text-xs text-slate-400 flex flex-col gap-1">
              Caller label

              <input
                value={callerLabel}
                onChange={(e) =>
                  setCallerLabel(e.target.value)
                }
                className="bg-panel2 border border-line rounded-md px-2 py-1.5 text-slate-200 text-sm focus:outline-none focus:border-accent/50"
              />
            </label>

            <label className="text-xs text-slate-400 flex flex-col gap-1">
              Call channel

              <select
                value={callChannel}
                onChange={(e) =>
                  setCallChannel(e.target.value)
                }
                className="bg-panel2 border border-line rounded-md px-2 py-1.5 text-slate-200 text-sm focus:outline-none focus:border-accent/50"
              >
                <option value="voip">
                  VoIP
                </option>

                <option value="pstn">
                  PSTN / mobile
                </option>

                <option value="collab">
                  Collaboration platform
                </option>
              </select>
            </label>

            <label className="text-xs text-slate-400 flex items-center gap-2 mt-1">
              <input
                type="checkbox"
                checked={knownContact}
                onChange={(e) =>
                  setKnownContact(e.target.checked)
                }
              />

              Caller is a verified known contact
            </label>

            <label className="text-xs text-slate-400 flex items-center gap-2 mt-1">
              <input
                type="checkbox"
                checked={highValueTransaction}
                onChange={(e) =>
                  setHighValueTransaction(
                    e.target.checked
                  )
                }
              />

              High-value transaction pending
            </label>
          </div>
        </section>

        {/* ===================================================
            RISK GAUGE
        ==================================================== */}
        <section className="card p-5 flex flex-col items-center justify-center">

          <RiskGauge
            score={risk?.score}
            level={risk?.level}
            color={risk?.color}
            summary={risk?.summary}
          />

        </section>

        {/* ===================================================
            FEATURE TELEMETRY
        ==================================================== */}
        <section className="card p-5">

          <h2 className="text-sm font-medium text-slate-200 mb-3">
            Acoustic telemetry
          </h2>

          <FeaturePanel
            featureMap={result?.featureMap || null}
            heuristicFlags={risk?.heuristicFlags}
          />

        </section>

        {/* ===================================================
            ALERTS
        ==================================================== */}
        <section className="lg:col-span-2">

          <AlertBanner risk={risk} />

        </section>
      </div>

      {/* =====================================================
          HISTORY
      ====================================================== */}
      <section className="card p-5 mt-6">

        <h2 className="text-sm font-medium text-slate-200 mb-3">
          Recent calls (this session)
        </h2>

        <CallHistoryTable calls={calls} />

      </section>

    </main>
  );
}