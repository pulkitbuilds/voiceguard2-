# VoiceGuard — Real-Time Voice Clone & Spoof Detection

Prototype for **Problem Statement 26104 — AI-Powered Real-Time Detection and
Prevention of Voice Cloning Impersonation Attacks**.

A Next.js app that records or accepts an uploaded call clip, extracts
acoustic/prosodic features, runs them through an ML spoof classifier, blends
that with rule-based heuristics and call context into a live **impersonation
risk score**, and surfaces it on a dashboard with alerts and recommended
actions — end to end, deployable on Vercel with no native audio codecs or
GPU required.

## Why this architecture

Vercel serverless functions have no native audio codec support and no
GPU/heavy-ML runtime by default. Rather than fight that, the pipeline is
split at the one place that's genuinely free: the **browser already knows
how to decode audio** (Web Audio API handles WAV/MP3/OGG/WebM/M4A natively).
So:

- **Client (browser):** decode audio → mono 16kHz PCM → extract a 45-dim
  acoustic feature vector (spectral, cepstral/MFCC, prosodic).
- **Server (Vercel serverless function):** run the trained classifier +
  risk-scoring engine on that feature vector and return a risk score,
  flags, and recommendations. Pure JS, no native deps, sub-50ms.

This mirrors the brief's own architecture ("extract discriminative
features... compute a dynamic impersonation risk score... expose APIs for
integration") while staying inside what a serverless platform can run.

```
 ┌─────────────────────────── Browser ───────────────────────────┐
 │  mic / file  →  Web Audio decode  →  feature extraction        │
 │  (lib/clientAudio.js)              (lib/audioFeatures.js)      │
 └──────────────────────────────┬──────────────────────────────--┘
                                 │ POST /api/analyze  { featureVector, context }
 ┌───────────────────────────── Vercel  ───────────────────────────┐
 │  ML spoof classifier   →   risk scoring engine   →   response   │
 │  (lib/classifier.js)       (lib/riskEngine.js)                  │
 │  reads lib/model_weights.json                                   │
 └───────────────────────────────────────────────────────────────-┘
                                 │
                          dashboard (app/page.js): waveform, risk gauge,
                          acoustic telemetry, alert banner, call history
```

## What maps to the problem statement

| Brief component | Implementation |
|---|---|
| Multi-layer voice authenticity analysis | `lib/audioFeatures.js` — spectral (centroid, flatness, rolloff, bandwidth), cepstral (13 MFCCs mean/std), prosodic (pitch/F0, jitter, shimmer) |
| Real-time risk scoring engine | `lib/riskEngine.js` — blends classifier probability with weighted rule-based flags, configurable thresholds, contextual escalation |
| Alerting / user interaction layer | `components/AlertBanner.jsx` + `recommendations` in the risk response |
| Contextual enrichment | `context` object (known contact, high-value transaction, call channel) sent with each analysis |
| Privacy / compliance module | Only the derived feature vector is ever sent to the server or stored — raw audio never leaves the browser (see `lib/store.js` header) |
| Platform & integration APIs | `POST /api/analyze`, `GET /api/history` — plain REST/JSON, easy to call from a banking app, contact-center platform, or telecom system |
| Multilingual / Indian accents | Features are language-agnostic (acoustic/prosodic, not phoneme-based); retrain `scripts/train_classifier.py` on Indian-language + accent data for production accuracy |

## The ML spoof classifier

`lib/classifier.js` runs a small 2-layer network (Linear → ReLU → Linear →
Sigmoid) in pure JS against weights in `lib/model_weights.json`.

**The shipped `model_weights.json` is a hand-authored demo model** — its
hidden units directly encode known spoof-detection heuristics from the
literature (flat spectrum, abnormally low jitter/shimmer, low pitch/MFCC
variance, HF synthesis artifacts) so the dashboard is fully functional out
of the box, but it is **not trained on a labeled corpus** and should not be
trusted for real fraud decisions.

### Training a real model

```bash
cd scripts
pip install -r requirements.txt

# Expected layout: dataset/bonafide/*.wav, dataset/spoof/*.wav
python train_classifier.py --data_dir /path/to/dataset --out ../lib/model_weights.json
```

Good labeled datasets to start from: **ASVspoof 2019/2021** (LA/DF
partitions), **WaveFake**, **In-the-Wild**, or your own recorded-vs-cloned
pairs (e.g. clone a few speakers with a couple of TTS/voice-conversion
tools to get realistic spoof examples). `scripts/extract_features.py`
mirrors `lib/audioFeatures.js` feature-for-feature, so the exported
`model_weights.json` drops straight in — **no app code changes needed**.

## Local development

```bash
npm install
npm run dev
# → http://localhost:3000
```

Click **Record from mic** (grants mic permission, click **Stop & analyze**
when done) or **Upload audio clip** to test with a file. The risk gauge,
acoustic telemetry panel, and alert banner update immediately; the call
history table logs every analysis for the session.

## Deploying to Vercel

**Option A — Vercel dashboard (no CLI):**
1. Push this project to a GitHub/GitLab/Bitbucket repo.
2. Go to [vercel.com/new](https://vercel.com/new) and import the repo.
3. Framework preset auto-detects **Next.js** — leave build command
   (`next build`) and output settings as default.
4. Click **Deploy**. No environment variables are required for the demo.

**Option B — Vercel CLI:**
```bash
npm install -g vercel
cd voice-guard
vercel        # first deploy, follow the prompts
vercel --prod # promote to production
```

**Notes for production:**
- API routes are set to `runtime = 'nodejs'` (see `app/api/*/route.js`) —
  works on any Vercel plan, no Edge-runtime constraints.
- Microphone recording requires HTTPS or `localhost` (Vercel deployments are
  HTTPS by default, so this works out of the box).
- No environment variables, database, or secrets are required for the base
  prototype to run.

### Persistence

`lib/store.js` is an **in-memory, demo-only** call history — Vercel
functions are stateless and multi-instance, so this resets on cold start
and isn't shared across instances. For real deployments, swap it for
[Vercel KV](https://vercel.com/docs/storage/vercel-kv) or
[Vercel Postgres](https://vercel.com/docs/storage/vercel-postgres): the
`pushCall` / `listCalls` functions are the only two integration points to
change.

## Project structure

```
voice-guard/
  app/
    page.js                 dashboard UI
    layout.js, globals.css
    api/analyze/route.js    POST: classifier + risk engine
    api/history/route.js    GET: recent calls
  components/                Waveform, RiskGauge, FeaturePanel, AlertBanner, CallHistoryTable
  lib/
    fft.js                   dependency-free radix-2 FFT
    audioFeatures.js          spectral/cepstral/prosodic feature extraction (browser + Node)
    clientAudio.js             mic recording + Web Audio decode/resample (browser only)
    classifier.js               MLP inference (reads model_weights.json)
    model_weights.json           classifier weights (demo — see "Training a real model")
    riskEngine.js                 rule-based heuristics + risk blending/scoring
    store.js                       in-memory call history (demo only)
  scripts/
    extract_features.py     Python feature extraction, mirrors lib/audioFeatures.js
    train_classifier.py       offline training → exports lib/model_weights.json
    requirements.txt
```

## Known limitations (prototype scope)

- Classifier weights are a heuristic demo, not trained on real spoof audio — see "Training a real model" above.
- Call history is in-memory and per-instance; not a real audit log.
- No authentication on the API routes — add before exposing publicly / integrating with real banking/telecom systems.
- Pitch tracking uses simple autocorrelation, not a production pitch tracker (e.g. CREPE/YIN) — adequate for a prototype's prosody features, but a real deployment should upgrade this alongside classifier retraining.
- No streaming/near-real-time (in-call) analysis yet — this analyzes a completed clip; wiring this into a live call would mean chunking incoming audio and calling `/api/analyze` on a rolling window.
