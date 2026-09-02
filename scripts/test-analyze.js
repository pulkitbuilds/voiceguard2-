// Diagnostic script: exercises the backend directly (no browser involved) so
// you can tell for certain whether the problem is server-side or client-side.
//
// Usage (with `npm run dev` already running in another terminal):
//   node scripts/test-analyze.js
//
// It generates a synthetic 2-second signal, extracts the same 45-feature
// vector the browser would, POSTs it to /api/analyze, then reads it back
// from /api/history and /api/ledger. If this script succeeds, the backend
// (classifier + risk engine + store + blockchain) is fully working and the
// issue is purely in the browser (mic permission / secure context / a
// console error) -- not the server.

const path = require('path');
const { extractFeatures } = require(path.join(__dirname, '..', 'lib', 'audioFeatures.js'));

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';

function synthSignal(sr = 16000, dur = 2) {
  const n = sr * dur;
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const f0 = 130 + 8 * Math.sin(2 * Math.PI * 2.2 * t) + 3 * (Math.random() - 0.5);
    samples[i] = 0.3 * Math.sin(2 * Math.PI * f0 * t) + 0.04 * (Math.random() - 0.5);
  }
  return samples;
}

async function main() {
  console.log(`Testing backend at ${BASE_URL} ...\n`);

  // 1. Generate features exactly like the browser does.
  const samples = synthSignal();
  const { vector, featureMap, meta } = extractFeatures(samples, 16000);
  console.log(`✓ Extracted ${vector.length}-dim feature vector locally.`);

  // 2. POST to /api/analyze
  const analyzeRes = await fetch(`${BASE_URL}/api/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      featureVector: vector,
      featureMap,
      meta,
      callerLabel: 'test-analyze.js',
      context: { knownContact: true, highValueTransaction: false, callChannel: 'pstn' },
    }),
  });

  const analyzeBody = await analyzeRes.json().catch(() => null);
  if (!analyzeRes.ok || !analyzeBody?.ok) {
    console.error(`✗ POST /api/analyze failed: HTTP ${analyzeRes.status}`);
    console.error(analyzeBody || '(response was not valid JSON)');
    process.exit(1);
  }
  console.log(`✓ POST /api/analyze succeeded: score=${analyzeBody.record.risk.score} level=${analyzeBody.record.risk.level}`);
  console.log(`  ledger block: #${analyzeBody.record.ledger?.blockIndex} hash=${analyzeBody.record.ledger?.blockHash?.slice(0, 12)}...`);

  // 3. Confirm it shows up in /api/history
  const historyRes = await fetch(`${BASE_URL}/api/history`);
  const historyBody = await historyRes.json();
  const found = historyBody.calls?.some((c) => c.id === analyzeBody.record.id);
  console.log(found ? '✓ Record appears in GET /api/history.' : '✗ Record NOT found in GET /api/history.');

  // 4. Confirm the ledger validates
  const ledgerRes = await fetch(`${BASE_URL}/api/ledger`);
  const ledgerBody = await ledgerRes.json();
  console.log(
    ledgerBody.valid
      ? `✓ GET /api/ledger reports chain valid (${ledgerBody.length} blocks).`
      : `✗ GET /api/ledger reports TAMPERED chain at block #${ledgerBody.brokenAtIndex}.`
  );

  console.log('\nAll backend checks passed. If the dashboard still shows nothing when');
  console.log('you record/upload in the browser, the problem is client-side -- see the');
  console.log('browser DevTools Console + Network tab checks in the deployment guide.');
}

main().catch((err) => {
  console.error('✗ Script failed:', err.message);
  console.error('  Is `npm run dev` running in another terminal, and reachable at', BASE_URL, '?');
  process.exit(1);
});
