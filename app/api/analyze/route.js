import { NextResponse } from 'next/server';
import { classify } from '../../../lib/classifier';
import { scoreRisk } from '../../../lib/riskEngine';
import { FEATURE_NAMES } from '../../../lib/audioFeatures';
import { pushCall } from '../../../lib/store';

// Node runtime (not edge): keeps this simple and compatible with the JSON
// model-weights require() in lib/classifier.js. Cheap enough (~ms) that cold
// starts are not a concern for a real-time call-analysis endpoint.
export const runtime = 'nodejs';

export async function POST(req) {
  let body;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body.' }, { status: 400 });
  }

  const { featureVector, featureMap, meta, context, callerLabel } = body || {};

  if (!Array.isArray(featureVector) || featureVector.length !== FEATURE_NAMES.length) {
    return NextResponse.json(
      {
        error: `featureVector must be an array of length ${FEATURE_NAMES.length} (got ${featureVector?.length ?? 'undefined'}). Extract it client-side with lib/audioFeatures.js before calling this endpoint.`,
      },
      { status: 400 }
    );
  }
  if (!featureMap || typeof featureMap !== 'object') {
    return NextResponse.json({ error: 'featureMap object is required alongside featureVector.' }, { status: 400 });
  }

  let classifierResult;
  try {
    classifierResult = classify(featureVector);
  } catch (err) {
    return NextResponse.json({ error: `Classifier error: ${err.message}` }, { status: 500 });
  }

  const risk = scoreRisk(classifierResult.spoofProbability, featureMap, context || {});

  const record = {
    id: `call_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    timestamp: new Date().toISOString(),
    callerLabel: callerLabel || 'Unknown caller',
    durationSec: meta?.durationSec ?? null,
    context: context || {},
    risk,
    // Feature-only (no raw audio) is retained per the Privacy & Compliance
    // module in the brief -- safe to store/echo since it cannot reconstruct speech.
    featureMap,
  };

  pushCall(record);

  return NextResponse.json({ ok: true, record });
}
