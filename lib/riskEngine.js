// Real-Time Risk Scoring Engine
// ------------------------------
// Combines the ML spoof classifier's probability with transparent rule-based
// acoustic heuristics and call context (transaction value, known-contact
// status) into a single 0-100 impersonation risk score, a risk tier, and
// concrete recommended actions -- mirroring the "Real-Time Risk Scoring
// Engine" + "Alerting and User Interaction Layer" components of the brief.

const RISK_LEVELS = [
  { max: 30, level: 'LOW', color: 'safe', label: 'Likely genuine' },
  { max: 60, level: 'MEDIUM', color: 'warn', label: 'Some anomalies detected' },
  { max: 80, level: 'HIGH', color: 'danger', label: 'Likely synthetic / cloned voice' },
  { max: 101, level: 'CRITICAL', color: 'danger', label: 'High-confidence impersonation' },
];

function classifyRiskLevel(score) {
  return RISK_LEVELS.find((r) => score < r.max);
}

/**
 * Rule-based acoustic red flags, each independently explainable to an
 * analyst (this is what lets the dashboard show *why* a call was flagged,
 * not just a black-box number).
 */
function evaluateHeuristics(featureMap) {
  const flags = [];

  if (featureMap.jitter < 0.015) {
    flags.push({ id: 'low_jitter', severity: 'high', message: 'Pitch period is unnaturally stable (very low jitter) — consistent with neural TTS.' });
  }
  if (featureMap.shimmer < 0.02) {
    flags.push({ id: 'low_shimmer', severity: 'medium', message: 'Amplitude shows little natural micro-variation (low shimmer).' });
  }
  if (featureMap.pitchStd < 6 && featureMap.pitchVoicedRatio > 0.4) {
    flags.push({ id: 'flat_prosody', severity: 'medium', message: 'Pitch contour is unusually flat for conversational speech (flat prosody).' });
  }
  if (featureMap.spectralFlatnessMean > 0.35) {
    flags.push({ id: 'spectral_flatness', severity: 'medium', message: 'Spectrum is abnormally flat/noise-like, a common vocoder artifact.' });
  }
  if (featureMap.hfEnergyRatio > 0.28) {
    flags.push({ id: 'hf_artifact', severity: 'low', message: 'Elevated high-frequency energy suggestive of synthesis/upsampling artifacts.' });
  }
  if (featureMap.silenceRatio < 0.03) {
    flags.push({ id: 'no_breaths', severity: 'low', message: 'Almost no silence/breath gaps detected across the clip.' });
  }

  return flags;
}

/**
 * @param {number} spoofProbability - classifier output in [0,1]
 * @param {object} featureMap - named features from lib/audioFeatures.js
 * @param {object} context - optional call context:
 *   { knownContact: boolean, highValueTransaction: boolean, callChannel: 'voip'|'pstn'|'collab', claimedIdentity: string }
 */
function scoreRisk(spoofProbability, featureMap, context = {}) {
  const heuristicFlags = evaluateHeuristics(featureMap);

  // Rule-based sub-score: each flag contributes weighted points, capped.
  const severityWeight = { high: 18, medium: 10, low: 5 };
  let heuristicPoints = heuristicFlags.reduce((sum, f) => sum + severityWeight[f.severity], 0);
  heuristicPoints = Math.min(heuristicPoints, 60);

  // Blend: classifier probability dominates (65%), heuristics corroborate (35%).
  let baseScore = 0.65 * (spoofProbability * 100) + 0.35 * heuristicPoints;

  // Contextual enrichment / escalation multipliers.
  const contextAdjustments = [];
  if (context.highValueTransaction) {
    baseScore *= 1.15;
    contextAdjustments.push('High-value transaction context: +15% escalation.');
  }
  if (context.knownContact === false) {
    baseScore *= 1.1;
    contextAdjustments.push('Caller not in verified contact list: +10% escalation.');
  }
  if (context.callChannel === 'voip') {
    baseScore *= 1.05;
    contextAdjustments.push('VoIP origin (higher spoofing prevalence): +5% escalation.');
  }

  const score = Math.max(0, Math.min(100, Math.round(baseScore)));
  const tier = classifyRiskLevel(score);

  const recommendations = buildRecommendations(tier.level, context);

  return {
    score,
    level: tier.level,
    color: tier.color,
    summary: tier.label,
    spoofProbability,
    heuristicPoints,
    heuristicFlags,
    contextAdjustments,
    recommendations,
  };
}

function buildRecommendations(level, context) {
  const base = [];
  if (level === 'LOW') {
    base.push('No action required. Continue standard call handling.');
  } else if (level === 'MEDIUM') {
    base.push('Proceed with caution.', 'Ask an out-of-band knowledge question the caller could not have scripted.');
  } else if (level === 'HIGH') {
    base.push(
      'Pause any sensitive request (fund transfer, credential/OTP disclosure).',
      'Trigger secondary verification: call back on a known-good number.',
      'Notify the security/fraud desk of this session.'
    );
  } else {
    base.push(
      'Block or hold the transaction immediately.',
      'Escalate to a supervisor / fraud response team now.',
      'Require multi-factor, in-person, or video verification before proceeding.',
      'Log and retain (feature-only) evidence per the privacy & compliance policy.'
    );
  }
  if (context.highValueTransaction && (level === 'HIGH' || level === 'CRITICAL')) {
    base.push('Freeze the pending high-value transaction pending manual review.');
  }
  return base;
}

module.exports = { scoreRisk, evaluateHeuristics, classifyRiskLevel, RISK_LEVELS };
