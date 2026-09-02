// DEMO-ONLY in-memory store.
// History + tamper-evident ledger for the local/SIH demo.
//
// NOTE:
// Vercel serverless functions are stateless/multi-instance, so this is NOT
// persistent production storage. Use a database/KV store for production.

const crypto = require('crypto');

const MAX_HISTORY = 100;

let history = globalThis.__voiceGuardHistory || [];
let ledger = globalThis.__voiceGuardLedger || [];

globalThis.__voiceGuardHistory = history;
globalThis.__voiceGuardLedger = ledger;


// --------------------------------------------------
// CALL HISTORY
// --------------------------------------------------

function pushCall(record) {
  history.unshift(record);

  if (history.length > MAX_HISTORY) {
    history.length = MAX_HISTORY;
  }

  // Add the call to the tamper-evident ledger.
  const previousBlock = ledger.length > 0
    ? ledger[ledger.length - 1]
    : null;

  const blockIndex = ledger.length;

  const blockData = {
    blockIndex,
    timestamp: new Date().toISOString(),
    recordId: record.id,
    riskScore: record.risk?.score ?? null,
    riskLevel: record.risk?.level ?? null,
    previousHash: previousBlock?.blockHash || 'GENESIS',
  };

  const blockHash = crypto
    .createHash('sha256')
    .update(JSON.stringify(blockData))
    .digest('hex');

  const block = {
    ...blockData,
    blockHash,
  };

  ledger.push(block);

  // Attach ledger information to the record returned by /api/analyze.
  record.ledger = {
    blockIndex,
    blockHash,
    previousHash: blockData.previousHash,
  };

  return record;
}


function listCalls(limit = 25) {
  return history.slice(0, limit);
}


// --------------------------------------------------
// LEDGER
// --------------------------------------------------

function getLedger() {
  return ledger;
}


function validateLedger() {
  for (let i = 0; i < ledger.length; i++) {
    const block = ledger[i];

    const expectedPreviousHash =
      i === 0
        ? 'GENESIS'
        : ledger[i - 1].blockHash;

    if (block.previousHash !== expectedPreviousHash) {
      return {
        valid: false,
        brokenAtIndex: i,
      };
    }

    const blockData = {
      blockIndex: block.blockIndex,
      timestamp: block.timestamp,
      recordId: block.recordId,
      riskScore: block.riskScore,
      riskLevel: block.riskLevel,
      previousHash: block.previousHash,
    };

    const expectedHash = crypto
      .createHash('sha256')
      .update(JSON.stringify(blockData))
      .digest('hex');

    if (block.blockHash !== expectedHash) {
      return {
        valid: false,
        brokenAtIndex: i,
      };
    }
  }

  return {
    valid: true,
    brokenAtIndex: null,
  };
}


module.exports = {
  pushCall,
  listCalls,
  getLedger,
  validateLedger,
};