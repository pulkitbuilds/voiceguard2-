import { NextResponse } from 'next/server';
import { execFile } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import crypto from 'crypto';

const execFileAsync = promisify(execFile);

export const runtime = 'nodejs';

export async function POST(request) {
  let inputPath = null;
  let outputPath = null;

  try {
    const formData = await request.formData();
    const file = formData.get('file');

    if (!file) {
      return NextResponse.json(
        { ok: false, error: 'No audio file provided.' },
        { status: 400 }
      );
    }

    const buffer = Buffer.from(await file.arrayBuffer());

    const id = crypto.randomUUID();
    inputPath = path.join(os.tmpdir(), `voiceguard-${id}-input`);
    outputPath = path.join(os.tmpdir(), `voiceguard-${id}-output.wav`);

    await fs.writeFile(inputPath, buffer);

    console.log(
      `[VoiceGuard] Converting ${file.name || 'audio'} (${buffer.length} bytes)`
    );

    await execFileAsync(
      'ffmpeg',
      [
        '-y',
        '-i', inputPath,

        // VoiceGuard target format
        '-ac', '1',
        '-ar', '16000',
        '-sample_fmt', 's16',

        '-f', 'wav',
        outputPath,
      ],
      {
        windowsHide: true,
        maxBuffer: 10 * 1024 * 1024,
      }
    );

    const wavBuffer = await fs.readFile(outputPath);

    console.log(
      `[VoiceGuard] Conversion successful: ${wavBuffer.length} bytes`
    );

    return new NextResponse(wavBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'audio/wav',
        'Content-Length': String(wavBuffer.length),
        'Cache-Control': 'no-store',
      },
    });
  } catch (error) {
    console.error('[VoiceGuard] FFmpeg conversion failed:', error);

    return NextResponse.json(
      {
        ok: false,
        error: error.message || 'Audio conversion failed.',
      },
      { status: 500 }
    );
  } finally {
    if (inputPath) {
      await fs.unlink(inputPath).catch(() => {});
    }

    if (outputPath) {
      await fs.unlink(outputPath).catch(() => {});
    }
  }
}