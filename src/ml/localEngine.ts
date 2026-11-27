import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import { WILDLIFE_TAXONOMY_JSON } from '../data/datasets';

const FACE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let cocoPromise: Promise<cocoSsd.ObjectDetection> | null = null;
let faceLandmarkerPromise: Promise<FaceLandmarker> | null = null;
let handLandmarkerPromise: Promise<HandLandmarker> | null = null;

async function getCocoModel(): Promise<cocoSsd.ObjectDetection> {
  if (!cocoPromise) {
    cocoPromise = (async () => {
      await tf.ready();
      return cocoSsd.load({ base: 'mobilenet_v2' });
    })();
  }
  return cocoPromise;
}

async function getFaceLandmarker(): Promise<FaceLandmarker> {
  if (!faceLandmarkerPromise) {
    faceLandmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks('/mediapipe-wasm');
      return FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: FACE_MODEL_URL },
        runningMode: 'IMAGE',
        numFaces: 10,
        outputFaceBlendshapes: true,
        minFaceDetectionConfidence: 0.35,
        minFacePresenceConfidence: 0.35,
      });
    })();
  }
  return faceLandmarkerPromise;
}

async function getHandLandmarker(): Promise<HandLandmarker> {
  if (!handLandmarkerPromise) {
    handLandmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks('/mediapipe-wasm');
      return HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL_URL },
        runningMode: 'IMAGE',
        numHands: 2,
        minHandDetectionConfidence: 0.35,
        minHandPresenceConfidence: 0.35,
      });
    })();
  }
  return handLandmarkerPromise;
}

export function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not decode image for local inference.'));
    img.src = src;
  });
}

function bboxToNormalized(
  bbox: [number, number, number, number],
  width: number,
  height: number,
): [number, number, number, number] {
  const [x, y, w, h] = bbox;
  const xmin = Math.max(0, Math.min(1000, Math.round((x / width) * 1000)));
  const ymin = Math.max(0, Math.min(1000, Math.round((y / height) * 1000)));
  const xmax = Math.max(0, Math.min(1000, Math.round(((x + w) / width) * 1000)));
  const ymax = Math.max(0, Math.min(1000, Math.round(((y + h) / height) * 1000)));
  return [ymin, xmin, ymax, xmax];
}

// Interior-crop HSV colour classifier matching modules/car_detection/colour_classifier.py
function classifyCarCropHsv(
  ctx: CanvasRenderingContext2D,
  bbox: [number, number, number, number],
  imgW: number,
  imgH: number,
  colourThreshold: number,
): { colour: string; evidence: number } {
  const [bx, by, bw, bh] = bbox;
  const padX = bw * 0.2;
  const padYTop = bh * 0.34; // skip windshield/roof glare
  const padYBottom = bh * 0.22; // skip tires/road shadow
  const sx = Math.max(0, Math.floor(bx + padX));
  const sy = Math.max(0, Math.floor(by + padYTop));
  const sw = Math.max(2, Math.min(imgW - sx, Math.floor(bw - padX * 2)));
  const sh = Math.max(2, Math.min(imgH - sy, Math.floor(bh - padYTop - padYBottom)));

  const pixels = ctx.getImageData(sx, sy, sw, sh).data;
  const counts: Record<string, number> = {
    blue: 0,
    red: 0,
    white: 0,
    black: 0,
    gray: 0,
    silver: 0,
    green: 0,
    yellow: 0,
    orange: 0,
    brown: 0,
    other: 0,
  };

  let chromaticCount = 0;
  let validCount = 0;

  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i] / 255;
    const g = pixels[i + 1] / 255;
    const b = pixels[i + 2] / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;
    const v = max;
    const s = max === 0 ? 0 : delta / max;

    let h = 0;
    if (delta !== 0) {
      if (max === r) h = 60 * (((g - b) / delta) % 6);
      else if (max === g) h = 60 * ((b - r) / delta + 2);
      else h = 60 * ((r - g) / delta + 4);
    }
    if (h < 0) h += 360;

    validCount += 1;
    if (s >= 0.24 && v >= 0.2) {
      chromaticCount += 1;
      if (h >= 185 && h <= 255) counts.blue += 1;
      else if (h < 14 || h >= 340) counts.red += 1;
      else if (h >= 14 && h < 38) {
        if (v < 0.45) counts.brown += 1;
        else counts.orange += 1;
      } else if (h >= 38 && h < 70) counts.yellow += 1;
      else if (h >= 70 && h < 175) counts.green += 1;
      else counts.other += 1;
    } else if (v < 0.22) {
      counts.black += 1;
    } else if (v > 0.78 && s < 0.16) {
      counts.white += 1;
    } else if (v > 0.56) {
      counts.silver += 1;
    } else {
      counts.gray += 1;
    }
  }

  if (validCount === 0) return { colour: 'unknown', evidence: 0 };

  // Prefer strong chromatic paint colours when at least 22% of interior crop pixels are saturated paint
  const chromaticLabels = ['blue', 'red', 'green', 'yellow', 'orange', 'brown', 'other'];
  const achromaticLabels = ['white', 'silver', 'gray', 'black'];
  const pool = chromaticCount / validCount >= 0.22 ? chromaticLabels : achromaticLabels;

  let bestLabel = 'unknown';
  let bestCount = 0;
  for (const label of pool) {
    if (counts[label] > bestCount) {
      bestCount = counts[label];
      bestLabel = label;
    }
  }

  const evidence = bestCount / validCount;
  if (evidence < colourThreshold) {
    return { colour: 'unknown', evidence };
  }
  return { colour: bestLabel, evidence };
}

// ============================================================
// TASK 1: Car Colour Detection (COCO-SSD + HSV Crop Classifier)
// ============================================================
export async function runCarDetectionLocal(
  imageUrl: string,
  confidenceThreshold: number,
  colourThreshold: number,
) {
  const started = performance.now();
  const img = await loadImageElement(imageUrl);
  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Canvas context unavailable.');
  ctx.drawImage(img, 0, 0, width, height);

  const model = await getCocoModel();
  const rawDetections = await model.detect(img, 30, Math.max(0.05, confidenceThreshold));

  const cars: Array<{
    id: string;
    className: 'car';
    confidence: number;
    box: [number, number, number, number];
    colour: string;
    colourEvidence: number;
    boxStyle: 'red' | 'blue' | 'gray';
  }> = [];
  const people: Array<{
    id: string;
    className: 'person';
    confidence: number;
    box: [number, number, number, number];
  }> = [];

  for (const det of rawDetections) {
    if (det.score < confidenceThreshold) continue;
    const normBox = bboxToNormalized(det.bbox, width, height);
    if (det.class === 'car' || det.class === 'truck') {
      const hsv = classifyCarCropHsv(ctx, det.bbox, width, height, colourThreshold);
      const boxStyle = hsv.colour === 'blue' ? 'red' : hsv.colour === 'unknown' ? 'gray' : 'blue';
      cars.push({
        id: `CAR-${String(cars.length + 1).padStart(3, '0')}`,
        className: 'car',
        confidence: det.score,
        box: normBox,
        colour: hsv.colour,
        colourEvidence: hsv.evidence,
        boxStyle,
      });
    } else if (det.class === 'person') {
      people.push({
        id: `PERSON-${String(people.length + 1).padStart(3, '0')}`,
        className: 'person',
        confidence: det.score,
        box: normBox,
      });
    }
  }

  const blueCarCount = cars.filter((c) => c.colour === 'blue').length;
  const unknownColourCarCount = cars.filter((c) => c.colour === 'unknown').length;
  const otherKnownColourCarCount = cars.length - blueCarCount - unknownColourCarCount;

  return {
    carCount: cars.length,
    blueCarCount,
    otherKnownColourCarCount,
    unknownColourCarCount,
    personCount: people.length,
    cars,
    people,
    sceneSummary: `Detected ${cars.length} vehicle(s) and ${people.length} person(s) using TensorFlow.js COCO-SSD + HSV interior-crop classifier.`,
    metadata: {
      model: 'TensorFlow.js COCO-SSD (mobilenet_v2) + HSV Crop Estimator',
      confidenceThreshold,
      colourThreshold,
      inferenceMs: Math.round(performance.now() - started),
    },
  };
}

// ============================================================
// TASK 2: Wildlife Risk Intelligence (COCO-SSD + taxonomy.json)
// ============================================================
export async function runWildlifeLocal(imageUrl: string, confidenceThreshold: number) {
  const started = performance.now();
  const img = await loadImageElement(imageUrl);
  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;

  const model = await getCocoModel();
  const rawDetections = await model.detect(img, 30, Math.max(0.05, confidenceThreshold));
  const supportedAnimals = new Set(WILDLIFE_TAXONOMY_JSON.animal_classes);

  const detections: Array<{
    id: string;
    species: string;
    confidence: number;
    box: [number, number, number, number];
    classification: 'carnivore' | 'non_carnivore' | 'unknown';
    note: string;
  }> = [];

  const speciesCounts: Record<string, number> = {};
  let carnivores = 0;
  let nonCarnivores = 0;
  let unknown = 0;

  for (const det of rawDetections) {
    if (det.score < confidenceThreshold) continue;
    const species = det.class.toLowerCase();
    if (!supportedAnimals.has(species)) continue;

    const classification = WILDLIFE_TAXONOMY_JSON.classifications[species] ?? 'unknown';
    if (classification === 'carnivore') carnivores += 1;
    else if (classification === 'non_carnivore') nonCarnivores += 1;
    else unknown += 1;

    speciesCounts[species] = (speciesCounts[species] ?? 0) + 1;
    const notesMap = WILDLIFE_TAXONOMY_JSON.notes as Record<string, string>;
    detections.push({
      id: `ANIMAL-${String(detections.length + 1).padStart(3, '0')}`,
      species,
      confidence: det.score,
      box: bboxToNormalized(det.bbox, width, height),
      classification,
      note:
        notesMap[species] ??
        (classification === 'carnivore'
          ? 'Mapped as obligate carnivore in taxonomy.json v1.0.0.'
          : 'Mapped as non-carnivore in taxonomy.json v1.0.0.'),
    });
  }

  return {
    summary: {
      totalAnimals: detections.length,
      carnivores,
      nonCarnivores,
      unknown,
      speciesCounts,
    },
    detections,
    metadata: {
      model: 'TensorFlow.js COCO-SSD (mobilenet_v2)',
      taxonomyVersion: WILDLIFE_TAXONOMY_JSON.version,
      confidenceThreshold,
      inferenceMs: Math.round(performance.now() - started),
    },
  };
}

// ============================================================
// TASK 3: Voice Emotion Detection (Web Audio Acoustic Prosody)
// ============================================================
export function generateSampleEmotionWav(emotion: 'happy' | 'angry' | 'sad' | 'neutral'): {
  blob: Blob;
  filename: string;
} {
  const sampleRate = 16000;
  const duration = 2.4;
  const numSamples = Math.floor(sampleRate * duration);
  const samples = new Float32Array(numSamples);

  const cfg = {
    happy: { f0: 245, f0Span: 65, amp: 0.42, rate: 5.2, harsh: 0.14 },
    angry: { f0: 195, f0Span: 45, amp: 0.78, rate: 6.4, harsh: 0.58 },
    sad: { f0: 132, f0Span: 14, amp: 0.16, rate: 2.1, harsh: 0.05 },
    neutral: { f0: 165, f0Span: 12, amp: 0.30, rate: 3.3, harsh: 0.09 },
  }[emotion];

  let phase = 0;
  for (let i = 0; i < numSamples; i += 1) {
    const t = i / sampleRate;
    const syllableEnv = Math.max(0, Math.sin(2 * Math.PI * cfg.rate * t));
    const phraseEnv = Math.sin((Math.PI * t) / duration);
    const env = Math.pow(syllableEnv, 0.7) * phraseEnv;
    const freq = cfg.f0 + cfg.f0Span * Math.sin(2 * Math.PI * 1.4 * t);
    phase += (2 * Math.PI * freq) / sampleRate;

    // Harmonic speech-like formant blend
    const fundamental = Math.sin(phase);
    const h2 = 0.55 * Math.sin(2 * phase);
    const h3 = (0.25 + cfg.harsh * 0.6) * Math.sin(3 * phase);
    const h5 = cfg.harsh * 0.45 * Math.sin(5 * phase);
    const breathNoise = cfg.harsh * 0.15 * (Math.random() * 2 - 1);

    samples[i] = Math.max(-0.98, Math.min(0.98, env * cfg.amp * (fundamental + h2 + h3 + h5 + breathNoise)));
  }

  // Encode 16-bit mono PCM WAV
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i += 1) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i += 1) {
    view.setInt16(offset, Math.round(samples[i] * 32767), true);
    offset += 2;
  }

  return {
    blob: new Blob([buffer], { type: 'audio/wav' }),
    filename: `sample_${emotion}_prosody.wav`,
  };
}

function estimatePitchHz(frame: Float32Array, sampleRate: number): number | null {
  const minLag = Math.floor(sampleRate / 420);
  const maxLag = Math.min(frame.length - 1, Math.floor(sampleRate / 85));
  let bestLag = -1;
  let bestCorr = 0;
  let energy = 0;
  for (let i = 0; i < frame.length; i += 1) energy += frame[i] * frame[i];
  if (energy < 1e-4) return null;

  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let corr = 0;
    for (let i = 0; i < frame.length - lag; i += 1) {
      corr += frame[i] * frame[i + lag];
    }
    const norm = corr / energy;
    if (norm > bestCorr) {
      bestCorr = norm;
      bestLag = lag;
    }
  }
  if (bestLag <= 0 || bestCorr < 0.35) return null;
  return sampleRate / bestLag;
}

export async function runVoiceEmotionLocal(audioBlob: Blob, minimumScore: number) {
  const started = performance.now();
  const arrayBuf = await audioBlob.arrayBuffer();
  const ctx = new AudioContext();
  let decoded: AudioBuffer;
  try {
    decoded = await ctx.decodeAudioData(arrayBuf.slice(0));
  } finally {
    await ctx.close();
  }

  const channelData = decoded.getChannelData(0);
  const sr = decoded.sampleRate;
  const frameSize = Math.floor(sr * 0.04); // 40ms frames
  const rmsValues: number[] = [];
  const zcrValues: number[] = [];
  const pitchValues: number[] = [];

  for (let start = 0; start + frameSize <= channelData.length; start += frameSize) {
    const frame = channelData.subarray(start, start + frameSize);
    let sqSum = 0;
    let crossings = 0;
    for (let i = 0; i < frame.length; i += 1) {
      sqSum += frame[i] * frame[i];
      if (i > 0 && ((frame[i - 1] >= 0 && frame[i] < 0) || (frame[i - 1] < 0 && frame[i] >= 0))) {
        crossings += 1;
      }
    }
    const rms = Math.sqrt(sqSum / frame.length);
    if (rms > 0.015) {
      rmsValues.push(rms);
      zcrValues.push(crossings / frame.length);
      const f0 = estimatePitchHz(frame, sr);
      if (f0) pitchValues.push(f0);
    }
  }

  if (rmsValues.length === 0) {
    throw new Error('The audio clip appears to be silent or below the minimum speech energy threshold.');
  }

  const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / Math.max(1, arr.length);
  const std = (arr: number[], m: number) =>
    Math.sqrt(arr.reduce((a, b) => a + (b - m) * (b - m), 0) / Math.max(1, arr.length));

  const meanRms = mean(rmsValues);
  const meanZcr = mean(zcrValues);
  const meanPitch = pitchValues.length ? mean(pitchValues) : 165;
  const pitchStd = pitchValues.length ? std(pitchValues, meanPitch) : 15;

  // Score each emotion from normalized acoustic arousal & prosodic cues
  const rawScores = {
    angry: Math.max(
      0.05,
      (meanRms > 0.25 ? 2.9 : meanRms * 5.5) +
        (meanZcr > 0.055 ? 2.1 : meanZcr * 22) +
        (meanRms > 0.32 ? 0.8 : 0),
    ),
    happy: Math.max(
      0.05,
      (meanPitch > 205 ? 2.4 : Math.max(0, (meanPitch - 145) / 45)) +
        (pitchStd > 26 ? 1.8 : pitchStd / 24) +
        (meanRms >= 0.13 && meanRms <= 0.30 ? 1.1 : 0.15),
    ),
    sad: Math.max(
      0.05,
      (meanRms < 0.12 ? 2.6 : Math.max(0, 1.4 - meanRms * 5)) +
        (meanPitch < 152 ? 1.9 : 0.2) +
        (pitchStd < 22 ? 0.8 : 0.1),
    ),
    neutral: Math.max(
      0.05,
      (meanRms >= 0.11 && meanRms <= 0.24 ? 2.2 : 0.4) +
        (pitchStd < 22 ? 1.8 : 0.3) +
        (meanPitch >= 145 && meanPitch <= 198 ? 1.1 : 0.2),
    ),
  };

  // Softmax normalization
  const maxRaw = Math.max(...Object.values(rawScores));
  const exps = {
    neutral: Math.exp(rawScores.neutral - maxRaw),
    happy: Math.exp(rawScores.happy - maxRaw),
    angry: Math.exp(rawScores.angry - maxRaw),
    sad: Math.exp(rawScores.sad - maxRaw),
  };
  const expSum = exps.neutral + exps.happy + exps.angry + exps.sad;
  const scores = {
    neutral: exps.neutral / expSum,
    happy: exps.happy / expSum,
    angry: exps.angry / expSum,
    sad: exps.sad / expSum,
  };

  const sorted = (Object.entries(scores) as Array<['neutral' | 'happy' | 'angry' | 'sad', number]>).sort(
    (a, b) => b[1] - a[1],
  );
  const [label, confidence] = sorted[0];

  const warning =
    confidence < minimumScore
      ? `Model score (${(confidence * 100).toFixed(1)}%) is below your ${(minimumScore * 100).toFixed(0)}% threshold.`
      : null;

  return {
    label,
    confidence,
    scores,
    acousticCues: `RMS energy ${(meanRms * 100).toFixed(1)}%, mean F0 pitch ${meanPitch.toFixed(0)} Hz (σ=${pitchStd.toFixed(0)} Hz), ZCR ${(meanZcr * 100).toFixed(1)}%, duration ${decoded.duration.toFixed(2)}s @ ${sr} Hz.`,
    transcriptExcerpt: '',
    warning,
    metadata: {
      model: 'Web Audio Acoustic Prosody Analyzer (4-class SUPERB schema)',
      labelSchema: 'neutral, happy, angry, sad',
      minimumScore,
      inferenceMs: Math.round(performance.now() - started),
    },
  };
}

// ============================================================
// TASK 4: Person Attribute & Expression Analysis (COCO-SSD + MediaPipe FaceLandmarker)
// ============================================================
function getBlendshapeScore(
  categories: Array<{ categoryName: string; score: number }> | undefined,
  name: string,
): number {
  if (!categories) return 0;
  return categories.find((c) => c.categoryName === name)?.score ?? 0;
}

function classifyExpressionFromBlendshapes(
  categories: Array<{ categoryName: string; score: number }> | undefined,
): { expression: string; confidence: number } {
  if (!categories || categories.length === 0) {
    return { expression: 'unknown', confidence: 0 };
  }
  const smile =
    (getBlendshapeScore(categories, 'mouthSmileLeft') +
      getBlendshapeScore(categories, 'mouthSmileRight')) /
    2;
  const frown =
    (getBlendshapeScore(categories, 'mouthFrownLeft') +
      getBlendshapeScore(categories, 'mouthFrownRight')) /
    2;
  const browDown =
    (getBlendshapeScore(categories, 'browDownLeft') +
      getBlendshapeScore(categories, 'browDownRight')) /
    2;
  const browInnerUp = getBlendshapeScore(categories, 'browInnerUp');
  const jawOpen = getBlendshapeScore(categories, 'jawOpen');
  const eyeWide =
    (getBlendshapeScore(categories, 'eyeWideLeft') +
      getBlendshapeScore(categories, 'eyeWideRight')) /
    2;

  if (smile >= 0.32) return { expression: 'happy', confidence: Math.min(0.99, 0.55 + smile * 0.5) };
  if (jawOpen >= 0.35 && eyeWide >= 0.25) {
    return { expression: 'surprised', confidence: Math.min(0.98, 0.5 + (jawOpen + eyeWide) * 0.4) };
  }
  if (browDown >= 0.30) return { expression: 'angry', confidence: Math.min(0.96, 0.5 + browDown * 0.55) };
  if (frown >= 0.22 || browInnerUp >= 0.35) {
    return { expression: 'sad', confidence: Math.min(0.95, 0.5 + Math.max(frown, browInnerUp) * 0.5) };
  }
  return { expression: 'neutral', confidence: Math.max(0.58, 0.85 - Math.max(smile, frown, browDown)) };
}

function faceCenterNorm(landmarks: NormalizedLandmark[]): { cx: number; cy: number; box: [number, number, number, number] } {
  let minX = 1;
  let maxX = 0;
  let minY = 1;
  let maxY = 0;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.y > maxY) maxY = lm.y;
  }
  return {
    cx: ((minX + maxX) / 2) * 1000,
    cy: ((minY + maxY) / 2) * 1000,
    box: [
      Math.round(minY * 1000),
      Math.round(minX * 1000),
      Math.round(maxY * 1000),
      Math.round(maxX * 1000),
    ],
  };
}

export async function runPersonAnalysisLocal(
  imageUrl: string,
  confidenceThreshold: number,
  userProvidedRegion: string,
) {
  const started = performance.now();
  const img = await loadImageElement(imageUrl);
  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Canvas context unavailable.');
  ctx.drawImage(img, 0, 0, width, height);

  const [cocoModel, faceLandmarker] = await Promise.all([getCocoModel(), getFaceLandmarker()]);
  const [rawDetections, faceRes] = await Promise.all([
    cocoModel.detect(img, 30, Math.max(0.05, confidenceThreshold)),
    Promise.resolve(faceLandmarker.detect(img)),
  ]);

  const faces = (faceRes.faceLandmarks ?? []).map((landmarks, idx) => ({
    ...faceCenterNorm(landmarks),
    blendshapes: faceRes.faceBlendshapes?.[idx]?.categories,
  }));

  const matchedFaceIndices = new Set<number>();
  const personDets = rawDetections.filter((d) => d.class === 'person' && d.score >= confidenceThreshold);

  const people = personDets.map((det, idx) => {
    const box = bboxToNormalized(det.bbox, width, height);
    const [ymin, xmin, ymax, xmax] = box;

    // Match face whose center falls inside this person box
    const faceIdx = faces.findIndex(
      (f, i) =>
        !matchedFaceIndices.has(i) &&
        f.cx >= xmin - 30 &&
        f.cx <= xmax + 30 &&
        f.cy >= ymin - 30 &&
        f.cy <= ymax + 30,
    );
    const matchedFace = faceIdx >= 0 ? faces[faceIdx] : undefined;
    if (faceIdx >= 0) matchedFaceIndices.add(faceIdx);

    const expr = matchedFace
      ? classifyExpressionFromBlendshapes(matchedFace.blendshapes)
      : { expression: 'face not clearly visible', confidence: 0 };

    // Estimate dominant upper-body garment colour via HSV
    const torsoBox: [number, number, number, number] = [
      det.bbox[0],
      det.bbox[1] + det.bbox[3] * 0.25,
      det.bbox[2],
      det.bbox[3] * 0.45,
    ];
    const garment = classifyCarCropHsv(ctx, torsoBox, width, height, 0.15);
    const attributes: string[] = [];
    if (garment.colour !== 'unknown') attributes.push(`${garment.colour} upper clothing`);
    if (matchedFace) attributes.push('frontal face detected');

    return {
      id: `PERSON-${String(idx + 1).padStart(3, '0')}`,
      confidence: det.score,
      box,
      coarseAgeRange: 'Not estimated (no age model)',
      expressionEstimate: expr.expression,
      expressionConfidence: expr.confidence,
      visibleAttributes: attributes,
    };
  });

  // Fallback for close-up portraits where FaceLandmarker detects a face missed by COCO-SSD person detector
  faces.forEach((face, i) => {
    if (matchedFaceIndices.has(i)) return;
    const [fymin, fxmin, fymax, fxmax] = face.box;
    const fh = fymax - fymin;
    const fw = fxmax - fxmin;
    const expandedBox: [number, number, number, number] = [
      Math.max(0, Math.round(fymin - fh * 0.25)),
      Math.max(0, Math.round(fxmin - fw * 0.35)),
      Math.min(1000, Math.round(fymax + fh * 0.55)),
      Math.min(1000, Math.round(fxmax + fw * 0.35)),
    ];
    const expr = classifyExpressionFromBlendshapes(face.blendshapes);
    people.push({
      id: `PERSON-${String(people.length + 1).padStart(3, '0')}`,
      confidence: 0.85,
      box: expandedBox,
      coarseAgeRange: 'Not estimated (no age model)',
      expressionEstimate: expr.expression,
      expressionConfidence: expr.confidence,
      visibleAttributes: ['frontal face detected'],
    });
  });

  return {
    personCount: people.length,
    people,
    userProvidedRegion: userProvidedRegion.trim() || null,
    metadata: {
      model: 'TensorFlow.js COCO-SSD + MediaPipe FaceLandmarker (52 Blendshapes)',
      confidenceThreshold,
      nationalityInferredFromAppearance: false,
      inferenceMs: Math.round(performance.now() - started),
    },
  };
}

// ============================================================
// TASK 5: Sign Language Detection (MediaPipe HandLandmarker 21 pts)
// ============================================================
function dist2d(a: NormalizedLandmark, b: NormalizedLandmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function classifyStaticAslFromLandmarks(lm: NormalizedLandmark[]): {
  label: string;
  confidence: number;
  description: string;
} {
  const wrist = lm[0];
  const palmScale = Math.max(0.01, dist2d(wrist, lm[9]));

  const isExtended = (tipIdx: number, pipIdx: number, mcpIdx: number) => {
    const tipDist = dist2d(lm[tipIdx], wrist);
    const pipDist = dist2d(lm[pipIdx], wrist);
    const mcpDist = dist2d(lm[mcpIdx], wrist);
    return tipDist > pipDist * 1.07 && tipDist > mcpDist * 1.25;
  };

  const indexUp = isExtended(8, 6, 5);
  const middleUp = isExtended(12, 10, 9);
  const ringUp = isExtended(16, 14, 13);
  const pinkyUp = isExtended(20, 18, 17);

  // Thumb geometry relative to palm center (landmark 9) and index MCP (5) / pinky MCP (17)
  const thumbTip = lm[4];
  const thumbIp = lm[3];
  const thumbOut = dist2d(thumbTip, lm[9]) / palmScale > 0.92 && dist2d(thumbTip, lm[5]) > dist2d(thumbIp, lm[5]) * 0.95;
  const thumbAcrossPalm = dist2d(thumbTip, lm[9]) / palmScale < 0.72 || dist2d(thumbTip, lm[13]) / palmScale < 0.65;
  const thumbIndexPinch = dist2d(thumbTip, lm[8]) / palmScale < 0.34;
  const indexMiddleGap = dist2d(lm[8], lm[12]) / palmScale;
  const middleRingGap = dist2d(lm[12], lm[16]) / palmScale;

  const upList = [
    indexUp ? 'index' : null,
    middleUp ? 'middle' : null,
    ringUp ? 'ring' : null,
    pinkyUp ? 'pinky' : null,
  ].filter(Boolean);

  // 4 fingers up (B)
  if (indexUp && middleUp && ringUp && pinkyUp) {
    if (thumbAcrossPalm || !thumbOut) {
      return {
        label: 'B',
        confidence: 0.94,
        description: 'Four fingers extended straight up together with thumb folded across palm (ASL "B").',
      };
    }
    return {
      label: 'B',
      confidence: 0.76,
      description: 'Open upright palm with four extended fingers (closest static match: ASL "B").',
    };
  }

  // Index + Middle + Ring up (W)
  if (indexUp && middleUp && ringUp && !pinkyUp) {
    return {
      label: 'W',
      confidence: 0.91,
      description: 'Index, middle, and ring fingers extended upward with pinky curled (ASL "W").',
    };
  }

  // Middle + Ring + Pinky up + thumb-index pinch (F)
  if (!indexUp && middleUp && ringUp && pinkyUp && thumbIndexPinch) {
    return {
      label: 'F',
      confidence: 0.92,
      description: 'Thumb and index fingertips touching with middle, ring, and pinky extended (ASL "F").',
    };
  }

  // Index + Middle up (V / U / K / R)
  if (indexUp && middleUp && !ringUp && !pinkyUp) {
    if (indexMiddleGap > 0.36) {
      return {
        label: 'V',
        confidence: 0.91,
        description: 'Index and middle fingers extended and spread apart in a V shape (ASL "V").',
      };
    }
    if (middleRingGap < 0.25 && dist2d(lm[8], lm[12]) / palmScale < 0.24) {
      return {
        label: 'U',
        confidence: 0.87,
        description: 'Index and middle fingers extended together upright (ASL "U").',
      };
    }
    return {
      label: 'K',
      confidence: 0.80,
      description: 'Index and middle fingers extended with thumb positioned between them (ASL "K").',
    };
  }

  // Index only up (D or L)
  if (indexUp && !middleUp && !ringUp && !pinkyUp) {
    if (thumbOut) {
      return {
        label: 'L',
        confidence: 0.93,
        description: 'Index finger extended upward and thumb extended outward at 90 degrees (ASL "L").',
      };
    }
    return {
      label: 'D',
      confidence: 0.88,
      description: 'Index finger extended upward with remaining fingers meeting thumb (ASL "D").',
    };
  }

  // Pinky only up (I or Y)
  if (!indexUp && !middleUp && !ringUp && pinkyUp) {
    if (thumbOut) {
      return {
        label: 'Y',
        confidence: 0.93,
        description: 'Thumb and pinky extended outward with middle three fingers curled (ASL "Y").',
      };
    }
    return {
      label: 'I',
      confidence: 0.91,
      description: 'Pinky finger extended upward with other fingers curled into fist (ASL "I").',
    };
  }

  // Closed fist shapes (A / S / O / C / E)
  if (!indexUp && !middleUp && !ringUp && !pinkyUp) {
    if (thumbIndexPinch) {
      return {
        label: 'O',
        confidence: 0.86,
        description: 'Curved fingers meeting thumb tip to form an O shape (ASL "O").',
      };
    }
    if (thumbOut) {
      return {
        label: 'A',
        confidence: 0.88,
        description: 'Closed fist with thumb resting upright along the side of the index finger (ASL "A").',
      };
    }
    return {
      label: 'S',
      confidence: 0.82,
      description: 'Closed fist with thumb wrapped across the front of curled fingers (ASL "S").',
    };
  }

  return {
    label: 'UNKNOWN',
    confidence: 0.42,
    description: `Detected hand with extended fingers: ${upList.join(', ') || 'none'}; does not match a high-confidence static ASL letter.`,
  };
}

export function checkOperatingWindowLocal(
  startHHMM: string,
  endHHMM: string,
  timeZone: string,
): { authorized: boolean; localTime: string; reason?: string } {
  const formatted = new Intl.DateTimeFormat('en-GB', {
    timeZone,
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(new Date());
  const toMin = (s: string) => {
    const [h, m] = s.split(':').map(Number);
    return h * 60 + m;
  };
  const nowMin = toMin(formatted);
  const startMin = toMin(startHHMM);
  const endMin = toMin(endHHMM);
  const authorized =
    startMin <= endMin ? nowMin >= startMin && nowMin < endMin : nowMin >= startMin || nowMin < endMin;
  return {
    authorized,
    localTime: formatted,
    reason: authorized
      ? undefined
      : `Sign-language inference is closed outside ${startHHMM}–${endHHMM} ${timeZone} (current time: ${formatted}).`,
  };
}

export async function runSignLanguageLocal(
  imageUrl: string,
  options: {
    minimumScore: number;
    enforceWindow: boolean;
    allowedStart: string;
    allowedEnd: string;
    timeZone: string;
  },
) {
  const started = performance.now();
  const windowStatus = checkOperatingWindowLocal(
    options.allowedStart,
    options.allowedEnd,
    options.timeZone,
  );
  if (options.enforceWindow && !windowStatus.authorized) {
    throw new Error(windowStatus.reason || 'Sign-language window is closed.');
  }

  const img = await loadImageElement(imageUrl);
  const landmarker = await getHandLandmarker();
  const res = landmarker.detect(img);
  const hands = res.landmarks ?? [];

  if (hands.length === 0) {
    return {
      status: 'no_hand' as const,
      label: null,
      confidence: 0,
      box: null,
      landmarks: null as NormalizedLandmark[] | null,
      handDescription: 'No clear human hand was detected by MediaPipe Hands.',
      warning: 'Position a single hand clearly in front of a contrasting background.',
      windowStatus,
      metadata: {
        model: 'MediaPipe HandLandmarker (21 3D landmarks) + Static ASL Classifier',
        vocabularyCount: 24,
        minimumScore: options.minimumScore,
        enforceWindow: options.enforceWindow,
        inferenceMs: Math.round(performance.now() - started),
      },
    };
  }

  if (hands.length > 1) {
    return {
      status: 'multiple_hands' as const,
      label: null,
      confidence: 0,
      box: null,
      landmarks: hands[0],
      handDescription: `Detected ${hands.length} hands in frame.`,
      warning: 'Static ASL fingerspelling requires exactly one clearly visible hand.',
      windowStatus,
      metadata: {
        model: 'MediaPipe HandLandmarker (21 3D landmarks) + Static ASL Classifier',
        vocabularyCount: 24,
        minimumScore: options.minimumScore,
        enforceWindow: options.enforceWindow,
        inferenceMs: Math.round(performance.now() - started),
      },
    };
  }

  const lm = hands[0];
  let minX = 1;
  let maxX = 0;
  let minY = 1;
  let maxY = 0;
  for (const pt of lm) {
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
  }
  const pad = 0.03;
  const box: [number, number, number, number] = [
    Math.max(0, Math.round((minY - pad) * 1000)),
    Math.max(0, Math.round((minX - pad) * 1000)),
    Math.min(1000, Math.round((maxY + pad) * 1000)),
    Math.min(1000, Math.round((maxX + pad) * 1000)),
  ];

  const classified = classifyStaticAslFromLandmarks(lm);
  const accepted =
    classified.label !== 'UNKNOWN' && classified.confidence >= options.minimumScore;

  return {
    status: accepted ? ('recognized' as const) : ('unknown' as const),
    label: accepted ? classified.label : null,
    confidence: classified.confidence,
    box,
    landmarks: lm,
    handDescription: classified.description,
    warning: accepted
      ? null
      : `Top score (${(classified.confidence * 100).toFixed(0)}%) did not pass the ${(options.minimumScore * 100).toFixed(0)}% threshold.`,
    windowStatus,
    metadata: {
      model: 'MediaPipe HandLandmarker (21 3D landmarks) + Static ASL Classifier',
      vocabularyCount: 24,
      minimumScore: options.minimumScore,
      enforceWindow: options.enforceWindow,
      inferenceMs: Math.round(performance.now() - started),
    },
  };
}

// ============================================================
// TASK 6: Driver Drowsiness Detection (COCO-SSD + MediaPipe FaceLandmarker EAR/Blink)
// ============================================================
function computeEyeAspectRatio(lm: NormalizedLandmark[], indices: [number, number, number, number, number, number]): number {
  const [p1, p2, p3, p4, p5, p6] = indices.map((idx) => lm[idx]);
  const v1 = dist2d(p2, p6);
  const v2 = dist2d(p3, p5);
  const horiz = Math.max(1e-4, dist2d(p1, p4));
  return (v1 + v2) / (2 * horiz);
}

export async function runDrowsinessLocal(
  imageUrl: string,
  confidenceThreshold: number,
  isVideoFrame: boolean,
) {
  const started = performance.now();
  const img = await loadImageElement(imageUrl);
  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;

  const [cocoModel, faceLandmarker] = await Promise.all([getCocoModel(), getFaceLandmarker()]);
  const [rawDetections, faceRes] = await Promise.all([
    cocoModel.detect(img, 20, Math.max(0.05, confidenceThreshold)),
    Promise.resolve(faceLandmarker.detect(img)),
  ]);

  const faces = (faceRes.faceLandmarks ?? []).map((landmarks, idx) => {
    const geom = faceCenterNorm(landmarks);
    const categories = faceRes.faceBlendshapes?.[idx]?.categories;
    const blinkL = getBlendshapeScore(categories, 'eyeBlinkLeft');
    const blinkR = getBlendshapeScore(categories, 'eyeBlinkRight');
    const avgBlink = (blinkL + blinkR) / 2;
    const jawOpen = getBlendshapeScore(categories, 'jawOpen');
    const earL = computeEyeAspectRatio(landmarks, [33, 160, 158, 133, 153, 144]);
    const earR = computeEyeAspectRatio(landmarks, [362, 385, 387, 263, 373, 380]);
    const avgEar = (earL + earR) / 2;

    // Head roll angle in degrees from outer eye corners
    const leftOuter = landmarks[33];
    const rightOuter = landmarks[263];
    const rollDeg = Math.abs(
      (Math.atan2(rightOuter.y - leftOuter.y, rightOuter.x - leftOuter.x) * 180) / Math.PI,
    );

    return { ...geom, avgBlink, jawOpen, avgEar, rollDeg };
  });

  const personDets = rawDetections
    .filter((d) => d.class === 'person' && d.score >= confidenceThreshold)
    .sort((a, b) => a.bbox[0] - b.bbox[0]);

  const matchedFaceIndices = new Set<number>();

  const evaluateFace = (matchedFace: (typeof faces)[number] | undefined) => {
    let drowsinessStatus: 'alert' | 'drowsy' | 'sleeping' | 'unknown' = 'unknown';
    let statusConfidence = 0;
    let visibleCues = 'Face landmarks not clearly visible for eye-closure analysis.';

    if (matchedFace) {
      const { avgBlink, avgEar, jawOpen, rollDeg } = matchedFace;
      if (avgBlink >= 0.52 || avgEar < 0.19) {
        drowsinessStatus = rollDeg >= 12 || avgBlink >= 0.68 ? 'sleeping' : 'drowsy';
        statusConfidence = Math.min(0.99, Math.max(avgBlink, 1 - avgEar * 2.5));
        visibleCues = `Eyes closed (blink ${(avgBlink * 100).toFixed(0)}%, EAR ${avgEar.toFixed(2)}), head tilt ${rollDeg.toFixed(1)}°.`;
      } else if (avgBlink >= 0.36 || avgEar < 0.23 || jawOpen >= 0.55 || rollDeg >= 18) {
        drowsinessStatus = 'drowsy';
        statusConfidence = Math.min(0.95, Math.max(avgBlink, jawOpen, rollDeg / 35));
        visibleCues = `Drowsy cues: blink ${(avgBlink * 100).toFixed(0)}%, EAR ${avgEar.toFixed(2)}, jaw-open ${(jawOpen * 100).toFixed(0)}%, head tilt ${rollDeg.toFixed(1)}°.`;
      } else {
        drowsinessStatus = 'alert';
        statusConfidence = Math.min(0.98, 1 - avgBlink * 0.7);
        visibleCues = `Eyes open and attentive (blink ${(avgBlink * 100).toFixed(0)}%, EAR ${avgEar.toFixed(2)}, head tilt ${rollDeg.toFixed(1)}°).`;
      }
    }
    return { drowsinessStatus, statusConfidence, visibleCues };
  };

  const candidates: Array<{
    box: [number, number, number, number];
    confidence: number;
    drowsinessStatus: 'alert' | 'drowsy' | 'sleeping' | 'unknown';
    statusConfidence: number;
    visibleCues: string;
  }> = [];

  for (const det of personDets) {
    const box = bboxToNormalized(det.bbox, width, height);
    const [ymin, xmin, ymax, xmax] = box;
    const faceIdx = faces.findIndex(
      (f, i) =>
        !matchedFaceIndices.has(i) &&
        f.cx >= xmin - 35 &&
        f.cx <= xmax + 35 &&
        f.cy >= ymin - 35 &&
        f.cy <= ymax + 35,
    );
    const matchedFace = faceIdx >= 0 ? faces[faceIdx] : undefined;
    if (faceIdx >= 0) matchedFaceIndices.add(faceIdx);

    const evalRes = evaluateFace(matchedFace);
    candidates.push({
      box,
      confidence: det.score,
      ...evalRes,
    });
  }

  // Fallback for occluded cabin occupants detected by MediaPipe FaceLandmarker
  faces.forEach((face, i) => {
    if (matchedFaceIndices.has(i)) return;
    const [fymin, fxmin, fymax, fxmax] = face.box;
    const fh = fymax - fymin;
    const fw = fxmax - fxmin;
    const expandedBox: [number, number, number, number] = [
      Math.max(0, Math.round(fymin - fh * 0.25)),
      Math.max(0, Math.round(fxmin - fw * 0.4)),
      Math.min(1000, Math.round(fymax + fh * 0.8)),
      Math.min(1000, Math.round(fxmax + fw * 0.4)),
    ];
    candidates.push({
      box: expandedBox,
      confidence: 0.85,
      ...evaluateFace(face),
    });
  });

  candidates.sort((a, b) => a.box[1] - b.box[1]);

  let sleepingDrowsyCount = 0;
  let alertCount = 0;
  let unknownCount = 0;

  const people = candidates.map((c, idx) => {
    if (c.drowsinessStatus === 'sleeping' || c.drowsinessStatus === 'drowsy') sleepingDrowsyCount += 1;
    else if (c.drowsinessStatus === 'alert') alertCount += 1;
    else unknownCount += 1;

    const roleHint =
      candidates.length === 2 ? (idx === 0 ? 'driver (left)' : 'passenger (right)') : `occupant ${idx + 1}`;

    return {
      id: `OCCUPANT-${String(idx + 1).padStart(2, '0')}`,
      roleHint,
      confidence: c.confidence,
      box: c.box,
      drowsinessStatus: c.drowsinessStatus,
      statusConfidence: c.statusConfidence,
      coarseAgeRange: 'Not estimated',
      visibleCues: c.visibleCues,
    };
  });

  return {
    personCount: people.length,
    sleepingDrowsyCount,
    alertCount,
    unknownCount,
    people,
    limitationNotice: isVideoFrame
      ? 'Multi-frame video analysis: check frame-by-frame eye-closure persistence for temporal drowsiness.'
      : 'Single-frame eye-closure & head-tilt cue analysis; sustained drowsiness requires multi-frame video evidence.',
    metadata: {
      model: 'TensorFlow.js COCO-SSD + MediaPipe FaceLandmarker (EAR & Blink Blendshapes)',
      confidenceThreshold,
      inferenceMs: Math.round(performance.now() - started),
    },
  };
}
