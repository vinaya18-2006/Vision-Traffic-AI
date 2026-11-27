import express from 'express';
import type { Request, Response } from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;
const HOST = '0.0.0.0';
const VISION_AUDIO_MODEL = 'gemini-4-hydrogen';
const TTS_MODEL = 'gemini-3.8-flash-lite-tts';

const ALLOWED_IMAGE_MIMES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const ALLOWED_AUDIO_MIMES = new Set([
  'audio/wav',
  'audio/x-wav',
  'audio/mpeg',
  'audio/mp3',
  'audio/webm',
  'audio/ogg',
  'audio/mp4',
  'audio/x-m4a',
]);
const MAX_BASE64_LENGTH = 20 * 1024 * 1024; // ~15 MB decoded binary

const KNOWN_CAR_COLOURS = new Set([
  'blue',
  'red',
  'white',
  'black',
  'gray',
  'silver',
  'green',
  'yellow',
  'orange',
  'brown',
  'other',
]);

const ASL_STATIC_VOCABULARY = new Set([
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
  'U', 'V', 'W', 'X', 'Y',
]);

// Deterministic dietary taxonomy (COCO 2017 animal classes + common wildlife species)
const WILDLIFE_DIET_MAP: Record<string, { category: 'carnivore' | 'non_carnivore' | 'unknown'; note: string }> = {
  cat: { category: 'carnivore', note: 'Obligate carnivore (COCO class).' },
  lion: { category: 'carnivore', note: 'Obligate carnivore (feline wildlife).' },
  tiger: { category: 'carnivore', note: 'Obligate carnivore (feline wildlife).' },
  leopard: { category: 'carnivore', note: 'Obligate carnivore (feline wildlife).' },
  cheetah: { category: 'carnivore', note: 'Obligate carnivore (feline wildlife).' },
  wolf: { category: 'carnivore', note: 'Primary carnivore (canid wildlife).' },
  crocodile: { category: 'carnivore', note: 'Obligate carnivore (reptile).' },
  alligator: { category: 'carnivore', note: 'Obligate carnivore (reptile).' },
  snake: { category: 'carnivore', note: 'Carnivorous reptile.' },
  eagle: { category: 'carnivore', note: 'Raptor / carnivore.' },
  hawk: { category: 'carnivore', note: 'Raptor / carnivore.' },
  owl: { category: 'carnivore', note: 'Raptor / carnivore.' },
  horse: { category: 'non_carnivore', note: 'Herbivore (COCO class).' },
  sheep: { category: 'non_carnivore', note: 'Herbivore (COCO class).' },
  cow: { category: 'non_carnivore', note: 'Herbivore (COCO class).' },
  elephant: { category: 'non_carnivore', note: 'Herbivore (COCO class).' },
  zebra: { category: 'non_carnivore', note: 'Herbivore (COCO class).' },
  giraffe: { category: 'non_carnivore', note: 'Herbivore (COCO class).' },
  deer: { category: 'non_carnivore', note: 'Herbivore (wildlife).' },
  goat: { category: 'non_carnivore', note: 'Herbivore (livestock/wildlife).' },
  rabbit: { category: 'non_carnivore', note: 'Herbivore (mammal).' },
  bird: { category: 'unknown', note: 'Broad COCO class spanning many species and diets; left unclassified.' },
  dog: { category: 'unknown', note: 'Domestic dogs are dietary generalists; broad label is unclassified.' },
  bear: { category: 'unknown', note: 'COCO bear label spans varied diets; left unclassified.' },
  fox: { category: 'unknown', note: 'Omnivorous canid; left unclassified under strict carnivore policy.' },
};

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      'GEMINI_API_KEY is not configured on the server. Add it in Settings > Secrets to run inference.',
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function validateMediaPayload(
  body: unknown,
  allowedMimes: Set<string>,
  mediaLabel: string,
): { base64: string; mimeType: string; record: Record<string, unknown> } {
  if (!body || typeof body !== 'object') {
    throw new Error('Request body must be a JSON object.');
  }
  const record = body as Record<string, unknown>;
  const rawBase64 = typeof record.mediaBase64 === 'string' ? record.mediaBase64.trim() : '';
  const rawMime = typeof record.mimeType === 'string' ? record.mimeType.trim().toLowerCase() : '';
  const cleanMime = rawMime.split(';')[0].trim();

  if (!rawBase64) {
    throw new Error(`Missing ${mediaLabel} payload.`);
  }
  if (rawBase64.length > MAX_BASE64_LENGTH) {
    throw new Error(`Uploaded ${mediaLabel} exceeds the 15 MB size limit.`);
  }
  if (!allowedMimes.has(cleanMime)) {
    throw new Error(
      `Unsupported ${mediaLabel} MIME type "${cleanMime || 'unknown'}". Allowed: ${Array.from(allowedMimes).join(', ')}.`,
    );
  }
  return { base64: rawBase64, mimeType: cleanMime, record };
}

function clampThreshold(value: unknown, fallback: number): number {
  const num = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.max(0, Math.min(1, num));
}

function sanitizeBox(raw: unknown): [number, number, number, number] {
  if (!Array.isArray(raw) || raw.length !== 4) {
    return [0, 0, 0, 0];
  }
  const nums = raw.map((n) => Math.max(0, Math.min(1000, Math.round(Number(n) || 0))));
  const ymin = Math.min(nums[0], nums[2]);
  const xmin = Math.min(nums[1], nums[3]);
  const ymax = Math.max(nums[0], nums[2]);
  const xmax = Math.max(nums[1], nums[3]);
  return [ymin, xmin, ymax, xmax];
}

function handleApiError(res: Response, error: unknown, context: string) {
  const message = error instanceof Error ? error.message : 'Unexpected inference error.';
  console.error(`[${context}]`, error);
  const status = message.includes('Missing ') || message.includes('Unsupported ') || message.includes('exceeds')
    ? 400
    : message.includes('GEMINI_API_KEY')
      ? 503
      : 500;
  res.status(status).json({ error: message });
}

function parseJsonResponse<T>(text: string | undefined): T {
  const cleaned = (text ?? '').trim();
  if (!cleaned) {
    throw new Error('Model returned an empty response.');
  }
  return JSON.parse(cleaned) as T;
}

function checkTimeWindow(startHHMM: string, endHHMM: string, timeZone: string): {
  authorized: boolean;
  localTime: string;
  reason?: string;
} {
  try {
    const now = new Date();
    const formatted = new Intl.DateTimeFormat('en-GB', {
      timeZone,
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).format(now);
    const toMinutes = (hhmm: string) => {
      const [h, m] = hhmm.split(':').map(Number);
      return h * 60 + m;
    };
    const nowMin = toMinutes(formatted);
    const startMin = toMinutes(startHHMM);
    const endMin = toMinutes(endHHMM);
    if (!Number.isFinite(startMin) || !Number.isFinite(endMin)) {
      return { authorized: false, localTime: formatted, reason: 'Invalid operating window format.' };
    }
    const authorized = startMin <= endMin
      ? nowMin >= startMin && nowMin < endMin
      : nowMin >= startMin || nowMin < endMin;
    return {
      authorized,
      localTime: formatted,
      reason: authorized
        ? undefined
        : `Sign-language inference is closed outside ${startHHMM}–${endHHMM} ${timeZone} (current time: ${formatted}).`,
    };
  } catch {
    return { authorized: false, localTime: 'unknown', reason: `Invalid timezone "${timeZone}".` };
  }
}

async function createApp() {
  const app = express();
  app.use(express.json({ limit: '25mb' }));
  app.use(
    '/mediapipe-wasm',
    express.static(path.resolve(__dirname, 'node_modules/@mediapipe/tasks-vision/wasm')),
  );

  app.get('/api/health', (_req: Request, res: Response) => {
    res.json({
      ok: true,
      model: VISION_AUDIO_MODEL,
      apiConfigured: Boolean(process.env.GEMINI_API_KEY),
    });
  });

  // TASK 1: Car Colour Detection
  app.post('/api/analyze/car-detection', async (req: Request, res: Response) => {
    try {
      const started = Date.now();
      const { base64, mimeType, record } = validateMediaPayload(req.body, ALLOWED_IMAGE_MIMES, 'image');
      const confidenceThreshold = clampThreshold(record.confidenceThreshold, 0.25);
      const colourThreshold = clampThreshold(record.colourThreshold, 0.15);

      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: VISION_AUDIO_MODEL,
        contents: {
          parts: [
            { inlineData: { mimeType, data: base64 } },
            {
              text: [
                'Detect all clearly visible cars (sedans, hatchbacks, SUVs, passenger cars) and people/pedestrians in this traffic scene.',
                'Return normalized bounding boxes [ymin, xmin, ymax, xmax] scaled 0 to 1000.',
                'For each car, estimate its apparent primary body colour from: blue, red, white, black, gray, silver, green, yellow, orange, brown, other, unknown.',
                'Include a detection confidence (0 to 1) and for cars a colourEvidence score (0 to 1). Do not fabricate detections if none are visible.',
              ].join(' '),
            },
          ],
        },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              detections: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    className: { type: Type.STRING, description: 'Either "car" or "person".' },
                    confidence: { type: Type.NUMBER },
                    box_2d: { type: Type.ARRAY, items: { type: Type.INTEGER } },
                    colour: { type: Type.STRING, description: 'Apparent car body colour or "n/a" for person.' },
                    colourEvidence: { type: Type.NUMBER },
                  },
                  required: ['className', 'confidence', 'box_2d'],
                },
              },
              sceneSummary: { type: Type.STRING },
            },
            required: ['detections'],
          },
        },
      });

      const parsed = parseJsonResponse<{
        detections: Array<{
          className: string;
          confidence: number;
          box_2d: number[];
          colour?: string;
          colourEvidence?: number;
        }>;
        sceneSummary?: string;
      }>(response.text);

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

      for (const det of parsed.detections ?? []) {
        const conf = clampThreshold(det.confidence, 0);
        if (conf < confidenceThreshold) continue;
        const box = sanitizeBox(det.box_2d);
        if (box[2] - box[0] < 5 || box[3] - box[1] < 5) continue;

        const cls = (det.className || '').toLowerCase().trim();
        if (cls === 'car') {
          const ev = clampThreshold(det.colourEvidence ?? 0.85, 0.85);
          const rawColour = (det.colour || 'unknown').toLowerCase().trim();
          const finalColour = ev < colourThreshold || !KNOWN_CAR_COLOURS.has(rawColour) ? 'unknown' : rawColour;
          const boxStyle = finalColour === 'blue' ? 'red' : finalColour === 'unknown' ? 'gray' : 'blue';
          cars.push({
            id: `CAR-${String(cars.length + 1).padStart(3, '0')}`,
            className: 'car',
            confidence: conf,
            box,
            colour: finalColour,
            colourEvidence: ev,
            boxStyle,
          });
        } else if (cls === 'person') {
          people.push({
            id: `PERSON-${String(people.length + 1).padStart(3, '0')}`,
            className: 'person',
            confidence: conf,
            box,
          });
        }
      }

      const blueCars = cars.filter((c) => c.colour === 'blue').length;
      const unknownColourCars = cars.filter((c) => c.colour === 'unknown').length;
      const otherKnownColourCars = cars.length - blueCars - unknownColourCars;

      res.json({
        carCount: cars.length,
        blueCarCount: blueCars,
        otherKnownColourCarCount: otherKnownColourCars,
        unknownColourCarCount: unknownColourCars,
        personCount: people.length,
        cars,
        people,
        sceneSummary: parsed.sceneSummary ?? '',
        metadata: {
          model: VISION_AUDIO_MODEL,
          confidenceThreshold,
          colourThreshold,
          inferenceMs: Date.now() - started,
        },
      });
    } catch (error) {
      handleApiError(res, error, 'car-detection');
    }
  });

  // TASK 2: Wildlife Risk Intelligence
  app.post('/api/analyze/wildlife', async (req: Request, res: Response) => {
    try {
      const started = Date.now();
      const { base64, mimeType, record } = validateMediaPayload(req.body, ALLOWED_IMAGE_MIMES, 'image');
      const confidenceThreshold = clampThreshold(record.confidenceThreshold, 0.25);

      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: VISION_AUDIO_MODEL,
        contents: {
          parts: [
            { inlineData: { mimeType, data: base64 } },
            {
              text: [
                'Detect all clearly visible animals in this image.',
                'Return normalized bounding boxes [ymin, xmin, ymax, xmax] scaled 0 to 1000.',
                'Use canonical lowercase singular English species names matching COCO categories when applicable (cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe) or specific wildlife species (lion, tiger, wolf, fox, deer, goat, rabbit, eagle, snake, crocodile).',
                'Return detection confidence from 0 to 1. Do not fabricate animals if none exist.',
              ].join(' '),
            },
          ],
        },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              detections: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    species: { type: Type.STRING },
                    confidence: { type: Type.NUMBER },
                    box_2d: { type: Type.ARRAY, items: { type: Type.INTEGER } },
                  },
                  required: ['species', 'confidence', 'box_2d'],
                },
              },
            },
            required: ['detections'],
          },
        },
      });

      const parsed = parseJsonResponse<{
        detections: Array<{ species: string; confidence: number; box_2d: number[] }>;
      }>(response.text);

      const detections = [];
      const speciesCounts: Record<string, number> = {};
      let carnivores = 0;
      let nonCarnivores = 0;
      let unknown = 0;

      for (const det of parsed.detections ?? []) {
        const conf = clampThreshold(det.confidence, 0);
        if (conf < confidenceThreshold) continue;
        const box = sanitizeBox(det.box_2d);
        if (box[2] - box[0] < 5 || box[3] - box[1] < 5) continue;

        const species = (det.species || 'unknown').toLowerCase().trim();
        const mapped = WILDLIFE_DIET_MAP[species] ?? {
          category: 'unknown' as const,
          note: 'Species is outside the configured dietary taxonomy; left unclassified.',
        };

        if (mapped.category === 'carnivore') carnivores += 1;
        else if (mapped.category === 'non_carnivore') nonCarnivores += 1;
        else unknown += 1;

        speciesCounts[species] = (speciesCounts[species] ?? 0) + 1;
        detections.push({
          id: `ANIMAL-${String(detections.length + 1).padStart(3, '0')}`,
          species,
          confidence: conf,
          box,
          classification: mapped.category,
          note: mapped.note,
        });
      }

      res.json({
        summary: {
          totalAnimals: detections.length,
          carnivores,
          nonCarnivores,
          unknown,
          speciesCounts,
        },
        detections,
        metadata: {
          model: VISION_AUDIO_MODEL,
          taxonomyVersion: '1.0.0',
          confidenceThreshold,
          inferenceMs: Date.now() - started,
        },
      });
    } catch (error) {
      handleApiError(res, error, 'wildlife');
    }
  });

  // TASK 3: Voice Emotion Detection
  app.post('/api/analyze/voice-emotion', async (req: Request, res: Response) => {
    try {
      const started = Date.now();
      const { base64, mimeType, record } = validateMediaPayload(req.body, ALLOWED_AUDIO_MIMES, 'audio');
      const minimumScore = clampThreshold(record.minimumScore, 0.0);

      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: VISION_AUDIO_MODEL,
        contents: {
          parts: [
            { inlineData: { mimeType, data: base64 } },
            {
              text: [
                'Classify the dominant speech emotion in this voice clip into exactly one of these 4 SUPERB/IEMOCAP labels: neutral, happy, angry, sad.',
                'Provide normalized score estimates (0 to 1 summing roughly to 1) for all four classes, the primary label, confidence score, brief acoustic prosody observations (tone, energy, speaking rate), and a short transcript snippet if speech is intelligible.',
                'If the clip is silent, noisy, or non-speech, note that in warning and keep confidence low.',
                'IMPORTANT POLICY: Never infer or mention the speaker gender, identity, age, or nationality.',
              ].join(' '),
            },
          ],
        },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              label: { type: Type.STRING, description: 'One of: neutral, happy, angry, sad' },
              confidence: { type: Type.NUMBER },
              scores: {
                type: Type.OBJECT,
                properties: {
                  neutral: { type: Type.NUMBER },
                  happy: { type: Type.NUMBER },
                  angry: { type: Type.NUMBER },
                  sad: { type: Type.NUMBER },
                },
                required: ['neutral', 'happy', 'angry', 'sad'],
              },
              acousticCues: { type: Type.STRING },
              transcriptExcerpt: { type: Type.STRING },
              warning: { type: Type.STRING },
            },
            required: ['label', 'confidence', 'scores', 'acousticCues'],
          },
        },
      });

      const parsed = parseJsonResponse<{
        label: string;
        confidence: number;
        scores: { neutral: number; happy: number; angry: number; sad: number };
        acousticCues: string;
        transcriptExcerpt?: string;
        warning?: string;
      }>(response.text);

      const validLabels = new Set(['neutral', 'happy', 'angry', 'sad']);
      const label = validLabels.has((parsed.label || '').toLowerCase())
        ? parsed.label.toLowerCase()
        : 'neutral';
      const confidence = clampThreshold(parsed.confidence, 0);
      const warnings: string[] = [];
      if (parsed.warning) warnings.push(parsed.warning);
      if (confidence < minimumScore) {
        warnings.push(
          `Model score (${(confidence * 100).toFixed(1)}%) is below your configured threshold (${(minimumScore * 100).toFixed(0)}%).`,
        );
      }

      res.json({
        label,
        confidence,
        scores: {
          neutral: clampThreshold(parsed.scores?.neutral, 0),
          happy: clampThreshold(parsed.scores?.happy, 0),
          angry: clampThreshold(parsed.scores?.angry, 0),
          sad: clampThreshold(parsed.scores?.sad, 0),
        },
        acousticCues: parsed.acousticCues || '',
        transcriptExcerpt: parsed.transcriptExcerpt || '',
        warning: warnings.join(' ') || null,
        metadata: {
          model: VISION_AUDIO_MODEL,
          labelSchema: 'SUPERB/IEMOCAP (neutral, happy, angry, sad)',
          minimumScore,
          inferenceMs: Date.now() - started,
        },
      });
    } catch (error) {
      handleApiError(res, error, 'voice-emotion');
    }
  });

  // TASK 3 Helper: Generate a realistic spoken WAV sample for instant testing
  app.post('/api/sample-voice', async (req: Request, res: Response) => {
    try {
      const requested = typeof req.body?.emotion === 'string' ? req.body.emotion.toLowerCase() : 'happy';
      const presets: Record<string, { text: string; style: string; voice: string }> = {
        happy: {
          text: 'Oh, this is wonderful news! We passed every single test on the first try and I could not be happier!',
          style: 'Bright, enthusiastic, joyful and smiling voice',
          voice: 'Kore',
        },
        angry: {
          text: 'I already told you three times to stop ignoring the warning lights! This is completely unacceptable!',
          style: 'Frustrated, sharp, forceful angry tone',
          voice: 'Fenrir',
        },
        sad: {
          text: 'I really hoped things would turn out differently today... it just feels so exhausting and disappointing.',
          style: 'Soft, slow, heavy-hearted and sorrowful tone',
          voice: 'Charon',
        },
        neutral: {
          text: 'The traffic monitoring report for Tuesday has been saved in the archive folder for review.',
          style: 'Calm, flat, objective informational voice',
          voice: 'Puck',
        },
      };
      const chosen = presets[requested] ?? presets.happy;
      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: TTS_MODEL,
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: chosen.text,
                speechMetadata: { style: chosen.style },
              },
            ],
          },
        ],
        config: {
          responseModalities: ['AUDIO'],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: chosen.voice },
            },
          },
        },
      });

      const audioBase64 = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (!audioBase64) {
        throw new Error('Speech synthesis did not return audio data.');
      }
      res.json({
        emotionHint: requested,
        mimeType: 'audio/wav',
        audioBase64,
        filename: `sample_${requested}_speech.wav`,
      });
    } catch (error) {
      handleApiError(res, error, 'sample-voice');
    }
  });

  // TASK 4: Person Attribute & Emotion Analysis
  app.post('/api/analyze/person-analysis', async (req: Request, res: Response) => {
    try {
      const started = Date.now();
      const { base64, mimeType, record } = validateMediaPayload(req.body, ALLOWED_IMAGE_MIMES, 'image');
      const confidenceThreshold = clampThreshold(record.confidenceThreshold, 0.25);
      const userProvidedRegion =
        typeof record.userProvidedRegion === 'string'
          ? record.userProvidedRegion.trim().slice(0, 80)
          : '';

      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: VISION_AUDIO_MODEL,
        contents: {
          parts: [
            { inlineData: { mimeType, data: base64 } },
            {
              text: [
                'Detect all clearly visible foreground/midground people in this image with normalized bounding boxes [ymin, xmin, ymax, xmax] scaled 0 to 1000.',
                'For each person provide:',
                '- confidence (0 to 1)',
                '- coarseAgeRange (a coarse 10-year range such as "20-30", "45-55", or "unknown" if unclear; never a single exact age)',
                '- expressionEstimate (visible facial expression only: happy, neutral, sad, surprised, angry, serious, or unknown)',
                '- expressionConfidence (0 to 1)',
                '- visibleAttributes (short list of non-sensitive visible cues like "eyeglasses", "green jacket", "navy coat", "tote bag").',
                'STRICT POLICY: Never infer or mention nationality, ethnicity, race, or identity.',
              ].join(' '),
            },
          ],
        },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              people: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    confidence: { type: Type.NUMBER },
                    box_2d: { type: Type.ARRAY, items: { type: Type.INTEGER } },
                    coarseAgeRange: { type: Type.STRING },
                    expressionEstimate: { type: Type.STRING },
                    expressionConfidence: { type: Type.NUMBER },
                    visibleAttributes: { type: Type.ARRAY, items: { type: Type.STRING } },
                  },
                  required: ['confidence', 'box_2d', 'coarseAgeRange', 'expressionEstimate'],
                },
              },
            },
            required: ['people'],
          },
        },
      });

      const parsed = parseJsonResponse<{
        people: Array<{
          confidence: number;
          box_2d: number[];
          coarseAgeRange: string;
          expressionEstimate: string;
          expressionConfidence?: number;
          visibleAttributes?: string[];
        }>;
      }>(response.text);

      const people = [];
      for (const p of parsed.people ?? []) {
        const conf = clampThreshold(p.confidence, 0);
        if (conf < confidenceThreshold) continue;
        const box = sanitizeBox(p.box_2d);
        if (box[2] - box[0] < 5 || box[3] - box[1] < 5) continue;
        people.push({
          id: `PERSON-${String(people.length + 1).padStart(3, '0')}`,
          confidence: conf,
          box,
          coarseAgeRange: p.coarseAgeRange || 'unknown',
          expressionEstimate: (p.expressionEstimate || 'unknown').toLowerCase(),
          expressionConfidence: clampThreshold(p.expressionConfidence ?? 0.7, 0.7),
          visibleAttributes: Array.isArray(p.visibleAttributes) ? p.visibleAttributes.slice(0, 5) : [],
        });
      }

      res.json({
        personCount: people.length,
        people,
        userProvidedRegion: userProvidedRegion || null,
        metadata: {
          model: VISION_AUDIO_MODEL,
          confidenceThreshold,
          nationalityInferredFromAppearance: false,
          inferenceMs: Date.now() - started,
        },
      });
    } catch (error) {
      handleApiError(res, error, 'person-analysis');
    }
  });

  // TASK 5: Sign Language Detection (Static ASL Alphabet A-Y excluding J, Z)
  app.post('/api/analyze/sign-language', async (req: Request, res: Response) => {
    try {
      const started = Date.now();
      const { base64, mimeType, record } = validateMediaPayload(req.body, ALLOWED_IMAGE_MIMES, 'image');
      const minimumScore = clampThreshold(record.minimumScore, 0.6);
      const enforceWindow = record.enforceWindow !== false;
      const allowedStart = typeof record.allowedStart === 'string' ? record.allowedStart : '09:00';
      const allowedEnd = typeof record.allowedEnd === 'string' ? record.allowedEnd : '17:00';
      const timeZone = typeof record.timeZone === 'string' ? record.timeZone : 'America/New_York';

      const windowStatus = checkTimeWindow(allowedStart, allowedEnd, timeZone);
      if (enforceWindow && !windowStatus.authorized) {
        res.status(403).json({
          error: windowStatus.reason || 'Sign-language recognition is outside the configured operating window.',
          windowStatus,
        });
        return;
      }

      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: VISION_AUDIO_MODEL,
        contents: {
          parts: [
            { inlineData: { mimeType, data: base64 } },
            {
              text: [
                'Recognize a single static American Sign Language (ASL) fingerspelled letter from this image.',
                'Supported vocabulary (24 static letters ONLY): A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y.',
                'Dynamic letters J and Z are NOT supported (return status "unknown" if shown).',
                'If no clear human hand is visible, return status "no_hand". If multiple hands are gesturing, return status "multiple_hands".',
                'If a single hand makes one of the 24 static ASL letters, return status "recognized", the uppercase letter in label, confidence (0 to 1), hand bounding box [ymin, xmin, ymax, xmax] scaled 0 to 1000, and a short finger-posture description.',
              ].join(' '),
            },
          ],
        },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              status: { type: Type.STRING, description: 'recognized, no_hand, multiple_hands, or unknown' },
              label: { type: Type.STRING },
              confidence: { type: Type.NUMBER },
              box_2d: { type: Type.ARRAY, items: { type: Type.INTEGER } },
              handDescription: { type: Type.STRING },
              warning: { type: Type.STRING },
            },
            required: ['status', 'confidence', 'handDescription'],
          },
        },
      });

      const parsed = parseJsonResponse<{
        status: string;
        label?: string;
        confidence: number;
        box_2d?: number[];
        handDescription: string;
        warning?: string;
      }>(response.text);

      const confidence = clampThreshold(parsed.confidence, 0);
      const rawStatus = (parsed.status || 'unknown').toLowerCase();
      const rawLabel = (parsed.label || '').toUpperCase().trim();

      let finalStatus: 'recognized' | 'no_hand' | 'multiple_hands' | 'unknown' = 'unknown';
      let finalLabel: string | null = null;
      const warnings: string[] = [];
      if (parsed.warning) warnings.push(parsed.warning);

      if (rawStatus === 'no_hand') {
        finalStatus = 'no_hand';
      } else if (rawStatus === 'multiple_hands') {
        finalStatus = 'multiple_hands';
        warnings.push('Multiple hands detected; static ASL fingerspelling requires a single clearly visible hand.');
      } else if (rawStatus === 'recognized' && ASL_STATIC_VOCABULARY.has(rawLabel)) {
        if (confidence >= minimumScore) {
          finalStatus = 'recognized';
          finalLabel = rawLabel;
        } else {
          finalStatus = 'unknown';
          warnings.push(
            `Top candidate "${rawLabel}" (${(confidence * 100).toFixed(1)}%) was below the ${(minimumScore * 100).toFixed(0)}% minimum score threshold.`,
          );
        }
      } else {
        finalStatus = 'unknown';
      }

      const box = parsed.box_2d ? sanitizeBox(parsed.box_2d) : null;
      const validBox = box && box[2] - box[0] > 5 && box[3] - box[1] > 5 ? box : null;

      res.json({
        status: finalStatus,
        label: finalLabel,
        confidence,
        box: validBox,
        handDescription: parsed.handDescription || '',
        warning: warnings.join(' ') || null,
        windowStatus,
        metadata: {
          model: VISION_AUDIO_MODEL,
          vocabularyCount: ASL_STATIC_VOCABULARY.size,
          minimumScore,
          enforceWindow,
          inferenceMs: Date.now() - started,
        },
      });
    } catch (error) {
      handleApiError(res, error, 'sign-language');
    }
  });

  // TASK 6: Driver Drowsiness Detection
  app.post('/api/analyze/drowsiness', async (req: Request, res: Response) => {
    try {
      const started = Date.now();
      const { base64, mimeType, record } = validateMediaPayload(req.body, ALLOWED_IMAGE_MIMES, 'image');
      const confidenceThreshold = clampThreshold(record.confidenceThreshold, 0.25);
      const isVideoFrame = Boolean(record.isVideoFrame);

      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: VISION_AUDIO_MODEL,
        contents: {
          parts: [
            { inlineData: { mimeType, data: base64 } },
            {
              text: [
                'Detect all people (driver and passengers) inside this vehicle scene with normalized bounding boxes [ymin, xmin, ymax, xmax] scaled 0 to 1000.',
                'For each person, carefully evaluate visible eye closure, heavy eyelids, head nodding/slumping, or yawning to classify visible state as one of: alert, drowsy, sleeping, unknown.',
                'Also estimate a coarse 10-year age range (e.g., "25-35", "30-40", or "unknown") and list the exact observable visual cues.',
                'Return detection confidence (0 to 1) and statusConfidence (0 to 1).',
              ].join(' '),
            },
          ],
        },
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              people: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    roleHint: { type: Type.STRING, description: 'e.g., driver, passenger, or occupant' },
                    confidence: { type: Type.NUMBER },
                    box_2d: { type: Type.ARRAY, items: { type: Type.INTEGER } },
                    drowsinessStatus: { type: Type.STRING, description: 'alert, drowsy, sleeping, or unknown' },
                    statusConfidence: { type: Type.NUMBER },
                    coarseAgeRange: { type: Type.STRING },
                    visibleCues: { type: Type.STRING },
                  },
                  required: ['confidence', 'box_2d', 'drowsinessStatus', 'statusConfidence', 'visibleCues'],
                },
              },
            },
            required: ['people'],
          },
        },
      });

      const parsed = parseJsonResponse<{
        people: Array<{
          roleHint?: string;
          confidence: number;
          box_2d: number[];
          drowsinessStatus: string;
          statusConfidence: number;
          coarseAgeRange?: string;
          visibleCues: string;
        }>;
      }>(response.text);

      const validStates = new Set(['alert', 'drowsy', 'sleeping', 'unknown']);
      const people = [];
      let sleepingDrowsyCount = 0;
      let alertCount = 0;
      let unknownCount = 0;

      for (const p of parsed.people ?? []) {
        const conf = clampThreshold(p.confidence, 0);
        if (conf < confidenceThreshold) continue;
        const box = sanitizeBox(p.box_2d);
        if (box[2] - box[0] < 5 || box[3] - box[1] < 5) continue;

        const rawState = (p.drowsinessStatus || 'unknown').toLowerCase().trim();
        const drowsinessStatus = (validStates.has(rawState) ? rawState : 'unknown') as
          | 'alert'
          | 'drowsy'
          | 'sleeping'
          | 'unknown';

        if (drowsinessStatus === 'drowsy' || drowsinessStatus === 'sleeping') {
          sleepingDrowsyCount += 1;
        } else if (drowsinessStatus === 'alert') {
          alertCount += 1;
        } else {
          unknownCount += 1;
        }

        people.push({
          id: `OCCUPANT-${String(people.length + 1).padStart(2, '0')}`,
          roleHint: p.roleHint || 'occupant',
          confidence: conf,
          box,
          drowsinessStatus,
          statusConfidence: clampThreshold(p.statusConfidence, 0),
          coarseAgeRange: p.coarseAgeRange || 'unknown',
          visibleCues: p.visibleCues || '',
        });
      }

      res.json({
        personCount: people.length,
        sleepingDrowsyCount,
        alertCount,
        unknownCount,
        people,
        limitationNotice: isVideoFrame
          ? 'Video frame evaluation: aggregate across frames for temporal drowsiness context.'
          : 'Single-frame visual sleep/drowsiness cues only; sustained drowsiness requires multi-frame temporal video evidence.',
        metadata: {
          model: VISION_AUDIO_MODEL,
          confidenceThreshold,
          inferenceMs: Date.now() - started,
        },
      });
    } catch (error) {
      handleApiError(res, error, 'drowsiness');
    }
  });

  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: false },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('/{*splat}', (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, HOST, () => {
    console.log(`RoadSight Intelligence server running at http://${HOST}:${PORT}`);
  });
}

createApp().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
