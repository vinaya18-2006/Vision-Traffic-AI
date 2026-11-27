import { useEffect, useRef, useState } from 'react';
import {
  Camera,
  Clock,
  Download,
  Hand,
  Image as ImageIcon,
  Loader2,
  Play,
  Sparkles,
  Upload,
} from 'lucide-react';
import sampleAslBImg from '../../assets/images/sample_asl_b_1795784440758.jpg';
import { SIGN_LANGUAGE_VOCABULARY_JSON } from '../../data/datasets';
import { annotateImageWithBoxes } from '../../utils/mediaUtils';
import { runSignLanguageLocal } from '../../ml/localEngine';

interface SignLanguageResponse {
  status: 'recognized' | 'no_hand' | 'multiple_hands' | 'unknown';
  label: string | null;
  confidence: number;
  box: [number, number, number, number] | null;
  handDescription: string;
  warning: string | null;
  windowStatus: {
    authorized: boolean;
    localTime: string;
    reason?: string;
  };
  metadata: {
    model: string;
    vocabularyCount: number;
    minimumScore: number;
    enforceWindow: boolean;
    inferenceMs: number;
  };
}

export function SignLanguageTask() {
  const [mode, setMode] = useState<'image' | 'webcam'>('image');
  const [minimumScore, setMinimumScore] = useState(0.6);
  const [enforceWindow, setEnforceWindow] = useState(false);
  const [allowedStart, setAllowedStart] = useState('09:00');
  const [allowedEnd, setAllowedEnd] = useState('17:00');
  const [sourcePreview, setSourcePreview] = useState<string | null>(null);
  const [annotatedPreview, setAnnotatedPreview] = useState<string | null>(null);
  const [result, setResult] = useState<SignLanguageResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraActive(false);
  };

  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
    } catch {
      setError('Camera permission was denied or no webcam was found.');
    }
  };

  const runRecognition = async (previewUrl: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setAnnotatedPreview(null);
    try {
      const data = await runSignLanguageLocal(previewUrl, {
        minimumScore,
        enforceWindow,
        allowedStart,
        allowedEnd,
        timeZone: 'America/New_York',
      });
      setSourcePreview(previewUrl);
      if (data.box || data.landmarks) {
        const boxes = data.box
          ? [
              {
                box: data.box,
                color: data.status === 'recognized' ? '#10b981' : '#f59e0b',
                label:
                  data.status === 'recognized' && data.label
                    ? `ASL "${data.label}" ${(data.confidence * 100).toFixed(0)}%`
                    : `UNKNOWN ${(data.confidence * 100).toFixed(0)}%`,
              },
            ]
          : [];
        const annotated = await annotateImageWithBoxes(previewUrl, boxes, data.landmarks);
        setAnnotatedPreview(annotated);
      } else {
        setAnnotatedPreview(previewUrl);
      }
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign language recognition failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleRunSample = async () => {
    stopCamera();
    setMode('image');
    setSourcePreview(sampleAslBImg);
    await runRecognition(sampleAslBImg);
  };

  const handleFileUpload = (file: File | undefined) => {
    if (!file) return;
    setError(null);
    setResult(null);
    setAnnotatedPreview(null);
    const preview = URL.createObjectURL(file);
    setSourcePreview(preview);
  };

  const handleCaptureWebcam = async () => {
    const video = videoRef.current;
    if (!video || !cameraActive) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    await runRecognition(dataUrl);
  };

  return (
    <div className="space-y-5">
      <div className="flex flex-col justify-between gap-4 rounded-2xl border border-slate-700/60 bg-gradient-to-br from-[#0d2535] to-[#0a1822] p-6 sm:flex-row sm:items-center">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.15em] text-cyan-300">
            Task 5 · Static ASL Fingerspelling Recognition
          </div>
          <h1 className="mt-1 flex items-center gap-2 text-2xl font-bold text-white">
            <Hand className="h-6 w-6 text-cyan-300" /> Sign Language Detection
          </h1>
          <p className="mt-1 text-sm text-slate-300">
            Recognizes 24 static ASL alphabet letters (A–Y excluding motion letters J and Z) with
            configurable operating window authorization.
          </p>
        </div>
        <button
          type="button"
          disabled={loading}
          onClick={handleRunSample}
          className="inline-flex shrink-0 items-center gap-2 rounded-xl bg-cyan-400 px-4 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-300 disabled:opacity-50"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
          Test ASL Letter &ldquo;B&rdquo; Sample
        </button>
      </div>

      {/* Controls */}
      <div className="grid gap-4 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5 lg:grid-cols-12">
        <div className="space-y-3 lg:col-span-5">
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => {
                stopCamera();
                setMode('image');
              }}
              className={`flex flex-1 items-center justify-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-semibold ${
                mode === 'image'
                  ? 'border-cyan-400 bg-cyan-500/20 text-white'
                  : 'border-slate-700 bg-[#07121a] text-slate-300'
              }`}
            >
              <ImageIcon className="h-3.5 w-3.5" /> Image Upload
            </button>
            <button
              type="button"
              onClick={() => setMode('webcam')}
              className={`flex flex-1 items-center justify-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-semibold ${
                mode === 'webcam'
                  ? 'border-cyan-400 bg-cyan-500/20 text-white'
                  : 'border-slate-700 bg-[#07121a] text-slate-300'
              }`}
            >
              <Camera className="h-3.5 w-3.5" /> Live Webcam
            </button>
          </div>

          {mode === 'image' ? (
            <label className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-dashed border-slate-600 bg-[#07121a] px-4 py-3 text-xs font-medium text-slate-200 hover:border-cyan-400">
              <Upload className="h-4 w-4 text-cyan-300" />
              Upload single-hand ASL image
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
                onChange={(e) => handleFileUpload(e.target.files?.[0])}
              />
            </label>
          ) : (
            <div className="space-y-2">
              <video
                ref={videoRef}
                muted
                playsInline
                className="h-44 w-full rounded-xl border border-slate-700 bg-black object-cover"
              />
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={cameraActive ? stopCamera : startCamera}
                  className="flex-1 rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-xs font-semibold text-white hover:bg-slate-700"
                >
                  {cameraActive ? 'Stop Camera' : 'Start Camera'}
                </button>
                <button
                  type="button"
                  disabled={!cameraActive || loading}
                  onClick={handleCaptureWebcam}
                  className="flex-1 rounded-lg bg-cyan-400 px-3 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-300 disabled:opacity-50"
                >
                  Capture &amp; Recognize
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="space-y-3 lg:col-span-4">
          <div>
            <label className="flex justify-between text-xs text-slate-300">
              <span>Minimum score threshold</span>
              <span className="font-mono">{(minimumScore * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={minimumScore}
              onChange={(e) => setMinimumScore(Number(e.target.value))}
              className="mt-1.5 w-full accent-cyan-400"
            />
          </div>

          <div className="rounded-xl border border-slate-800 bg-[#07121a] p-3 text-xs">
            <label className="flex cursor-pointer items-center gap-2 font-medium text-slate-200">
              <input
                type="checkbox"
                checked={enforceWindow}
                onChange={(e) => setEnforceWindow(e.target.checked)}
                className="h-4 w-4 accent-cyan-400"
              />
              <Clock className="h-3.5 w-3.5 text-cyan-300" />
              Enforce Operating Window (America/New_York)
            </label>
            <div className="mt-2 flex items-center gap-2">
              <input
                type="time"
                value={allowedStart}
                onChange={(e) => setAllowedStart(e.target.value)}
                className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-100"
              />
              <span className="text-slate-400">to</span>
              <input
                type="time"
                value={allowedEnd}
                onChange={(e) => setAllowedEnd(e.target.value)}
                className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-100"
              />
            </div>
          </div>
        </div>

        <div className="flex items-end lg:col-span-3">
          {mode === 'image' && (
            <button
              type="button"
              disabled={loading || !sourcePreview}
              onClick={() => sourcePreview && runRecognition(sourcePreview)}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-cyan-400/50 bg-cyan-500/20 px-4 py-3 text-sm font-semibold text-cyan-100 hover:bg-cyan-500/30 disabled:opacity-50"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {loading ? 'Recognizing...' : 'Recognize Letter'}
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/40 p-4 text-sm text-rose-200">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="grid gap-5 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5 md:grid-cols-12">
          <div className="space-y-3 md:col-span-6">
            {annotatedPreview && (
              <img
                src={annotatedPreview}
                alt="Annotated ASL hand gesture"
                className="max-h-80 w-full rounded-xl border border-slate-700 bg-[#07121a] object-contain"
              />
            )}
            {annotatedPreview && (
              <a
                href={annotatedPreview}
                download="asl_recognized.png"
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-600 bg-slate-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-700"
              >
                <Download className="h-3.5 w-3.5" /> Download Annotated PNG
              </a>
            )}
          </div>

          <div className="space-y-4 md:col-span-6">
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl border border-emerald-500/35 bg-[#07121a] p-4">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">Recognized Letter</div>
                <div className="mt-1 font-mono text-4xl font-bold text-emerald-300">
                  {result.status === 'recognized' && result.label ? result.label : 'Unknown'}
                </div>
                <div className="mt-1 text-xs capitalize text-slate-400">
                  Status: {result.status.replace('_', ' ')}
                </div>
              </div>
              <div className="rounded-xl border border-slate-700 bg-[#07121a] p-4">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">Model Score</div>
                <div className="mt-1 text-3xl font-bold text-cyan-200">
                  {(result.confidence * 100).toFixed(1)}%
                </div>
                <div className="mt-1 text-xs text-slate-400">
                  NY Time: {result.windowStatus.localTime}
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-[#07121a] p-3.5 text-xs text-slate-300">
              <strong className="text-slate-100">Hand Posture Observed:</strong> {result.handDescription}
            </div>

            {result.warning && (
              <div className="rounded-xl border border-amber-500/40 bg-amber-950/30 p-3 text-xs text-amber-200">
                {result.warning}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Vocabulary Grid */}
      <div className="rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white">
            Supported 24-Letter Static ASL Vocabulary (J &amp; Z excluded)
          </h2>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {SIGN_LANGUAGE_VOCABULARY_JSON.labels.map((letter) => {
            const matched = result?.status === 'recognized' && result.label === letter;
            return (
              <span
                key={letter}
                className={`flex h-9 w-9 items-center justify-center rounded-lg border font-mono text-xs font-bold transition ${
                  matched
                    ? 'border-emerald-400 bg-emerald-500/30 text-emerald-100 ring-2 ring-emerald-400/50'
                    : 'border-slate-700 bg-[#07121a] text-cyan-200'
                }`}
              >
                {letter}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
