import { useState } from 'react';
import {
  AlertTriangle,
  Download,
  Eye,
  Film,
  Image as ImageIcon,
  Loader2,
  Play,
  Sparkles,
  Upload,
} from 'lucide-react';
import sampleDrowsyImg from '../../assets/images/sample_drowsy_1795784441301.jpg';
import {
  annotateImageWithBoxes,
  extractVideoKeyFrames,
  type AnnotationBox,
} from '../../utils/mediaUtils';
import { runDrowsinessLocal } from '../../ml/localEngine';

interface OccupantResult {
  id: string;
  roleHint: string;
  confidence: number;
  box: [number, number, number, number];
  drowsinessStatus: 'alert' | 'drowsy' | 'sleeping' | 'unknown';
  statusConfidence: number;
  coarseAgeRange: string;
  visibleCues: string;
}

interface DrowsinessResponse {
  personCount: number;
  sleepingDrowsyCount: number;
  alertCount: number;
  unknownCount: number;
  people: OccupantResult[];
  limitationNotice: string;
  metadata: {
    model: string;
    confidenceThreshold: number;
    inferenceMs: number;
  };
}

function buildDrowsinessAnnotations(people: OccupantResult[]): AnnotationBox[] {
  return people.map((p) => {
    const isRisk = p.drowsinessStatus === 'drowsy' || p.drowsinessStatus === 'sleeping';
    const color = isRisk ? '#ff3b30' : p.drowsinessStatus === 'alert' ? '#10b981' : '#94a3b8';
    return {
      box: p.box,
      color,
      label: `${p.roleHint.toUpperCase()} · ${p.drowsinessStatus.toUpperCase()} (${(p.statusConfidence * 100).toFixed(0)}%)`,
    };
  });
}

export function DrowsinessTask() {
  const [mode, setMode] = useState<'image' | 'video'>('image');
  const [confidence, setConfidence] = useState(0.25);
  const [sourcePreview, setSourcePreview] = useState<string | null>(null);
  const [annotatedPreview, setAnnotatedPreview] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [result, setResult] = useState<DrowsinessResponse | null>(null);
  const [videoFrames, setVideoFrames] = useState<
    Array<{ frameIndex: number; timestampSeconds: number; annotatedUrl: string; result: DrowsinessResponse }>
  >([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runImage = async (previewUrl: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setVideoFrames([]);
    try {
      const data = await runDrowsinessLocal(previewUrl, confidence, false);
      const annotated = await annotateImageWithBoxes(previewUrl, buildDrowsinessAnnotations(data.people));
      setSourcePreview(previewUrl);
      setAnnotatedPreview(annotated);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Drowsiness detection failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleRunSample = async () => {
    setMode('image');
    setVideoFile(null);
    setSourcePreview(sampleDrowsyImg);
    await runImage(sampleDrowsyImg);
  };

  const handleFileChange = (file: File | undefined) => {
    if (!file) return;
    setError(null);
    setResult(null);
    setAnnotatedPreview(null);
    setVideoFrames([]);
    const preview = URL.createObjectURL(file);
    setSourcePreview(preview);
    if (mode === 'video') {
      setVideoFile(file);
    }
  };

  const handleAnalyze = async () => {
    if (mode === 'image') {
      if (!sourcePreview) {
        setError('Upload a cabin image or click "Run Test Sample" first.');
        return;
      }
      await runImage(sourcePreview);
      return;
    }
    if (!videoFile) {
      setError('Upload a vehicle cabin video file first.');
      return;
    }
    setLoading(true);
    setError(null);
    setVideoFrames([]);
    try {
      const frames = await extractVideoKeyFrames(videoFile, 4);
      const out = [];
      for (const frame of frames) {
        const data = await runDrowsinessLocal(frame.previewUrl, confidence, true);
        const annotatedUrl = await annotateImageWithBoxes(
          frame.previewUrl,
          buildDrowsinessAnnotations(data.people),
        );
        out.push({
          frameIndex: frame.frameIndex,
          timestampSeconds: frame.timestampSeconds,
          annotatedUrl,
          result: data,
        });
      }
      setVideoFrames(out);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Video drowsiness analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  const peakPeople = result
    ? result.personCount
    : Math.max(0, ...videoFrames.map((f) => f.result.personCount));
  const peakDrowsy = result
    ? result.sleepingDrowsyCount
    : Math.max(0, ...videoFrames.map((f) => f.result.sleepingDrowsyCount));
  const drowsyFrames = videoFrames.filter((f) => f.result.sleepingDrowsyCount > 0).length;
  const activePeople = result ? result.people : videoFrames.flatMap((f) => f.result.people);

  return (
    <div className="space-y-5">
      <div className="flex flex-col justify-between gap-4 rounded-2xl border border-slate-700/60 bg-gradient-to-br from-[#0d2535] to-[#0a1822] p-6 sm:flex-row sm:items-center">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.15em] text-cyan-300">
            Task 6 · Driver &amp; Occupant Monitoring
          </div>
          <h1 className="mt-1 flex items-center gap-2 text-2xl font-bold text-white">
            <Eye className="h-6 w-6 text-cyan-300" /> Driver Drowsiness Detection
          </h1>
          <p className="mt-1 text-sm text-slate-300">
            Detects vehicle occupants, flags visible sleeping/drowsy cues with{' '}
            <strong className="text-rose-300">RED</strong> boxes, and reports coarse age estimates.
          </p>
        </div>
        <button
          type="button"
          disabled={loading}
          onClick={handleRunSample}
          className="inline-flex shrink-0 items-center gap-2 rounded-xl bg-rose-500 px-4 py-2.5 text-sm font-semibold text-white hover:bg-rose-400 disabled:opacity-50"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
          Run Test Sample
        </button>
      </div>

      {/* Safety Notice */}
      <div className="flex items-start gap-3 rounded-xl border border-amber-500/35 bg-amber-950/25 p-3.5 text-xs text-amber-200">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />
        <span>
          Experimental computer-vision aid only — not a substitute for driver safety systems, rest,
          or medical assessment. Single-frame images evaluate visible sleep-like cues only; use video
          clips for multi-frame temporal context.
        </span>
      </div>

      {/* Controls */}
      <div className="grid gap-4 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5 lg:grid-cols-12">
        <div className="space-y-3 lg:col-span-5">
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => {
                setMode('image');
                setVideoFrames([]);
              }}
              className={`flex flex-1 items-center justify-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-semibold ${
                mode === 'image'
                  ? 'border-cyan-400 bg-cyan-500/20 text-white'
                  : 'border-slate-700 bg-[#07121a] text-slate-300'
              }`}
            >
              <ImageIcon className="h-3.5 w-3.5" /> Image
            </button>
            <button
              type="button"
              onClick={() => {
                setMode('video');
                setResult(null);
                setAnnotatedPreview(null);
              }}
              className={`flex flex-1 items-center justify-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-semibold ${
                mode === 'video'
                  ? 'border-cyan-400 bg-cyan-500/20 text-white'
                  : 'border-slate-700 bg-[#07121a] text-slate-300'
              }`}
            >
              <Film className="h-3.5 w-3.5" /> Video
            </button>
          </div>
          <label className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-dashed border-slate-600 bg-[#07121a] px-4 py-3 text-xs font-medium text-slate-200 hover:border-cyan-400">
            <Upload className="h-4 w-4 text-cyan-300" />
            {mode === 'image' ? 'Upload vehicle-interior image' : 'Upload vehicle-interior video'}
            <input
              type="file"
              accept={mode === 'image' ? 'image/jpeg,image/png,image/webp' : 'video/mp4,video/webm,video/quicktime'}
              className="hidden"
              onChange={(e) => handleFileChange(e.target.files?.[0])}
            />
          </label>
        </div>

        <div className="flex flex-col justify-center lg:col-span-4">
          <label className="flex justify-between text-xs text-slate-300">
            <span>Person detection threshold</span>
            <span className="font-mono">{confidence.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={confidence}
            onChange={(e) => setConfidence(Number(e.target.value))}
            className="mt-2 w-full accent-cyan-400"
          />
        </div>

        <div className="flex items-end lg:col-span-3">
          <button
            type="button"
            disabled={loading}
            onClick={handleAnalyze}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-cyan-400/50 bg-cyan-500/20 px-4 py-3 text-sm font-semibold text-cyan-100 hover:bg-cyan-500/30 disabled:opacity-50"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {loading ? 'Analyzing...' : 'Analyze Occupants'}
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/40 p-4 text-sm text-rose-200">
          {error}
        </div>
      )}

      {(result || videoFrames.length > 0) && (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-slate-700 bg-[#0d1c28] p-4">
              <div className="text-[11px] uppercase tracking-wider text-slate-400">
                {videoFrames.length > 0 ? 'Peak Occupants / Frame' : 'Detected People'}
              </div>
              <div className="mt-1 text-2xl font-bold text-white">{peakPeople}</div>
            </div>
            <div
              className={`rounded-xl border p-4 ${
                peakDrowsy > 0
                  ? 'border-rose-500/50 bg-rose-950/35'
                  : 'border-emerald-500/35 bg-emerald-950/20'
              }`}
            >
              <div className="text-[11px] uppercase tracking-wider text-rose-300">
                Sleeping / Drowsy People (RED Box)
              </div>
              <div className="mt-1 text-2xl font-bold text-rose-200">{peakDrowsy}</div>
            </div>
            <div className="rounded-xl border border-slate-700 bg-[#0d1c28] p-4">
              <div className="text-[11px] uppercase tracking-wider text-slate-400">
                {videoFrames.length > 0 ? 'Drowsy Frames Detected' : 'Alert Occupants'}
              </div>
              <div className="mt-1 text-2xl font-bold text-emerald-300">
                {videoFrames.length > 0 ? `${drowsyFrames} / ${videoFrames.length}` : result?.alertCount ?? 0}
              </div>
            </div>
          </div>

          {result && sourcePreview && annotatedPreview && (
            <div className="space-y-4 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex gap-4 text-xs text-slate-300">
                  <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-xs bg-[#ff3b30]" />Drowsy / Sleeping (RED)</span>
                  <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-xs bg-[#10b981]" />Alert (GREEN)</span>
                  <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-xs bg-[#94a3b8]" />Unknown</span>
                </div>
                <a
                  href={annotatedPreview}
                  download="drowsiness_annotated.png"
                  className="inline-flex items-center gap-1.5 rounded-lg border border-slate-600 bg-slate-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-700"
                >
                  <Download className="h-3.5 w-3.5" /> Download Annotated PNG
                </a>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <img src={sourcePreview} alt="Original cabin image" className="w-full rounded-xl border border-slate-800" />
                <img src={annotatedPreview} alt="Annotated drowsiness detections" className="w-full rounded-xl border border-rose-500/35" />
              </div>
            </div>
          )}

          {videoFrames.length > 0 && (
            <div className="grid gap-4 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5 md:grid-cols-2">
              {videoFrames.map((f) => (
                <div key={f.frameIndex} className="rounded-xl border border-slate-700 bg-[#07121a] p-3">
                  <div className="mb-2 flex justify-between text-xs text-slate-300">
                    <span>Frame #{f.frameIndex} ({f.timestampSeconds}s)</span>
                    <span>
                      {f.result.personCount} occupants · {f.result.sleepingDrowsyCount} drowsy/sleeping
                    </span>
                  </div>
                  <img src={f.annotatedUrl} alt={`Cabin frame ${f.frameIndex}`} className="w-full rounded-lg" />
                </div>
              ))}
            </div>
          )}

          <div className="rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
            <h2 className="mb-3 text-sm font-semibold text-white">Per-Occupant Drowsiness Details</h2>
            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-900/90 text-slate-400">
                  <tr>
                    <th className="px-3 py-2">ID</th>
                    <th className="px-3 py-2">Role</th>
                    <th className="px-3 py-2">Status</th>
                    <th className="px-3 py-2">Score</th>
                    <th className="px-3 py-2">Coarse Age</th>
                    <th className="px-3 py-2">Visible Cues</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 text-slate-200">
                  {activePeople.map((p, idx) => {
                    const isRisk = p.drowsinessStatus === 'drowsy' || p.drowsinessStatus === 'sleeping';
                    return (
                      <tr key={`${p.id}-${idx}`}>
                        <td className="px-3 py-2 font-mono">{p.id}</td>
                        <td className="px-3 py-2 capitalize">{p.roleHint}</td>
                        <td className="px-3 py-2">
                          <span
                            className={`rounded px-2 py-0.5 text-[11px] font-semibold uppercase ${
                              isRisk
                                ? 'bg-rose-500/25 text-rose-300'
                                : p.drowsinessStatus === 'alert'
                                  ? 'bg-emerald-500/20 text-emerald-300'
                                  : 'bg-slate-700 text-slate-300'
                            }`}
                          >
                            {p.drowsinessStatus}
                          </span>
                        </td>
                        <td className="px-3 py-2 font-mono">{(p.statusConfidence * 100).toFixed(1)}%</td>
                        <td className="px-3 py-2">{p.coarseAgeRange}</td>
                        <td className="px-3 py-2 text-slate-300">{p.visibleCues}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
