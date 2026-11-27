import { useState } from 'react';
import { Download, Film, Image as ImageIcon, Loader2, Play, Sparkles, Upload, UserSearch } from 'lucide-react';
import samplePersonImg from '../../assets/images/sample_person_1795784439520.jpg';
import {
  annotateImageWithBoxes,
  extractVideoKeyFrames,
  type AnnotationBox,
} from '../../utils/mediaUtils';
import { runPersonAnalysisLocal } from '../../ml/localEngine';

interface PersonResultItem {
  id: string;
  confidence: number;
  box: [number, number, number, number];
  coarseAgeRange: string;
  expressionEstimate: string;
  expressionConfidence: number;
  visibleAttributes: string[];
}

interface PersonAnalysisResponse {
  personCount: number;
  people: PersonResultItem[];
  userProvidedRegion: string | null;
  metadata: {
    model: string;
    confidenceThreshold: number;
    nationalityInferredFromAppearance: boolean;
    inferenceMs: number;
  };
}

function buildPersonAnnotations(people: PersonResultItem[]): AnnotationBox[] {
  return people.map((p) => ({
    box: p.box,
    color: '#38bdf8',
    label: `${p.id} · Age ${p.coarseAgeRange} · ${p.expressionEstimate}`,
  }));
}

export function PersonAnalysisTask() {
  const [mode, setMode] = useState<'image' | 'video'>('image');
  const [confidence, setConfidence] = useState(0.25);
  const [userRegion, setUserRegion] = useState('');
  const [sourcePreview, setSourcePreview] = useState<string | null>(null);
  const [annotatedPreview, setAnnotatedPreview] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [result, setResult] = useState<PersonAnalysisResponse | null>(null);
  const [videoFrames, setVideoFrames] = useState<
    Array<{ frameIndex: number; timestampSeconds: number; annotatedUrl: string; result: PersonAnalysisResponse }>
  >([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runImage = async (previewUrl: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setVideoFrames([]);
    try {
      const data = await runPersonAnalysisLocal(previewUrl, confidence, userRegion);
      const annotated = await annotateImageWithBoxes(previewUrl, buildPersonAnnotations(data.people));
      setSourcePreview(previewUrl);
      setAnnotatedPreview(annotated);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Person analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleRunSample = async () => {
    setMode('image');
    setVideoFile(null);
    setSourcePreview(samplePersonImg);
    await runImage(samplePersonImg);
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
        setError('Upload an image or click "Run Test Sample" first.');
        return;
      }
      await runImage(sourcePreview);
      return;
    }
    if (!videoFile) {
      setError('Upload a video file first.');
      return;
    }
    setLoading(true);
    setError(null);
    setVideoFrames([]);
    try {
      const frames = await extractVideoKeyFrames(videoFile, 4);
      const out = [];
      for (const frame of frames) {
        const data = await runPersonAnalysisLocal(frame.previewUrl, confidence, userRegion);
        const annotatedUrl = await annotateImageWithBoxes(
          frame.previewUrl,
          buildPersonAnnotations(data.people),
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
      setError(err instanceof Error ? err.message : 'Video analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  const allPeople = result ? result.people : videoFrames.flatMap((f) => f.result.people);

  return (
    <div className="space-y-5">
      <div className="flex flex-col justify-between gap-4 rounded-2xl border border-slate-700/60 bg-gradient-to-br from-[#0d2535] to-[#0a1822] p-6 sm:flex-row sm:items-center">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.15em] text-cyan-300">
            Task 4 · Person Detection &amp; Visual Attributes
          </div>
          <h1 className="mt-1 flex items-center gap-2 text-2xl font-bold text-white">
            <UserSearch className="h-6 w-6 text-cyan-300" /> Person Attribute &amp; Expression Analysis
          </h1>
          <p className="mt-1 text-sm text-slate-300">
            Detects people, estimates coarse age ranges, visible facial expressions, and non-sensitive
            visible attributes. Never infers nationality or ethnicity from appearance.
          </p>
        </div>
        <button
          type="button"
          disabled={loading}
          onClick={handleRunSample}
          className="inline-flex shrink-0 items-center gap-2 rounded-xl bg-cyan-400 px-4 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-300 disabled:opacity-50"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
          Run Test Sample
        </button>
      </div>

      {/* Controls */}
      <div className="grid gap-4 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5 lg:grid-cols-12">
        <div className="space-y-3 lg:col-span-4">
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
            {mode === 'image' ? 'Upload person image' : 'Upload video clip'}
            <input
              type="file"
              accept={mode === 'image' ? 'image/jpeg,image/png,image/webp' : 'video/mp4,video/webm,video/quicktime'}
              className="hidden"
              onChange={(e) => handleFileChange(e.target.files?.[0])}
            />
          </label>
        </div>

        <div className="space-y-3 lg:col-span-5">
          <div>
            <label className="flex justify-between text-xs text-slate-300">
              <span>Detection confidence</span>
              <span className="font-mono">{confidence.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(Number(e.target.value))}
              className="mt-1.5 w-full accent-cyan-400"
            />
          </div>
          <div>
            <label htmlFor="explicit-region" className="block text-xs text-slate-300">
              Optional Explicit Region / Nationality (user-provided only, never inferred)
            </label>
            <input
              id="explicit-region"
              type="text"
              value={userRegion}
              onChange={(e) => setUserRegion(e.target.value)}
              placeholder="e.g., India, United States (leave blank if unused)"
              className="mt-1 w-full rounded-lg border border-slate-700 bg-[#07121a] px-3 py-1.5 text-xs text-slate-100 placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none"
            />
          </div>
        </div>

        <div className="flex items-end lg:col-span-3">
          <button
            type="button"
            disabled={loading}
            onClick={handleAnalyze}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-cyan-400/50 bg-cyan-500/20 px-4 py-3 text-sm font-semibold text-cyan-100 hover:bg-cyan-500/30 disabled:opacity-50"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {loading ? 'Analyzing...' : 'Analyze People'}
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/40 p-4 text-sm text-rose-200">
          {error}
        </div>
      )}

      {result && sourcePreview && annotatedPreview && (
        <div className="space-y-4 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-sm font-semibold text-white">
              People Detected: <span className="text-cyan-300">{result.personCount}</span>
              {result.userProvidedRegion && (
                <span className="ml-3 rounded-md border border-slate-700 bg-slate-800 px-2 py-0.5 text-xs font-normal text-slate-300">
                  User-provided region: {result.userProvidedRegion}
                </span>
              )}
            </div>
            <a
              href={annotatedPreview}
              download="person_analysis_annotated.png"
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-600 bg-slate-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-700"
            >
              <Download className="h-3.5 w-3.5" /> Download Annotated PNG
            </a>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <img src={sourcePreview} alt="Original portrait" className="w-full rounded-xl border border-slate-800" />
            <img src={annotatedPreview} alt="Annotated people" className="w-full rounded-xl border border-cyan-500/30" />
          </div>
        </div>
      )}

      {videoFrames.length > 0 && (
        <div className="grid gap-4 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5 md:grid-cols-2">
          {videoFrames.map((f) => (
            <div key={f.frameIndex} className="rounded-xl border border-slate-700 bg-[#07121a] p-3">
              <div className="mb-2 flex justify-between text-xs text-slate-300">
                <span>Frame #{f.frameIndex} ({f.timestampSeconds}s)</span>
                <span>{f.result.personCount} people detected</span>
              </div>
              <img src={f.annotatedUrl} alt={`Person frame ${f.frameIndex}`} className="w-full rounded-lg" />
            </div>
          ))}
        </div>
      )}

      {(result || videoFrames.length > 0) && (
        <div className="rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
          <h2 className="mb-3 text-sm font-semibold text-white">Per-Person Analysis Results</h2>
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900/90 text-slate-400">
                <tr>
                  <th className="px-3 py-2">Person</th>
                  <th className="px-3 py-2">Confidence</th>
                  <th className="px-3 py-2">Coarse Age Estimate</th>
                  <th className="px-3 py-2">Visible Expression Estimate</th>
                  <th className="px-3 py-2">Non-Sensitive Visible Attributes</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-slate-200">
                {allPeople.map((p, idx) => (
                  <tr key={`${p.id}-${idx}`}>
                    <td className="px-3 py-2 font-mono">{p.id}</td>
                    <td className="px-3 py-2 font-mono">{(p.confidence * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2">{p.coarseAgeRange}</td>
                    <td className="px-3 py-2 capitalize">
                      {p.expressionEstimate}{' '}
                      <span className="text-slate-400">({(p.expressionConfidence * 100).toFixed(0)}%)</span>
                    </td>
                    <td className="px-3 py-2 text-slate-300">
                      {p.visibleAttributes.length > 0 ? p.visibleAttributes.join(', ') : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2.5 text-[11px] text-slate-400">
            Facial expression estimates reflect visible expression cues only, not internal emotional
            state. Age ranges are coarse estimates with inherent uncertainty.
          </p>
        </div>
      )}
    </div>
  );
}
