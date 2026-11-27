import { useState } from 'react';
import { Download, Film, Image as ImageIcon, Loader2, PawPrint, Play, Sparkles, Upload } from 'lucide-react';
import sampleWildlifeImg from '../../assets/images/sample_wildlife_1795784436867.jpg';
import { WILDLIFE_TAXONOMY_JSON } from '../../data/datasets';
import {
  annotateImageWithBoxes,
  extractVideoKeyFrames,
  type AnnotationBox,
} from '../../utils/mediaUtils';
import { runWildlifeLocal } from '../../ml/localEngine';

interface WildlifeDetection {
  id: string;
  species: string;
  confidence: number;
  box: [number, number, number, number];
  classification: 'carnivore' | 'non_carnivore' | 'unknown';
  note: string;
}

interface WildlifeResponse {
  summary: {
    totalAnimals: number;
    carnivores: number;
    nonCarnivores: number;
    unknown: number;
    speciesCounts: Record<string, number>;
  };
  detections: WildlifeDetection[];
  metadata: {
    model: string;
    taxonomyVersion: string;
    confidenceThreshold: number;
    inferenceMs: number;
  };
}

function buildWildlifeAnnotations(detections: WildlifeDetection[]): AnnotationBox[] {
  return detections.map((det) => ({
    box: det.box,
    color:
      det.classification === 'carnivore'
        ? '#ff3b30'
        : det.classification === 'non_carnivore'
          ? '#1d9bf0'
          : '#94a3b8',
    label: `${det.species.toUpperCase()} (${det.classification.replace('_', '-')}) ${(det.confidence * 100).toFixed(0)}%`,
  }));
}

export function WildlifeTask() {
  const [mode, setMode] = useState<'image' | 'video'>('image');
  const [confidence, setConfidence] = useState(0.25);
  const [sourcePreview, setSourcePreview] = useState<string | null>(null);
  const [annotatedPreview, setAnnotatedPreview] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [result, setResult] = useState<WildlifeResponse | null>(null);
  const [videoFrames, setVideoFrames] = useState<
    Array<{ frameIndex: number; timestampSeconds: number; annotatedUrl: string; result: WildlifeResponse }>
  >([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runImage = async (previewUrl: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setVideoFrames([]);
    try {
      const data = await runWildlifeLocal(previewUrl, confidence);
      const annotated = await annotateImageWithBoxes(previewUrl, buildWildlifeAnnotations(data.detections));
      setSourcePreview(previewUrl);
      setAnnotatedPreview(annotated);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Wildlife detection failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleRunSample = async () => {
    setMode('image');
    setVideoFile(null);
    setSourcePreview(sampleWildlifeImg);
    await runImage(sampleWildlifeImg);
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
        setError('Upload an animal image or click "Run Test Sample" first.');
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
        const data = await runWildlifeLocal(frame.previewUrl, confidence);
        const annotatedUrl = await annotateImageWithBoxes(
          frame.previewUrl,
          buildWildlifeAnnotations(data.detections),
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

  const allDetections = result ? result.detections : videoFrames.flatMap((f) => f.result.detections);
  const carnivoreCount = allDetections.filter((d) => d.classification === 'carnivore').length;
  const nonCarnivoreCount = allDetections.filter((d) => d.classification === 'non_carnivore').length;
  const unknownCount = allDetections.filter((d) => d.classification === 'unknown').length;

  return (
    <div className="space-y-5">
      <div className="flex flex-col justify-between gap-4 rounded-2xl border border-emerald-700/40 bg-gradient-to-br from-[#15271e] via-[#1c3828] to-[#13261c] p-6 sm:flex-row sm:items-center">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.15em] text-emerald-300">
            Task 2 · Wildlife Risk Intelligence
          </div>
          <h1 className="mt-1 text-2xl font-bold text-white">Animal Species &amp; Carnivore Detection</h1>
          <p className="mt-1 text-sm text-emerald-100/80">
            Detects animal species, maps diet via versioned taxonomy v{WILDLIFE_TAXONOMY_JSON.version},
            and highlights carnivorous animals in <strong className="text-rose-300">RED</strong>.
          </p>
        </div>
        <button
          type="button"
          disabled={loading}
          onClick={handleRunSample}
          className="inline-flex shrink-0 items-center gap-2 rounded-xl bg-emerald-400 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:opacity-50"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
          Run Test Sample
        </button>
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
                  ? 'border-emerald-400 bg-emerald-500/20 text-white'
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
                  ? 'border-emerald-400 bg-emerald-500/20 text-white'
                  : 'border-slate-700 bg-[#07121a] text-slate-300'
              }`}
            >
              <Film className="h-3.5 w-3.5" /> Video
            </button>
          </div>
          <label className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-dashed border-slate-600 bg-[#07121a] px-4 py-3 text-xs font-medium text-slate-200 hover:border-emerald-400">
            <Upload className="h-4 w-4 text-emerald-300" />
            {mode === 'image' ? 'Upload animal image' : 'Upload animal video'}
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
            className="mt-2 w-full accent-emerald-400"
          />
        </div>

        <div className="flex items-end lg:col-span-3">
          <button
            type="button"
            disabled={loading}
            onClick={handleAnalyze}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-emerald-400/50 bg-emerald-500/20 px-4 py-3 text-sm font-semibold text-emerald-100 hover:bg-emerald-500/30 disabled:opacity-50"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {loading ? 'Analyzing...' : 'Analyze Wildlife'}
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
          <div className="flex items-center justify-between rounded-xl border border-rose-500/45 bg-rose-950/35 px-5 py-3.5">
            <div className="flex items-center gap-2.5 text-sm font-semibold text-rose-100">
              <PawPrint className="h-5 w-5 text-rose-400" />
              Carnivores Detected: <span className="text-lg font-bold text-rose-300">{carnivoreCount}</span>
              <span className="text-xs font-normal text-rose-200/80">
                {videoFrames.length > 0 ? 'detections across sampled frames' : 'in this image'}
              </span>
            </div>
            <div className="text-xs text-slate-300">
              Total animals: <strong>{allDetections.length}</strong> · Non-carnivores:{' '}
              <strong>{nonCarnivoreCount}</strong> · Unknown: <strong>{unknownCount}</strong>
            </div>
          </div>

          {result && sourcePreview && annotatedPreview && (
            <div className="space-y-4 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex gap-4 text-xs text-slate-300">
                  <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-xs bg-[#ff3b30]" />Carnivore (RED)</span>
                  <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-xs bg-[#1d9bf0]" />Non-carnivore (BLUE)</span>
                  <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-xs bg-[#94a3b8]" />Unknown diet (GREY)</span>
                </div>
                <a
                  href={annotatedPreview}
                  download="wildlife_annotated.png"
                  className="inline-flex items-center gap-1.5 rounded-lg border border-slate-600 bg-slate-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-700"
                >
                  <Download className="h-3.5 w-3.5" /> Download Annotated PNG
                </a>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <img src={sourcePreview} alt="Original wildlife image" className="w-full rounded-xl border border-slate-800" />
                <img src={annotatedPreview} alt="Annotated wildlife detections" className="w-full rounded-xl border border-emerald-500/30" />
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
                      {f.result.summary.totalAnimals} animals ({f.result.summary.carnivores} carnivores)
                    </span>
                  </div>
                  <img src={f.annotatedUrl} alt={`Wildlife frame ${f.frameIndex}`} className="w-full rounded-lg" />
                </div>
              ))}
            </div>
          )}

          <div className="rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
            <h2 className="mb-3 text-sm font-semibold text-white">Detected Animals &amp; Dietary Classification</h2>
            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-900/90 text-slate-400">
                  <tr>
                    <th className="px-3 py-2">ID</th>
                    <th className="px-3 py-2">Species</th>
                    <th className="px-3 py-2">Diet Category</th>
                    <th className="px-3 py-2">Confidence</th>
                    <th className="px-3 py-2">Taxonomy Note</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 text-slate-200">
                  {allDetections.map((det, idx) => (
                    <tr key={`${det.id}-${idx}`}>
                      <td className="px-3 py-2 font-mono">{det.id}</td>
                      <td className="px-3 py-2 font-semibold capitalize">{det.species}</td>
                      <td className="px-3 py-2">
                        <span
                          className={`rounded px-2 py-0.5 text-[11px] font-semibold ${
                            det.classification === 'carnivore'
                              ? 'bg-rose-500/20 text-rose-300'
                              : det.classification === 'non_carnivore'
                                ? 'bg-sky-500/20 text-sky-300'
                                : 'bg-slate-700 text-slate-300'
                          }`}
                        >
                          {det.classification.replace('_', '-')}
                        </span>
                      </td>
                      <td className="px-3 py-2 font-mono">{(det.confidence * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2 text-slate-400">{det.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      <details className="rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
        <summary className="cursor-pointer text-sm font-semibold text-slate-200">
          Reference Dietary Taxonomy (`taxonomy.json` v{WILDLIFE_TAXONOMY_JSON.version})
        </summary>
        <div className="mt-3 grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-5">
          {Object.entries(WILDLIFE_TAXONOMY_JSON.classifications).map(([species, cat]) => (
            <div key={species} className="rounded-lg border border-slate-800 bg-[#07121a] p-2.5">
              <div className="font-mono font-semibold capitalize text-white">{species}</div>
              <div className="mt-0.5 text-[11px] text-slate-400">{cat.replace('_', '-')}</div>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}
