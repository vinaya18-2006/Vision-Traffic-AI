import { useState } from 'react';
import { Car, Download, Film, Image as ImageIcon, Loader2, Play, Sparkles, Upload } from 'lucide-react';
import sampleTrafficImg from '../../assets/images/sample_traffic_1795784436820.jpg';
import {
  annotateImageWithBoxes,
  extractVideoKeyFrames,
  type AnnotationBox,
} from '../../utils/mediaUtils';
import { runCarDetectionLocal } from '../../ml/localEngine';

interface CarDetectionItem {
  id: string;
  className: 'car';
  confidence: number;
  box: [number, number, number, number];
  colour: string;
  colourEvidence: number;
  boxStyle: 'red' | 'blue' | 'gray';
}

interface PersonDetectionItem {
  id: string;
  className: 'person';
  confidence: number;
  box: [number, number, number, number];
}

interface CarApiResponse {
  carCount: number;
  blueCarCount: number;
  otherKnownColourCarCount: number;
  unknownColourCarCount: number;
  personCount: number;
  cars: CarDetectionItem[];
  people: PersonDetectionItem[];
  sceneSummary: string;
  metadata: {
    model: string;
    confidenceThreshold: number;
    colourThreshold: number;
    inferenceMs: number;
  };
}

interface FrameAnalysis {
  frameIndex: number;
  timestampSeconds: number;
  annotatedUrl: string;
  result: CarApiResponse;
}

function buildCarAnnotations(result: CarApiResponse): AnnotationBox[] {
  const boxes: AnnotationBox[] = [];
  for (const car of result.cars) {
    const color =
      car.boxStyle === 'red' ? '#ff3b30' : car.boxStyle === 'blue' ? '#1d9bf0' : '#94a3b8';
    boxes.push({
      box: car.box,
      color,
      label: `${car.colour.toUpperCase()} CAR ${(car.confidence * 100).toFixed(0)}%`,
    });
  }
  for (const person of result.people) {
    boxes.push({
      box: person.box,
      color: '#10b981',
      label: `PERSON ${(person.confidence * 100).toFixed(0)}%`,
    });
  }
  return boxes;
}

export function CarDetectionTask() {
  const [mode, setMode] = useState<'image' | 'video'>('image');
  const [confidence, setConfidence] = useState(0.25);
  const [colourThreshold, setColourThreshold] = useState(0.15);
  const [sourcePreview, setSourcePreview] = useState<string | null>(null);
  const [annotatedPreview, setAnnotatedPreview] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [result, setResult] = useState<CarApiResponse | null>(null);
  const [videoFrames, setVideoFrames] = useState<FrameAnalysis[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runImageInference = async (previewUrl: string) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setVideoFrames([]);
    try {
      const data = await runCarDetectionLocal(previewUrl, confidence, colourThreshold);
      const annotated = await annotateImageWithBoxes(previewUrl, buildCarAnnotations(data));
      setSourcePreview(previewUrl);
      setAnnotatedPreview(annotated);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Car detection failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadAndRunSample = async () => {
    setMode('image');
    setVideoFile(null);
    setSourcePreview(sampleTrafficImg);
    await runImageInference(sampleTrafficImg);
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

  const handleAnalyzeClick = async () => {
    if (mode === 'image') {
      if (!sourcePreview) {
        setError('Upload a traffic image or click "Run Test Sample" first.');
        return;
      }
      await runImageInference(sourcePreview);
      return;
    }

    if (!videoFile) {
      setError('Upload an MP4/WebM video file first.');
      return;
    }
    setLoading(true);
    setError(null);
    setVideoFrames([]);
    try {
      const extracted = await extractVideoKeyFrames(videoFile, 4);
      const analyzed: FrameAnalysis[] = [];
      for (const frame of extracted) {
        const data = await runCarDetectionLocal(frame.previewUrl, confidence, colourThreshold);
        const annotatedUrl = await annotateImageWithBoxes(frame.previewUrl, buildCarAnnotations(data));
        analyzed.push({
          frameIndex: frame.frameIndex,
          timestampSeconds: frame.timestampSeconds,
          annotatedUrl,
          result: data,
        });
      }
      setVideoFrames(analyzed);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Video analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  const activeCars = result ? result.cars : videoFrames.flatMap((f) => f.result.cars);
  const activePeople = result ? result.people : videoFrames.flatMap((f) => f.result.people);
  const blueCount = activeCars.filter((c) => c.colour === 'blue').length;
  const otherKnownCount = activeCars.filter((c) => c.colour !== 'blue' && c.colour !== 'unknown').length;
  const avgConf =
    activeCars.length + activePeople.length > 0
      ? ([...activeCars, ...activePeople].reduce((sum, d) => sum + d.confidence, 0) /
          (activeCars.length + activePeople.length)) *
        100
      : null;

  return (
    <div className="space-y-5">
      <div className="flex flex-col justify-between gap-4 rounded-2xl border border-slate-700/60 bg-gradient-to-br from-[#0d2535] to-[#0a1822] p-6 sm:flex-row sm:items-center">
        <div>
          <div className="text-xs font-bold uppercase tracking-[0.15em] text-cyan-300">
            Task 1 · VisionTraffic AI
          </div>
          <h1 className="mt-1 text-2xl font-bold text-white">Car Colour &amp; People Detection</h1>
          <p className="mt-1 text-sm text-slate-300">
            Detects cars and pedestrians, draws <strong className="text-rose-300">RED</strong> boxes on blue
            cars and <strong className="text-sky-300">BLUE</strong> boxes on other known colours.
          </p>
        </div>
        <button
          type="button"
          disabled={loading}
          onClick={handleLoadAndRunSample}
          className="inline-flex shrink-0 items-center gap-2 rounded-xl bg-cyan-400 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:opacity-50"
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
                setError(null);
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
                setError(null);
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
            {mode === 'image' ? 'Upload traffic image (JPG/PNG/WebP)' : 'Upload traffic video (MP4/WebM)'}
            <input
              type="file"
              accept={mode === 'image' ? 'image/jpeg,image/png,image/webp' : 'video/mp4,video/webm,video/quicktime'}
              className="hidden"
              onChange={(e) => handleFileChange(e.target.files?.[0])}
            />
          </label>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:col-span-5">
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
              className="mt-2 w-full accent-cyan-400"
            />
          </div>
          <div>
            <label className="flex justify-between text-xs text-slate-300">
              <span>Min colour evidence</span>
              <span className="font-mono">{colourThreshold.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={colourThreshold}
              onChange={(e) => setColourThreshold(Number(e.target.value))}
              className="mt-2 w-full accent-cyan-400"
            />
          </div>
        </div>

        <div className="flex items-end lg:col-span-3">
          <button
            type="button"
            disabled={loading}
            onClick={handleAnalyzeClick}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-cyan-400/50 bg-cyan-500/20 px-4 py-3 text-sm font-semibold text-cyan-100 hover:bg-cyan-500/30 disabled:opacity-50"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {loading ? 'Analyzing...' : 'Analyze Media'}
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/40 p-4 text-sm text-rose-200">
          {error}
        </div>
      )}

      {/* KPI Strip */}
      {(result || videoFrames.length > 0) && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <div className="rounded-xl border border-slate-700/70 bg-[#0d1c28] p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              Total Cars {videoFrames.length > 0 ? '(Frame Sum)' : ''}
            </div>
            <div className="mt-1 text-2xl font-bold text-white">{activeCars.length}</div>
          </div>
          <div className="rounded-xl border border-rose-500/40 bg-rose-950/25 p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-rose-300">
              Blue Cars (RED Box)
            </div>
            <div className="mt-1 text-2xl font-bold text-rose-200">{blueCount}</div>
          </div>
          <div className="rounded-xl border border-sky-500/40 bg-sky-950/25 p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-sky-300">
              Other Known Colour (BLUE Box)
            </div>
            <div className="mt-1 text-2xl font-bold text-sky-200">{otherKnownCount}</div>
          </div>
          <div className="rounded-xl border border-emerald-500/40 bg-emerald-950/25 p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-emerald-300">
              People Detected
            </div>
            <div className="mt-1 text-2xl font-bold text-emerald-200">{activePeople.length}</div>
          </div>
          <div className="rounded-xl border border-slate-700/70 bg-[#0d1c28] p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              Avg Confidence
            </div>
            <div className="mt-1 text-2xl font-bold text-cyan-200">
              {avgConf !== null ? `${avgConf.toFixed(1)}%` : '—'}
            </div>
          </div>
        </div>
      )}

      {/* Image Comparison */}
      {result && sourcePreview && annotatedPreview && (
        <div className="space-y-4 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-4 text-xs text-slate-300">
              <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-xs bg-[#ff3b30]" />Blue Car (RED)</span>
              <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-xs bg-[#1d9bf0]" />Other Known Colour (BLUE)</span>
              <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-xs bg-[#94a3b8]" />Unknown Colour</span>
              <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-xs bg-[#10b981]" />Person</span>
            </div>
            <a
              href={annotatedPreview}
              download="visiontraffic_annotated.png"
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-600 bg-slate-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-700"
            >
              <Download className="h-3.5 w-3.5" /> Download Annotated PNG
            </a>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <div className="mb-1.5 text-xs font-medium text-slate-400">Original Image</div>
              <img src={sourcePreview} alt="Original traffic scene" className="w-full rounded-xl border border-slate-800" />
            </div>
            <div>
              <div className="mb-1.5 text-xs font-medium text-cyan-300">Annotated Detections</div>
              <img src={annotatedPreview} alt="Annotated traffic scene" className="w-full rounded-xl border border-cyan-500/30" />
            </div>
          </div>
        </div>
      )}

      {/* Video Frame Results */}
      {videoFrames.length > 0 && (
        <div className="space-y-3 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
          <div className="text-sm font-semibold text-white">
            Video Key-Frame Analysis (per-frame counts; tracking not applied)
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            {videoFrames.map((f) => (
              <div key={f.frameIndex} className="rounded-xl border border-slate-700 bg-[#07121a] p-3">
                <div className="mb-2 flex justify-between text-xs text-slate-300">
                  <span>Frame #{f.frameIndex} ({f.timestampSeconds}s)</span>
                  <span>
                    {f.result.carCount} cars ({f.result.blueCarCount} blue) · {f.result.personCount} people
                  </span>
                </div>
                <img src={f.annotatedUrl} alt={`Annotated frame ${f.frameIndex}`} className="w-full rounded-lg" />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detection Table */}
      {(result || videoFrames.length > 0) && (
        <div className="rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
          <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white">
            <Car className="h-4 w-4 text-cyan-300" /> Detection Details
          </h2>
          {activeCars.length === 0 && activePeople.length === 0 ? (
            <p className="text-xs text-slate-400">
              Inference completed cleanly; no cars or people met the confidence threshold.
            </p>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-900/90 text-slate-400">
                  <tr>
                    <th className="px-3 py-2">ID</th>
                    <th className="px-3 py-2">Class</th>
                    <th className="px-3 py-2">Apparent Colour</th>
                    <th className="px-3 py-2">Confidence</th>
                    <th className="px-3 py-2">Box Style</th>
                    <th className="px-3 py-2">Normalized Box [ymin, xmin, ymax, xmax]</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 text-slate-200">
                  {activeCars.map((car, i) => (
                    <tr key={`${car.id}-${i}`}>
                      <td className="px-3 py-2 font-mono">{car.id}</td>
                      <td className="px-3 py-2 capitalize">{car.className}</td>
                      <td className="px-3 py-2 font-semibold capitalize">{car.colour}</td>
                      <td className="px-3 py-2 font-mono">{(car.confidence * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2">
                        <span
                          className={`rounded px-2 py-0.5 text-[11px] font-semibold ${
                            car.boxStyle === 'red'
                              ? 'bg-rose-500/20 text-rose-300'
                              : car.boxStyle === 'blue'
                                ? 'bg-sky-500/20 text-sky-300'
                                : 'bg-slate-700 text-slate-300'
                          }`}
                        >
                          {car.boxStyle === 'red'
                            ? 'RED (BLUE CAR)'
                            : car.boxStyle === 'blue'
                              ? 'BLUE (OTHER)'
                              : 'GREY (UNKNOWN)'}
                        </span>
                      </td>
                      <td className="px-3 py-2 font-mono text-slate-400">{car.box.join(', ')}</td>
                    </tr>
                  ))}
                  {activePeople.map((p, i) => (
                    <tr key={`${p.id}-${i}`}>
                      <td className="px-3 py-2 font-mono">{p.id}</td>
                      <td className="px-3 py-2 capitalize">{p.className}</td>
                      <td className="px-3 py-2 text-slate-500">—</td>
                      <td className="px-3 py-2 font-mono">{(p.confidence * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2">
                        <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-[11px] font-semibold text-emerald-300">
                          GREEN (PERSON)
                        </span>
                      </td>
                      <td className="px-3 py-2 font-mono text-slate-400">{p.box.join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
