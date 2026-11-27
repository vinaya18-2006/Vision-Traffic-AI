import { useEffect, useRef, useState } from 'react';
import { Loader2, Mic, Play, Sparkles, Square, Upload, Volume2 } from 'lucide-react';
import { generateSampleEmotionWav, runVoiceEmotionLocal } from '../../ml/localEngine';

interface VoiceEmotionResponse {
  label: 'neutral' | 'happy' | 'angry' | 'sad';
  confidence: number;
  scores: {
    neutral: number;
    happy: number;
    angry: number;
    sad: number;
  };
  acousticCues: string;
  transcriptExcerpt: string;
  warning: string | null;
  metadata: {
    model: string;
    labelSchema: string;
    minimumScore: number;
    inferenceMs: number;
  };
}

export function VoiceEmotionTask() {
  const [minimumScore, setMinimumScore] = useState(0.4);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [pendingAudio, setPendingAudio] = useState<{
    blob: Blob;
    filename: string;
  } | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [sampleLoading, setSampleLoading] = useState<string | null>(null);
  const [result, setResult] = useState<VoiceEmotionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const runEmotionAnalysis = async (blob: Blob) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await runVoiceEmotionLocal(blob, minimumScore);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Voice emotion analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (file: File | undefined) => {
    if (!file) return;
    setError(null);
    setResult(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setPendingAudio({
      blob: file,
      filename: file.name,
    });
  };

  const handleGenerateAndRunSample = async (emotion: 'happy' | 'angry' | 'sad' | 'neutral') => {
    setSampleLoading(emotion);
    setError(null);
    setResult(null);
    try {
      const sample = generateSampleEmotionWav(emotion);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      const url = URL.createObjectURL(sample.blob);
      setAudioUrl(url);
      setPendingAudio({
        blob: sample.blob,
        filename: sample.filename,
      });
      setSampleLoading(null);
      await runEmotionAnalysis(sample.blob);
    } catch (err) {
      setSampleLoading(null);
      setError(err instanceof Error ? err.message : 'Could not generate sample voice clip.');
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
      return;
    }
    setError(null);
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        setAudioUrl(URL.createObjectURL(blob));
        setPendingAudio({
          blob,
          filename: 'recorded_voice.webm',
        });
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch {
      setError('Microphone access was denied or unavailable. Upload a WAV/MP3 file or use a sample speech button.');
    }
  };

  const emotionColors: Record<string, string> = {
    happy: 'bg-emerald-400 text-emerald-300',
    angry: 'bg-rose-400 text-rose-300',
    sad: 'bg-sky-400 text-sky-300',
    neutral: 'bg-slate-400 text-slate-300',
  };

  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-slate-700/60 bg-gradient-to-br from-[#0d2535] to-[#0a1822] p-6">
        <div className="text-xs font-bold uppercase tracking-[0.15em] text-cyan-300">
          Task 3 · Speech Emotion Recognition
        </div>
        <h1 className="mt-1 flex items-center gap-2 text-2xl font-bold text-white">
          <Mic className="h-6 w-6 text-cyan-300" /> Voice Emotion Detection
        </h1>
        <p className="mt-1 text-sm text-slate-300">
          Classifies speech across the 4 SUPERB/IEMOCAP emotion classes (<strong>neutral</strong>,{' '}
          <strong>happy</strong>, <strong>angry</strong>, <strong>sad</strong>). Never infers gender or identity.
        </p>
      </div>

      {/* Instant sample generator */}
      <div className="rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-sm font-semibold text-white">
              <Sparkles className="h-4 w-4 text-cyan-300" /> Quick Test with Synthesized Speech WAV
            </h2>
            <p className="mt-0.5 text-xs text-slate-400">
              Generates a real spoken WAV clip and immediately runs emotion classification.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {(['happy', 'angry', 'sad', 'neutral'] as const).map((emo) => (
              <button
                key={emo}
                type="button"
                disabled={loading || Boolean(sampleLoading)}
                onClick={() => handleGenerateAndRunSample(emo)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-400/40 bg-cyan-500/15 px-3 py-2 text-xs font-semibold capitalize text-cyan-100 hover:bg-cyan-500/25 disabled:opacity-50"
              >
                {sampleLoading === emo ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Volume2 className="h-3.5 w-3.5" />
                )}
                Test {emo}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Upload or Record */}
      <div className="grid gap-4 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5 lg:grid-cols-12">
        <div className="flex flex-col gap-3 sm:flex-row lg:col-span-6">
          <label className="flex flex-1 cursor-pointer items-center justify-center gap-2 rounded-xl border border-dashed border-slate-600 bg-[#07121a] px-4 py-4 text-xs font-medium text-slate-200 hover:border-cyan-400">
            <Upload className="h-4 w-4 text-cyan-300" />
            Upload Audio (WAV / MP3 / WebM)
            <input
              type="file"
              accept="audio/wav,audio/mpeg,audio/mp3,audio/webm,audio/ogg,audio/mp4,.wav,.mp3,.webm,.m4a"
              className="hidden"
              onChange={(e) => handleFileUpload(e.target.files?.[0])}
            />
          </label>
          <button
            type="button"
            onClick={toggleRecording}
            className={`flex flex-1 items-center justify-center gap-2 rounded-xl border px-4 py-4 text-xs font-semibold transition ${
              isRecording
                ? 'border-rose-400 bg-rose-500/25 text-rose-100'
                : 'border-slate-600 bg-[#07121a] text-slate-200 hover:border-cyan-400'
            }`}
          >
            {isRecording ? <Square className="h-4 w-4 fill-current" /> : <Mic className="h-4 w-4 text-cyan-300" />}
            {isRecording ? 'Stop Recording' : 'Record Microphone'}
          </button>
        </div>

        <div className="flex flex-col justify-center lg:col-span-3">
          <label className="flex justify-between text-xs text-slate-300">
            <span>Minimum score warning</span>
            <span className="font-mono">{(minimumScore * 100).toFixed(0)}%</span>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={minimumScore}
            onChange={(e) => setMinimumScore(Number(e.target.value))}
            className="mt-2 w-full accent-cyan-400"
          />
        </div>

        <div className="flex items-end lg:col-span-3">
          <button
            type="button"
            disabled={loading || !pendingAudio}
            onClick={() => pendingAudio && runEmotionAnalysis(pendingAudio.blob)}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-400 px-4 py-3 text-sm font-semibold text-slate-950 hover:bg-cyan-300 disabled:opacity-50"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {loading ? 'Analyzing...' : 'Analyze Audio'}
          </button>
        </div>
      </div>

      {audioUrl && pendingAudio && (
        <div className="rounded-xl border border-slate-700/70 bg-[#0d1c28] p-4">
          <div className="mb-2 text-xs text-slate-400">
            Loaded Clip: <span className="font-mono text-slate-200">{pendingAudio.filename}</span>
          </div>
          <audio controls src={audioUrl} className="w-full" />
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/40 p-4 text-sm text-rose-200">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4 rounded-2xl border border-slate-700/70 bg-[#0d1c28] p-5">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-cyan-500/35 bg-[#07121a] p-4">
              <div className="text-[11px] uppercase tracking-wider text-slate-400">Predicted Emotion Class</div>
              <div className="mt-1 text-3xl font-bold capitalize text-white">{result.label}</div>
              <div className="mt-1 text-xs text-slate-400">Model estimate, not internal state</div>
            </div>
            <div className="rounded-xl border border-slate-700 bg-[#07121a] p-4">
              <div className="text-[11px] uppercase tracking-wider text-slate-400">Model Score</div>
              <div className="mt-1 text-3xl font-bold text-cyan-200">
                {(result.confidence * 100).toFixed(1)}%
              </div>
              <div className="mt-1 text-xs text-slate-400">Uncalibrated model confidence</div>
            </div>
            <div className="rounded-xl border border-slate-700 bg-[#07121a] p-4">
              <div className="text-[11px] uppercase tracking-wider text-slate-400">Acoustic Cues</div>
              <div className="mt-1 text-xs leading-relaxed text-slate-200">{result.acousticCues}</div>
            </div>
          </div>

          {result.warning && (
            <div className="rounded-xl border border-amber-500/40 bg-amber-950/30 p-3 text-xs text-amber-200">
              {result.warning}
            </div>
          )}

          <div className="space-y-2.5">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Class Score Distribution
            </div>
            {(['happy', 'angry', 'sad', 'neutral'] as const).map((cls) => {
              const score = result.scores[cls] ?? 0;
              const barColor = emotionColors[cls].split(' ')[0];
              return (
                <div key={cls} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="capitalize text-slate-200">{cls}</span>
                    <span className="font-mono text-slate-300">{(score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                    <div className={`h-full ${barColor}`} style={{ width: `${Math.min(100, score * 100)}%` }} />
                  </div>
                </div>
              );
            })}
          </div>

          {result.transcriptExcerpt && (
            <div className="rounded-xl border border-slate-800 bg-[#07121a] p-3 text-xs text-slate-300">
              <strong className="text-slate-200">Speech Excerpt:</strong> &ldquo;{result.transcriptExcerpt}&rdquo;
            </div>
          )}
        </div>
      )}
    </div>
  );
}
