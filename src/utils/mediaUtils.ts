export interface AnnotationBox {
  box: [number, number, number, number]; // [ymin, xmin, ymax, xmax] in 0..1000
  label: string;
  color: string;
}

export interface VideoFrameSample {
  frameIndex: number;
  timestampSeconds: number;
  base64: string;
  mimeType: 'image/jpeg';
  previewUrl: string;
}

export async function blobToBase64(blob: Blob): Promise<{ base64: string; mimeType: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = typeof reader.result === 'string' ? reader.result : '';
      const commaIdx = result.indexOf(',');
      if (commaIdx === -1) {
        reject(new Error('Failed to encode file as base64.'));
        return;
      }
      const header = result.slice(0, commaIdx);
      const mimeMatch = header.match(/data:([^;]+);base64/i);
      resolve({
        base64: result.slice(commaIdx + 1),
        mimeType: mimeMatch?.[1]?.toLowerCase() || blob.type || 'application/octet-stream',
      });
    };
    reader.onerror = () => reject(new Error('Could not read the selected file.'));
    reader.readAsDataURL(blob);
  });
}

export async function urlToImagePayload(
  url: string,
): Promise<{ base64: string; mimeType: string; previewUrl: string }> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error('Failed to load sample media.');
  }
  const blob = await res.blob();
  const { base64, mimeType } = await blobToBase64(blob);
  return {
    base64,
    mimeType: mimeType === 'application/octet-stream' ? 'image/jpeg' : mimeType,
    previewUrl: url,
  };
}

export function base64ToBlobUrl(base64: string, mimeType: string): string {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: mimeType });
  return URL.createObjectURL(blob);
}

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export async function annotateImageWithBoxes(
  imageUrl: string,
  annotations: AnnotationBox[],
  handLandmarks?: Array<{ x: number; y: number }> | null,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Canvas context unavailable.'));
        return;
      }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Optional 21-point hand landmark skeleton overlay
      if (handLandmarks && handLandmarks.length === 21) {
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) * 0.004));
        for (const [a, b] of HAND_CONNECTIONS) {
          ctx.beginPath();
          ctx.moveTo(handLandmarks[a].x * canvas.width, handLandmarks[a].y * canvas.height);
          ctx.lineTo(handLandmarks[b].x * canvas.width, handLandmarks[b].y * canvas.height);
          ctx.stroke();
        }
        const dotR = Math.max(4, Math.round(Math.min(canvas.width, canvas.height) * 0.007));
        for (const pt of handLandmarks) {
          ctx.fillStyle = '#34d399';
          ctx.beginPath();
          ctx.arc(pt.x * canvas.width, pt.y * canvas.height, dotR, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      const lineWidth = Math.max(3, Math.round(Math.min(canvas.width, canvas.height) * 0.005));
      const fontSize = Math.max(14, Math.round(Math.min(canvas.width, canvas.height) * 0.024));
      ctx.font = `700 ${fontSize}px Inter, system-ui, sans-serif`;
      ctx.textBaseline = 'top';

      for (const item of annotations) {
        const [ymin, xmin, ymax, xmax] = item.box;
        const x = (xmin / 1000) * canvas.width;
        const y = (ymin / 1000) * canvas.height;
        const w = ((xmax - xmin) / 1000) * canvas.width;
        const h = ((ymax - ymin) / 1000) * canvas.height;

        if (w < 2 || h < 2) continue;

        ctx.strokeStyle = item.color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x, y, w, h);

        const padX = Math.round(fontSize * 0.45);
        const padY = Math.round(fontSize * 0.25);
        const textWidth = ctx.measureText(item.label).width;
        const boxH = fontSize + padY * 2;
        const labelX = Math.max(0, Math.min(x, canvas.width - textWidth - padX * 2));
        const labelY = y - boxH >= 0 ? y - boxH : y;

        ctx.fillStyle = item.color;
        ctx.fillRect(labelX, labelY, textWidth + padX * 2, boxH);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(item.label, labelX + padX, labelY + padY);
      }

      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = () => reject(new Error('Failed to render annotated image.'));
    img.src = imageUrl;
  });
}

export async function extractVideoKeyFrames(
  videoFile: File,
  maxFrames = 4,
): Promise<VideoFrameSample[]> {
  const objectUrl = URL.createObjectURL(videoFile);
  try {
    const video = document.createElement('video');
    video.src = objectUrl;
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';

    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('Could not decode the uploaded video file.'));
    });

    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1;
    const count = Math.max(1, Math.min(maxFrames, 6));
    const timestamps = Array.from({ length: count }, (_, idx) =>
      count === 1 ? Math.min(0.2, duration * 0.5) : (duration * (idx + 0.5)) / count,
    );

    const canvas = document.createElement('canvas');
    const maxDim = 960;
    const scale = Math.min(1, maxDim / Math.max(video.videoWidth || 640, video.videoHeight || 360));
    canvas.width = Math.max(160, Math.round((video.videoWidth || 640) * scale));
    canvas.height = Math.max(120, Math.round((video.videoHeight || 360) * scale));
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context unavailable.');

    const frames: VideoFrameSample[] = [];
    for (let i = 0; i < timestamps.length; i += 1) {
      const ts = Math.min(duration, Math.max(0, timestamps[i]));
      await new Promise<void>((resolve) => {
        video.onseeked = () => resolve();
        video.currentTime = ts;
      });
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      frames.push({
        frameIndex: i + 1,
        timestampSeconds: Number(ts.toFixed(2)),
        base64: dataUrl.split(',')[1] ?? '',
        mimeType: 'image/jpeg',
        previewUrl: dataUrl,
      });
    }
    return frames;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export async function postJson<T>(url: string, payload: Record<string, unknown>): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data?.error || `Request failed with status ${response.status}.`);
  }
  return data as T;
}
