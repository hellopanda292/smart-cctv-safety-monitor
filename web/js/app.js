// Smart CCTV Safety Monitor: browser app.
// Pipeline per video frame: MediaPipe person detection -> Tracker (IDs) -> SafetyRules (alerts) -> draw.
import { FilesetResolver, ObjectDetector } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { footPoint, pointInPolygon, SafetyRules, Tracker } from "./tracker.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL = "models/efficientdet_lite0.tflite";
const DEMO_VIDEO = "media/pedestrians.webm";
// Default zone for the demo video, as fractions of the frame (so it works at any size).
const DEMO_ZONE = [{ x: 0.8, y: 0.5 }, { x: 0.99, y: 0.5 }, { x: 0.99, y: 0.74 }, { x: 0.8, y: 0.74 }]; // the shop entrance on the right
const LABELS = { intrusion: "Intrusion", loitering: "Loitering", crowd: "Crowd" };

const $ = (id) => document.getElementById(id);
const screen = $("screen");
const canvas = $("view");
const ctx = canvas.getContext("2d");
const video = Object.assign(document.createElement("video"), { muted: true, playsInline: true, crossOrigin: "anonymous" });
const pixelCanvas = document.createElement("canvas");
const pixelCtx = pixelCanvas.getContext("2d");
const thumbCanvas = Object.assign(document.createElement("canvas"), { width: 192, height: 108 });
const thumbCtx = thumbCanvas.getContext("2d");

const tracker = new Tracker({ highThreshold: 0.5, maxMissedMs: 1500 });
const rules = new SafetyRules({ loiterMs: 3000, crowdLimit: 6 });
window.__cctv = { tracker, rules }; // handy for debugging in the browser console

let detector = null;
let stream = null;
let paused = false;
let zoneNorm = DEMO_ZONE.map((p) => ({ ...p }));
let drawMode = false;
let drag = null;
let visible = [];
let now = 0;
let lastVideoTime = -1;
let lastDetectAt = 0;
let inferMs = 0;
let detectStamps = [];
let alertCount = 0;
let events = [];
let audioCtx = null;

/* ---------- Helpers ---------- */

const isCamera = () => Boolean(video.srcObject);
const zonePx = () => (zoneNorm ? zoneNorm.map((p) => ({ x: p.x * canvas.width, y: p.y * canvas.height })) : null);
const fmtVideoTime = (ms) => {
  const s = ms / 1000;
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
};

function showMsg(text) {
  $("msg").textContent = text;
  $("msg").hidden = !text;
}

/* ---------- Model ---------- */

async function loadDetector() {
  const vision = await FilesetResolver.forVisionTasks(WASM);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: "VIDEO",
    scoreThreshold: 0.2, // low on purpose: weak detections help keep existing tracks alive (see Tracker)
    categoryAllowlist: ["person"],
    maxResults: 40,
  });
  try {
    detector = await ObjectDetector.createFromOptions(vision, options("GPU"));
  } catch (err) {
    console.warn("GPU delegate unavailable, falling back to CPU", err);
    detector = await ObjectDetector.createFromOptions(vision, options("CPU"));
  }
}

/* ---------- Sources ---------- */

function stopCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
}

function resetSession() {
  tracker.reset();
  rules.reset();
  visible = [];
  events = [];
  alertCount = 0;
  lastVideoTime = -1;
  $("events").innerHTML = "";
  $("empty").hidden = false;
  $("export").disabled = true;
  updateStats();
}

async function play() {
  resetSession();
  paused = false;
  $("pause").textContent = "Pause";

  try {
    await video.play();
    if (detector) showMsg("");
  } catch (err) {
    showMsg("Press play to start the video.");
  }
}

function setActive(src) {
  document.querySelectorAll(".src").forEach((b) => b.classList.toggle("active", b.dataset.src === src || (src === "file" && b.classList.contains("upload"))));
}

async function useDemo() {
  stopCamera();
  video.src = DEMO_VIDEO;
  video.loop = true;
  zoneNorm = DEMO_ZONE.map((p) => ({ ...p }));
  setActive("demo");
  await play();
}

async function useCamera() {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 540 } }, audio: false });
    stopCamera();
    stream = s;
    video.removeAttribute("src");
    video.srcObject = stream;
    video.loop = false;
    setActive("camera");
    await play();
  } catch (err) {
    console.warn(err);
    showMsg("Camera access was blocked or no camera was found. You can still use the demo video or upload a video.");
  }
}

async function useFile(file) {
  if (!file) return;
  stopCamera();
  video.src = URL.createObjectURL(file);
  video.loop = true;
  setActive("file");
  await play();
}

/* ---------- Main loop ---------- */

function loop() {
  requestAnimationFrame(loop);
  if (video.readyState < 2 || !video.videoWidth) return;

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.style.aspectRatio = `${canvas.width} / ${canvas.height}`;
  }

  const camera = isCamera();
  const t = performance.now();
  // Only run the detector when there is a new video frame (or ~20 times a second for a camera).
  const newFrame = camera ? t - lastDetectAt > 45 : video.currentTime !== lastVideoTime;

  if (detector && newFrame && !paused) {
    if (!camera && video.currentTime < lastVideoTime) {
      tracker.reset(); // the demo video looped: start counting again
      rules.reset();
    }
    lastVideoTime = video.currentTime;
    lastDetectAt = t;
    now = camera ? t : video.currentTime * 1000;

    const result = detector.detectForVideo(video, t);
    inferMs = inferMs ? 0.85 * inferMs + 0.15 * (performance.now() - t) : performance.now() - t;
    detectStamps.push(t);
    detectStamps = detectStamps.filter((s) => t - s < 1000);

    const detections = result.detections.map((d) => ({
      box: { x: d.boundingBox.originX, y: d.boundingBox.originY, w: d.boundingBox.width, h: d.boundingBox.height },
      score: d.categories[0]?.score ?? 0,
    }));
    visible = tracker.update(detections, now);
    const newEvents = rules.check(visible, tracker.tracks, zonePx(), now);
    draw();
    newEvents.forEach(addEvent);
    updateStats();
    return;
  }
  draw();
}

/* ---------- Drawing ---------- */

function pixelate(b) {
  const x = Math.max(0, Math.floor(b.x));
  const y = Math.max(0, Math.floor(b.y));
  const w = Math.min(canvas.width - x, Math.ceil(b.w));
  const h = Math.min(canvas.height - y, Math.ceil(b.h));
  if (w < 4 || h < 4) return;
  const block = Math.max(5, Math.round(w / 9));
  pixelCanvas.width = Math.max(1, Math.round(w / block));
  pixelCanvas.height = Math.max(1, Math.round(h / block));
  pixelCtx.drawImage(canvas, x, y, w, h, 0, 0, pixelCanvas.width, pixelCanvas.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(pixelCanvas, 0, 0, pixelCanvas.width, pixelCanvas.height, x, y, w, h);
  ctx.imageSmoothingEnabled = true;
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

function label(text, x, y, bg, u) {
  // (x is clamped below so labels never run off the right edge of the frame)
  ctx.font = `700 ${Math.round(13 * u)}px "JetBrains Mono", Consolas, monospace`;
  const pad = 5 * u;
  const w = ctx.measureText(text).width + pad * 2;
  x = Math.max(2 * u, Math.min(x, canvas.width - w - 2 * u));
  const h = 19 * u;
  roundRect(x, y - h, w, h, 5 * u);
  ctx.fillStyle = bg;
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x + pad, y - 5.5 * u);
}

function drawZone(zone, alarm, u, preview = false) {
  ctx.beginPath();
  zone.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.closePath();
  ctx.fillStyle = alarm ? "rgba(255, 77, 109, 0.22)" : "rgba(122, 77, 255, 0.16)";
  ctx.fill();
  ctx.setLineDash([10 * u, 7 * u]);
  ctx.lineWidth = 2.5 * u;
  ctx.strokeStyle = alarm ? "#ff4d6d" : preview ? "#ffd36b" : "#b49bff";
  ctx.stroke();
  ctx.setLineDash([]);
  if (!preview) {
    const top = zone.reduce((a, p) => (p.y < a.y || (p.y === a.y && p.x < a.x) ? p : a), zone[0]);
    label(alarm ? "ZONE · ALERT" : "RESTRICTED ZONE", top.x + 4 * u, top.y - 6 * u, alarm ? "#ff4d6d" : "#7a4dff", u);
  }
}

function draw() {
  const w = canvas.width;
  const h = canvas.height;
  const u = w / 854; // scale lines and text with the video size
  ctx.drawImage(video, 0, 0, w, h);
  if ($("privacy").checked) visible.forEach((t) => pixelate(t.box));

  const zone = zonePx();
  const inZone = new Set(zone ? visible.filter((t) => pointInPolygon(footPoint(t.box), zone)).map((t) => t.id) : []);
  if (zone) drawZone(zone, inZone.size > 0, u);
  if (drag?.rect) drawZone(rectToZone(drag.rect).map((p) => ({ x: p.x * w, y: p.y * h })), false, u, true);

  for (const t of visible) {
    const alarm = inZone.has(t.id);
    const color = alarm ? "#ff4d6d" : "#3ee0ff";
    ctx.lineWidth = 2.5 * u;
    ctx.strokeStyle = color;
    roundRect(t.box.x, t.box.y, t.box.w, t.box.h, 6 * u);
    ctx.stroke();
    const f = footPoint(t.box);
    ctx.beginPath();
    ctx.arc(f.x, f.y, 4 * u, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    let text = `#${t.id}`;
    if (alarm && t.zoneSince !== null) text += ` · ${Math.max(0, (now - t.zoneSince) / 1000).toFixed(0)}s`;
    label(text, t.box.x, Math.max(t.box.y, 22 * u), alarm ? "#ff4d6d" : "rgba(13, 18, 48, 0.85)", u);
  }

  // Camera-style overlay: REC dot, source, time, and the people count
  const stamp = isCamera() ? new Date().toLocaleTimeString() : fmtVideoTime(video.currentTime * 1000);
  ctx.fillStyle = "rgba(5, 8, 26, 0.6)";
  roundRect(10 * u, 10 * u, 214 * u, 26 * u, 8 * u);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(24 * u, 23 * u, 5 * u, 0, Math.PI * 2);
  ctx.fillStyle = paused ? "#9aa3c2" : "#ff4d6d";
  ctx.fill();
  ctx.font = `700 ${Math.round(12.5 * u)}px "JetBrains Mono", Consolas, monospace`;
  ctx.fillStyle = "#fff";
  ctx.fillText(`${isCamera() ? "CAM 01" : "DEMO"}  ${stamp}`, 36 * u, 27.5 * u);
  const count = `PEOPLE ${visible.length}`;
  const cw = ctx.measureText(count).width + 20 * u;
  ctx.fillStyle = "rgba(5, 8, 26, 0.6)";
  roundRect(w - cw - 10 * u, 10 * u, cw, 26 * u, 8 * u);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.fillText(count, w - cw, 27.5 * u);
}

/* ---------- Events ---------- */

function beep() {
  try {
    audioCtx ??= new AudioContext();
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.frequency.value = 880;
    g.gain.setValueAtTime(0.15, audioCtx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.25);
    o.connect(g).connect(audioCtx.destination);
    o.start();
    o.stop(audioCtx.currentTime + 0.25);
  } catch {
    /* sound is optional */
  }
}

function addEvent(e) {
  alertCount += 1;
  const time = isCamera() ? new Date().toLocaleTimeString() : fmtVideoTime(e.at);
  const detail =
    e.type === "crowd" ? `${e.count} people in view`
    : e.type === "loitering" ? `Person #${e.trackId} stayed ${e.seconds}s in the zone`
    : `Person #${e.trackId} entered the zone`;
  thumbCtx.drawImage(canvas, 0, 0, thumbCanvas.width, thumbCanvas.height);
  const entry = { time, type: e.type, person: e.trackId ?? "", detail, thumb: thumbCanvas.toDataURL("image/jpeg", 0.7) };
  events.unshift(entry);
  if (events.length > 60) events.pop();

  const li = document.createElement("li");
  li.className = e.type;
  const img = Object.assign(document.createElement("img"), { src: entry.thumb, alt: "" });
  const body = document.createElement("div");
  const title = Object.assign(document.createElement("p"), { className: "t", textContent: `${LABELS[e.type]} · ${time}` });
  const desc = Object.assign(document.createElement("p"), { className: "d", textContent: detail });
  body.append(title, desc);
  li.append(img, body);
  const list = $("events");
  list.prepend(li);
  while (list.children.length > 60) list.lastChild.remove();
  $("empty").hidden = true;
  $("export").disabled = false;

  screen.classList.remove("flash");
  void screen.offsetWidth;
  screen.classList.add("flash");
  if ($("sound").checked) beep();
}

function updateStats() {
  const zone = zonePx();
  $("s-now").textContent = visible.length;
  $("s-zone").textContent = zone ? visible.filter((t) => pointInPolygon(footPoint(t.box), zone)).length : "–";
  $("s-unique").textContent = tracker.confirmedCount;
  $("s-alerts").textContent = alertCount;
  const fps = detectStamps.length;
  $("s-fps").textContent = inferMs ? `${fps} fps · ${inferMs.toFixed(0)} ms per frame` : "–";
}

function exportCsv() {
  const rows = [["time", "type", "person", "detail"], ...events.slice().reverse().map((e) => [e.time, e.type, e.person, e.detail])];
  const csv = rows.map((r) => r.map((v) => `"${String(v).replaceAll('"', '""')}"`).join(",")).join("\n");
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })), download: "cctv-events.csv" });
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ---------- Zone drawing ---------- */

function rectToZone(r) {
  return [{ x: r.x0, y: r.y0 }, { x: r.x1, y: r.y0 }, { x: r.x1, y: r.y1 }, { x: r.x0, y: r.y1 }];
}

function normPoint(e) {
  const r = canvas.getBoundingClientRect();
  return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
}

function setDrawMode(on) {
  drawMode = on;
  $("draw-zone").setAttribute("aria-pressed", String(on));
  screen.classList.toggle("drawing", on);
  $("hint").textContent = on
    ? "Now drag a rectangle on the video. Anyone whose feet enter it triggers an alert."
    : "Drag on the video to draw a restricted zone. People are checked by where their feet are.";
}

canvas.addEventListener("pointerdown", (e) => {
  if (!drawMode) return;
  canvas.setPointerCapture(e.pointerId);
  const p = normPoint(e);
  drag = { start: p, rect: null };
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const p = normPoint(e);
  drag.rect = { x0: Math.min(drag.start.x, p.x), y0: Math.min(drag.start.y, p.y), x1: Math.max(drag.start.x, p.x), y1: Math.max(drag.start.y, p.y) };
});
canvas.addEventListener("pointerup", () => {
  if (!drag) return;
  const r = drag.rect;
  drag = null;
  if (r && r.x1 - r.x0 > 0.03 && r.y1 - r.y0 > 0.03) {
    zoneNorm = rectToZone(r);
    tracker.tracks.forEach((t) => (t.zoneSince = null));
    setDrawMode(false);
  }
});

/* ---------- Controls ---------- */

document.querySelector('[data-src="demo"]').addEventListener("click", useDemo);
document.querySelector('[data-src="camera"]').addEventListener("click", useCamera);
$("file").addEventListener("change", (e) => useFile(e.target.files[0]));
$("pause").addEventListener("click", () => {
  paused = !paused;
  if (paused) video.pause();
  else video.play();
  $("pause").textContent = paused ? "Play" : "Pause";

});
$("draw-zone").addEventListener("click", () => setDrawMode(!drawMode));
$("clear-zone").addEventListener("click", () => {
  zoneNorm = null;
  setDrawMode(false);
  updateStats();
});
$("export").addEventListener("click", exportCsv);
$("loiter").addEventListener("input", (e) => {
  rules.loiterMs = Number(e.target.value) * 1000;
  $("loiter-out").textContent = e.target.value;
});
$("crowd").addEventListener("input", (e) => {
  rules.crowdLimit = Number(e.target.value);
  $("crowd-out").textContent = e.target.value;
});
$("conf").addEventListener("input", (e) => {
  $("conf-out").textContent = e.target.value;
  tracker.highThreshold = Number(e.target.value) / 100;
});

/* ---------- Start ---------- */

(async () => {
  requestAnimationFrame(loop);
  await useDemo();
  try {
    await loadDetector();
    showMsg(video.paused && !paused ? "Press play to start the video." : "");
  } catch (err) {
    console.error(err);
    showMsg("The detection model couldn't load. Please check your connection and refresh.");
  }
})();
