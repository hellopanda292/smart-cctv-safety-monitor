// Tracking and safety rules for the Smart CCTV Safety Monitor.
//
// This file has no browser dependencies, so it can be unit-tested with Node
// (see tests/tracker.test.mjs). Boxes are { x, y, w, h } in pixels.

/* ---------- Geometry ---------- */

/** Intersection over Union: how much two boxes overlap (0 = not at all, 1 = identical). */
export function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

/** Fraction of box a that lies inside box b (1 = a is completely inside b). */
export function containment(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return a.w * a.h > 0 ? inter / (a.w * a.h) : 0;
}

/**
 * Removes duplicate detections of the same person. The detector sometimes returns a big box
 * and a smaller box inside it for one person; keep the most confident one.
 */
export function dedupe(detections, { iouLimit = 0.5, containLimit = 0.85 } = {}) {
  const kept = [];
  for (const d of [...detections].sort((a, b) => b.score - a.score)) {
    const duplicate = kept.some(
      (k) => iou(d.box, k.box) > iouLimit || containment(d.box, k.box) > containLimit || containment(k.box, d.box) > containLimit,
    );
    if (!duplicate) kept.push(d);
  }
  return kept;
}

/** Bottom-centre of a box: where the person is standing. Used for zone checks. */
export function footPoint(box) {
  return { x: box.x + box.w / 2, y: box.y + box.h };
}

/** Ray-casting test: is point p inside the polygon (list of {x, y})? */
export function pointInPolygon(p, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i];
    const b = polygon[j];
    const crosses = a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x;
    if (crosses) inside = !inside;
  }
  return inside;
}

/* ---------- Multi-object tracker ---------- */

/**
 * A simple IoU tracker (similar in spirit to SORT, with a constant-velocity prediction
 * instead of a full Kalman filter): every frame, each existing track is moved to where its
 * recent speed says it should be, then matched to the detection it overlaps most.
 * Unmatched detections start new tracks; tracks that go unseen for too long are dropped.
 * A track is only "confirmed" after it has been seen in a few frames, which filters out
 * one-frame false detections.
 */
export class Tracker {
  /**
   * highThreshold enables ByteTrack-style two-stage matching: detections scoring at least
   * highThreshold are matched first and may start new tracks; weaker detections are then used
   * only to *continue* tracks that are already being followed (for example a person who is
   * partly hidden for a moment), never to create new ones. 0 treats every detection as strong.
   */
  constructor({ iouThreshold = 0.25, maxMissedMs = 1500, minHits = 3, smoothing = 0.55, highThreshold = 0 } = {}) {
    this.iouThreshold = iouThreshold;
    this.maxMissedMs = maxMissedMs;
    this.minHits = minHits;
    this.smoothing = smoothing;
    this.highThreshold = highThreshold;
    this.reset();
  }

  reset() {
    this.tracks = [];
    this.nextId = 1;
    this.confirmedCount = 0;
  }

  /** Where we expect the track to be at time `now`, assuming it keeps its recent velocity. */
  predict(track, now) {
    const dt = Math.min(now - track.lastSeen, 500);
    return { ...track.box, x: track.box.x + track.vx * dt, y: track.box.y + track.vy * dt };
  }

  /**
   * Greedily matches the given tracks to detections and updates the matched tracks.
   * Overlap with the *predicted* position is the main signal; when allowDistance is true and
   * boxes don't overlap enough (fast movement, slow frame rate), a close centre distance still
   * counts, but always ranks below a real overlap. Returns the indexes of unused detections.
   */
  associate(tracks, detections, now, allowDistance) {
    const pairs = [];
    tracks.forEach((t, ti) => {
      const p = this.predict(t, now);
      detections.forEach((d, di) => {
        const overlap = iou(p, d.box);
        let score = overlap >= this.iouThreshold ? 1 + overlap : 0;
        if (!score && allowDistance) {
          const dist = Math.hypot(p.x + p.w / 2 - (d.box.x + d.box.w / 2), p.y + p.h / 2 - (d.box.y + d.box.h / 2));
          const sizeRatio = Math.min(p.h, d.box.h) / Math.max(p.h, d.box.h);
          const norm = dist / Math.max(p.h, d.box.h);
          if (norm < 0.6 && sizeRatio > 0.6) score = 0.6 - norm;
        }
        if (score > 0) pairs.push({ ti, di, score });
      });
    });
    pairs.sort((a, b) => b.score - a.score);

    const usedTracks = new Set();
    const usedDetections = new Set();
    for (const { ti, di } of pairs) {
      if (usedTracks.has(ti) || usedDetections.has(di)) continue;
      usedTracks.add(ti);
      usedDetections.add(di);
      this.applyMatch(tracks[ti], detections[di], now);
    }
    return detections.map((_, di) => di).filter((di) => !usedDetections.has(di));
  }

  applyMatch(track, d, now) {
    const s = this.smoothing;
    // Velocity (px per ms), smoothed so one noisy frame doesn't throw the prediction off.
    const dt = Math.max(now - track.lastSeen, 1);
    track.vx = 0.5 * track.vx + 0.5 * ((d.box.x - track.box.x) / dt);
    track.vy = 0.5 * track.vy + 0.5 * ((d.box.y - track.box.y) / dt);
    // Exponential smoothing keeps boxes from jittering between frames.
    track.box = {
      x: s * d.box.x + (1 - s) * track.box.x,
      y: s * d.box.y + (1 - s) * track.box.y,
      w: s * d.box.w + (1 - s) * track.box.w,
      h: s * d.box.h + (1 - s) * track.box.h,
    };
    track.score = d.score;
    track.hits += 1;
    track.lastSeen = now;
    if (!track.confirmed && track.hits >= this.minHits) {
      track.confirmed = true;
      this.confirmedCount += 1;
    }
  }

  /** detections: [{ box, score }], now: time in ms. Returns the currently visible confirmed tracks. */
  update(rawDetections, now) {
    const all = dedupe(rawDetections);
    const strong = all.filter((d) => d.score >= this.highThreshold);
    const weak = all.filter((d) => d.score < this.highThreshold);

    // Stage 1: strong detections against all tracks.
    const unusedStrong = this.associate(this.tracks, strong, now, true);
    // Stage 2: weak detections only keep already-followed (confirmed) tracks alive.
    const waiting = this.tracks.filter((t) => t.lastSeen !== now && t.confirmed);
    if (weak.length && waiting.length) this.associate(waiting, weak, now, false);

    for (const di of unusedStrong) {
      const d = strong[di];
      this.tracks.push({
        id: this.nextId++,
        box: { ...d.box },
        vx: 0,
        vy: 0,
        score: d.score,
        hits: 1,
        firstSeen: now,
        lastSeen: now,
        confirmed: this.minHits <= 1,
        zoneSince: null,
        outsideSince: null,
        loiterAlerted: false,
      });
      if (this.minHits <= 1) this.confirmedCount += 1;
    }

    this.tracks = this.tracks.filter((t) => now - t.lastSeen <= this.maxMissedMs);
    return this.tracks.filter((t) => t.confirmed && t.lastSeen === now);
  }
}

/* ---------- Safety rules ---------- */

/**
 * Turns tracks + zone into events. Each rule fires once per "episode":
 * - intrusion: a person's foot point enters the zone
 * - loitering: a person stays in the zone for loiterMs
 * - crowd: the number of people reaches crowdLimit (then waits for crowdCooldownMs)
 *
 * Hysteresis: someone standing on the zone edge jitters in and out every frame. A person only
 * counts as having *left* after being outside for exitGraceMs, so the edge can't spam alerts.
 */
export class SafetyRules {
  constructor({ loiterMs = 4000, crowdLimit = 8, crowdCooldownMs = 10000, exitGraceMs = 1000 } = {}) {
    this.loiterMs = loiterMs;
    this.crowdLimit = crowdLimit;
    this.crowdCooldownMs = crowdCooldownMs;
    this.exitGraceMs = exitGraceMs;
    this.reset();
  }

  reset() {
    this.lastCrowdAt = -Infinity;
    this.crowdActive = false;
  }

  /** zone: polygon in pixels, or null. Returns new events: [{ type, trackId?, count?, at }]. */
  check(visibleTracks, allTracks, zone, now) {
    const events = [];

    if (zone && zone.length >= 3) {
      for (const t of allTracks) {
        if (!t.confirmed) continue;
        const visible = t.lastSeen === now;
        const inside = visible && pointInPolygon(footPoint(t.box), zone);
        if (inside) {
          t.outsideSince = null;
          if (t.zoneSince === null) {
            t.zoneSince = now;
            t.loiterAlerted = false;
            events.push({ type: "intrusion", trackId: t.id, at: now });
          } else if (!t.loiterAlerted && now - t.zoneSince >= this.loiterMs) {
            t.loiterAlerted = true;
            events.push({ type: "loitering", trackId: t.id, seconds: Math.round((now - t.zoneSince) / 1000), at: now });
          }
        } else if (visible && t.zoneSince !== null) {
          t.outsideSince ??= now;
          if (now - t.outsideSince >= this.exitGraceMs) {
            t.zoneSince = null; // has really left the zone
            t.outsideSince = null;
          }
        }
      }
    }

    const count = visibleTracks.length;
    if (count >= this.crowdLimit) {
      if (!this.crowdActive && now - this.lastCrowdAt >= this.crowdCooldownMs) {
        events.push({ type: "crowd", count, at: now });
        this.lastCrowdAt = now;
      }
      this.crowdActive = true;
    } else {
      this.crowdActive = false;
    }

    return events;
  }
}

/** People currently inside the zone (by foot point). */
export function peopleInZone(tracks, zone) {
  if (!zone || zone.length < 3) return 0;
  return tracks.filter((t) => pointInPolygon(footPoint(t.box), zone)).length;
}
