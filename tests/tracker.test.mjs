// Run with:  npm test
import assert from "node:assert/strict";
import { test } from "node:test";
import { dedupe, footPoint, iou, peopleInZone, pointInPolygon, SafetyRules, Tracker } from "../web/js/tracker.js";

const box = (x, y, w = 40, h = 100) => ({ x, y, w, h });
const det = (x, y, score = 0.9) => ({ box: box(x, y), score });
const square = [
  { x: 100, y: 100 },
  { x: 300, y: 100 },
  { x: 300, y: 300 },
  { x: 100, y: 300 },
];

test("iou: identical, disjoint and half-overlapping boxes", () => {
  assert.equal(iou(box(0, 0), box(0, 0)), 1);
  assert.equal(iou(box(0, 0), box(500, 500)), 0);
  // 20 of 40 px overlap horizontally: intersection 20*100, union 60*100
  assert.ok(Math.abs(iou(box(0, 0), box(20, 0)) - 2000 / 6000) < 1e-9);
});

test("footPoint is the bottom centre of the box", () => {
  assert.deepEqual(footPoint(box(10, 20, 40, 100)), { x: 30, y: 120 });
});

test("pointInPolygon works for a square and a concave shape", () => {
  assert.equal(pointInPolygon({ x: 200, y: 200 }, square), true);
  assert.equal(pointInPolygon({ x: 50, y: 200 }, square), false);
  const L = [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 40 }, { x: 40, y: 40 }, { x: 40, y: 100 }, { x: 0, y: 100 }];
  assert.equal(pointInPolygon({ x: 20, y: 80 }, L), true);
  assert.equal(pointInPolygon({ x: 80, y: 80 }, L), false);
});

test("tracker keeps the same id for a person moving across frames", () => {
  const tr = new Tracker({ minHits: 2 });
  let visible;
  for (let f = 0; f < 10; f++) visible = tr.update([det(10 + f * 5, 50)], f * 40);
  assert.equal(visible.length, 1);
  assert.equal(visible[0].id, 1);
  assert.equal(tr.confirmedCount, 1);
});

test("tracker keeps the id of a fast-moving person (boxes barely overlap between frames)", () => {
  const tr = new Tracker({ minHits: 2 });
  let visible;
  // 32 px per frame with a 40 px wide box: consecutive boxes overlap only ~11%
  for (let f = 0; f < 12; f++) visible = tr.update([det(10 + f * 32, 50)], f * 100);
  assert.equal(visible.length, 1);
  assert.equal(visible[0].id, 1);
  assert.equal(tr.confirmedCount, 1);
});

test("two-stage matching: weak detections continue a track but never start one", () => {
  const tr = new Tracker({ minHits: 2, highThreshold: 0.5 });
  for (let f = 0; f < 3; f++) tr.update([det(10 + f * 4, 50, 0.9)], f * 40);
  // the person is half hidden: the detector is only 30% sure for a few frames
  let visible;
  for (let f = 3; f < 8; f++) visible = tr.update([det(10 + f * 4, 50, 0.3)], f * 40);
  assert.equal(visible.length, 1);
  assert.equal(visible[0].id, 1);
  // a weak detection somewhere new does not create a track
  tr.update([det(10 + 8 * 4, 50, 0.9), det(500, 50, 0.3)], 8 * 40);
  assert.equal(tr.tracks.length, 1);
});

test("tracker gives two people two different ids", () => {
  const tr = new Tracker({ minHits: 2 });
  let visible;
  for (let f = 0; f < 5; f++) visible = tr.update([det(10, 50), det(300, 50)], f * 40);
  assert.deepEqual(visible.map((t) => t.id).sort(), [1, 2]);
});

test("a one-frame false detection is never confirmed", () => {
  const tr = new Tracker({ minHits: 3 });
  tr.update([det(10, 50)], 0);
  for (let f = 1; f < 5; f++) tr.update([], f * 40);
  assert.equal(tr.confirmedCount, 0);
});

test("a track survives a short gap but is dropped after maxMissedMs", () => {
  const tr = new Tracker({ minHits: 2, maxMissedMs: 500 });
  tr.update([det(10, 50)], 0);
  tr.update([det(12, 50)], 40);
  tr.update([], 300); // short gap: still kept
  assert.equal(tr.update([det(14, 50)], 400)[0].id, 1);
  tr.update([], 1000); // long gap: dropped
  assert.equal(tr.update([det(16, 50)], 1040).length, 0); // a new, unconfirmed track
  assert.equal(tr.tracks[0].id, 2);
});

test("rules: intrusion fires once when a person enters, loitering after the time limit", () => {
  const tr = new Tracker({ minHits: 1 });
  const rules = new SafetyRules({ loiterMs: 1000, crowdLimit: 99 });
  const events = [];
  // walk from outside (foot y = 90) into the zone (foot y = 200) and stand there
  const ys = [-10, 0, 60, 100, 100, 100, 100, 100, 100];
  ys.forEach((y, f) => {
    const now = f * 250;
    const visible = tr.update([det(150, y)], now);
    events.push(...rules.check(visible, tr.tracks, square, now));
  });
  assert.deepEqual(events.map((e) => e.type), ["intrusion", "loitering"]);
  assert.equal(events[0].trackId, 1);
});

test("rules: someone jittering on the zone edge triggers only one intrusion", () => {
  const tr = new Tracker({ minHits: 1 });
  const rules = new SafetyRules({ loiterMs: 99999, crowdLimit: 99, exitGraceMs: 800 });
  const events = [];
  // foot point alternates just inside (y = 105) and just outside (y = 95) every 100 ms
  for (let f = 0; f < 20; f++) {
    const now = f * 100;
    const visibleTracks = tr.update([det(150, f % 2 ? 5 : -5)], now);
    events.push(...rules.check(visibleTracks, tr.tracks, square, now));
  }
  assert.equal(events.filter((e) => e.type === "intrusion").length, 1);
});

test("rules: leaving for longer than the grace time and coming back is a new intrusion", () => {
  const tr = new Tracker({ minHits: 1 });
  const rules = new SafetyRules({ loiterMs: 99999, crowdLimit: 99, exitGraceMs: 500 });
  // walk out of the zone (foot point below y = 100 is outside), wait ~1.4 s, walk back in
  const ys = [100, 100, 60, 20, -20, -50, -50, -50, -50, -50, -20, 20, 60, 100, 100];
  const events = [];
  ys.forEach((y, f) => {
    const now = f * 200;
    const visibleTracks = tr.update([det(150, y)], now);
    events.push(...rules.check(visibleTracks, tr.tracks, square, now));
  });
  assert.equal(events.filter((e) => e.type === "intrusion").length, 2);
});

test("dedupe keeps one box when the detector returns a box inside another box", () => {
  const big = { box: box(100, 100, 60, 150), score: 0.8 };
  const inner = { box: box(110, 110, 40, 120), score: 0.6 };
  const other = { box: box(400, 100, 60, 150), score: 0.7 };
  const kept = dedupe([inner, big, other]);
  assert.equal(kept.length, 2);
  assert.ok(kept.includes(big) && kept.includes(other));
});

test("rules: crowd alert fires on the rising edge and respects the cooldown", () => {
  const rules = new SafetyRules({ crowdLimit: 3, crowdCooldownMs: 1000 });
  const people = (n) => Array.from({ length: n }, (_, i) => ({ id: i, confirmed: true, box: box(i * 60, 0), lastSeen: 0 }));
  const fire = (n, now) => rules.check(people(n), [], null, now).map((e) => e.type);
  assert.deepEqual(fire(3, 0), ["crowd"]);
  assert.deepEqual(fire(4, 100), []); // still crowded: no repeat
  assert.deepEqual(fire(1, 200), []); // crowd gone
  assert.deepEqual(fire(3, 500), []); // back, but inside the cooldown
  assert.deepEqual(fire(1, 600), []);
  assert.deepEqual(fire(3, 1500), ["crowd"]);
});

test("peopleInZone counts by foot point", () => {
  const tracks = [{ box: box(150, 100) }, { box: box(400, 100) }];
  assert.equal(peopleInZone(tracks, square), 1);
  assert.equal(peopleInZone(tracks, null), 0);
});
