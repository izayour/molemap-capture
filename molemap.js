/* molemap.js — on-device port of the step-2 pipeline (fit camera + pose from 2D keypoints,
 * ray-cast mole pixels onto the posed Anny template → anchors) and the glue the capture page
 * needs.  Plain ES module, no dependencies; runs in the browser and in node (for the tests).
 *
 * Numerics mirror molemap/fit.py, raycast.py and lift.py: same residuals, priors, gates and
 * confidence factors, except the optimiser (Levenberg–Marquardt with numerical Jacobians here,
 * Adam + L-BFGS there) and the silhouette-distance factor (not computed here).
 */

// ------------------------------------------------------------------ small linear algebra
const EPS = 1e-12;
export function rotvecToMat(r) {                     // (3) → row-major 3x3
  const th = Math.hypot(r[0], r[1], r[2]);
  if (th < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const x = r[0] / th, y = r[1] / th, z = r[2] / th, c = Math.cos(th), s = Math.sin(th), C = 1 - c;
  return [c + x * x * C, x * y * C - z * s, x * z * C + y * s,
          y * x * C + z * s, c + y * y * C, y * z * C - x * s,
          z * x * C - y * s, z * y * C + x * s, c + z * z * C];
}
export function matToRotvec(m) {
  const tr = (m[0] + m[4] + m[8] - 1) / 2, th = Math.acos(Math.max(-1, Math.min(1, tr)));
  if (th < 1e-9) return [0, 0, 0];
  const k = th / (2 * Math.sin(th));
  return [(m[7] - m[5]) * k, (m[2] - m[6]) * k, (m[3] - m[1]) * k];
}
const m3mul = (a, b) => [
  a[0] * b[0] + a[1] * b[3] + a[2] * b[6], a[0] * b[1] + a[1] * b[4] + a[2] * b[7], a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
  a[3] * b[0] + a[4] * b[3] + a[5] * b[6], a[3] * b[1] + a[4] * b[4] + a[5] * b[7], a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
  a[6] * b[0] + a[7] * b[3] + a[8] * b[6], a[6] * b[1] + a[7] * b[4] + a[8] * b[7], a[6] * b[2] + a[7] * b[5] + a[8] * b[8]];
const m3T = a => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
const m3v = (a, v) => [a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2], a[6] * v[0] + a[7] * v[1] + a[8] * v[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = a => Math.hypot(a[0], a[1], a[2]);
const unit = a => { const n = norm(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
// 4x4 row-major in Float64Array(16)
function m4mul(out, a, b) {
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    out[4 * i + j] = a[4 * i] * b[j] + a[4 * i + 1] * b[4 + j] + a[4 * i + 2] * b[8 + j] + a[4 * i + 3] * b[12 + j];
  }
  return out;
}
function m4inv(m) {                                   // general 4x4 inverse (rigid here, but keep it general)
  const inv = new Float64Array(16), a = m;
  inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
  inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
  inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
  inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
  inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
  inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
  inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
  inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
  inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
  inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
  inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
  inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
  inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
  inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
  inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
  inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
  const det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
  for (let i = 0; i < 16; i++) inv[i] /= det;
  return inv;
}

// ------------------------------------------------------------------ template asset
export function parseTemplate(buffer) {
  const dv = new DataView(buffer);
  const n = dv.getUint32(0, true);
  const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, n)));
  const base = 4 + n;
  const typed = { float32: Float32Array, int32: Int32Array, int16: Int16Array, uint8: Uint8Array };
  const T = { boneLabels: hdr.bone_labels, regionLabels: hdr.region_labels, hash: hdr.template_hash, restOffset: hdr.rest_offset };
  for (const a of hdr.arrays) {
    const C = typed[a.dtype];
    const copy = buffer.slice(base + a.offset, base + a.offset + a.nbytes);   // aligned copy
    T[a.name] = new C(copy);
    T[a.name + "_shape"] = a.shape;
  }
  T.nV = T.vertices.length / 3; T.nF = T.faces.length / 3; T.nB = T.boneLabels.length; T.nK = T.coco_bone_points_shape[0];
  T.boneIndex = Object.fromEntries(T.boneLabels.map((b, i) => [b, i]));
  return T;
}
export async function loadTemplate(url) { return parseTemplate(await (await fetch(url)).arrayBuffer()); }

// ------------------------------------------------------------------ forward kinematics + skinning
export class FK {
  constructor(T) {
    this.T = T; const B = T.nB;
    this.P = []; this.Pinv = []; this.local = [];
    for (let b = 0; b < B; b++) {
      const P = new Float64Array(16); for (let i = 0; i < 16; i++) P[i] = T.rest_bone_poses[16 * b + i];
      this.P.push(P); this.Pinv.push(m4inv(P));
    }
    for (let b = 0; b < B; b++) {
      const p = T.bone_parents[b];
      this.local.push(p < 0 ? this.P[b] : m4mul(new Float64Array(16), this.Pinv[p], this.P[b]));
    }
    this.rest = new Float64Array(T.vertices.length);
    for (let v = 0; v < T.nV; v++) for (let k = 0; k < 3; k++) this.rest[3 * v + k] = T.vertices[3 * v + k] + T.restOffset[k];
    this._W = Array.from({ length: B }, () => new Float64Array(16));
    this._tmp = new Float64Array(16);
  }
  /** rotvecs: Map/obj boneIndex → [rx,ry,rz]. Returns Float64Array(B*16) of skinning transforms W·P⁻¹. */
  boneTransforms(rotvecs) {
    const B = this.T.nB, out = new Float64Array(B * 16), D = new Float64Array(16);
    for (let c = 0; c < B; c++) {
      const r = rotvecs[c];
      D.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
      if (r) { const m = rotvecToMat(r); D[0] = m[0]; D[1] = m[1]; D[2] = m[2]; D[4] = m[3]; D[5] = m[4]; D[6] = m[5]; D[8] = m[6]; D[9] = m[7]; D[10] = m[8]; }
      const p = this.T.bone_parents[c], W = this._W[c];
      if (p < 0) W.set(D);
      else { m4mul(this._tmp, this._W[p], this.local[c]); m4mul(W, this._tmp, D); }
      m4mul(this._tmp, W, this.Pinv[c]);
      out.set(this._tmp, 16 * c);
    }
    return out;
  }
  /** COCO-23 keypoints (K*3) from skinning transforms: Σ_b T_b p_kb. */
  keypoints(bt) {
    const T = this.T, K = T.nK, B = T.nB, cbp = T.coco_bone_points, out = new Float64Array(K * 3);
    for (let k = 0; k < K; k++) {
      let x = 0, y = 0, z = 0;
      for (let b = 0; b < B; b++) {
        const o = (k * B + b) * 4, px = cbp[o], py = cbp[o + 1], pz = cbp[o + 2], pw = cbp[o + 3];
        if (px === 0 && py === 0 && pz === 0 && pw === 0) continue;
        const t = 16 * b;
        x += bt[t] * px + bt[t + 1] * py + bt[t + 2] * pz + bt[t + 3] * pw;
        y += bt[t + 4] * px + bt[t + 5] * py + bt[t + 6] * pz + bt[t + 7] * pw;
        z += bt[t + 8] * px + bt[t + 9] * py + bt[t + 10] * pz + bt[t + 11] * pw;
      }
      out[3 * k] = x; out[3 * k + 1] = y; out[3 * k + 2] = z;
    }
    return out;
  }
  /** Linear blend skinning of the rest mesh → Float64Array(V*3), optionally scaled. */
  vertices(bt, scale = 1) {
    const T = this.T, V = T.nV, out = new Float64Array(V * 3), vbi = T.vertex_bone_indices, vbw = T.vertex_bone_weights, r = this.rest;
    for (let v = 0; v < V; v++) {
      const x = r[3 * v], y = r[3 * v + 1], z = r[3 * v + 2];
      let ox = 0, oy = 0, oz = 0;
      for (let k = 0; k < 9; k++) {
        const w = vbw[9 * v + k]; if (w === 0) continue;
        const t = 16 * vbi[9 * v + k];
        ox += w * (bt[t] * x + bt[t + 1] * y + bt[t + 2] * z + bt[t + 3]);
        oy += w * (bt[t + 4] * x + bt[t + 5] * y + bt[t + 6] * z + bt[t + 7]);
        oz += w * (bt[t + 8] * x + bt[t + 9] * y + bt[t + 10] * z + bt[t + 11]);
      }
      out[3 * v] = ox * scale; out[3 * v + 1] = oy * scale; out[3 * v + 2] = oz * scale;
    }
    return out;
  }
}

// ------------------------------------------------------------------ camera
export function focalPxFrom35mm(f35, W, H) { return f35 * Math.hypot(W, H) / 43.27; }
export function lookAt(position, target, up = [0, 0, 1]) {
  const fwd = unit([target[0] - position[0], target[1] - position[1], target[2] - position[2]]);
  const right = unit(cross(fwd, up)); const down = cross(fwd, right);
  const R = [...right, ...down, ...fwd];
  const t = m3v(R, position).map(v => -v);
  return { R, t };
}
export function orbitCamera(yawDeg, distance, height, targetZ, W, H, f35 = 26, focalPx = null) {
  const a = yawDeg * Math.PI / 180;
  const pos = [distance * Math.sin(a), -distance * Math.cos(a), height];
  const { R, t } = lookAt(pos, [0, 0, targetZ]);
  return { R, t, f: focalPx || focalPxFrom35mm(f35, W, H), cx: W / 2, cy: H / 2, W, H };
}
export const camPosition = c => m3v(m3T(c.R), c.t).map(v => -v);
export function project(c, X, n) {                 // X Float64Array(n*3) → uv (n*2), z (n)
  const uv = new Float64Array(n * 2), z = new Float64Array(n), R = c.R, t = c.t;
  for (let i = 0; i < n; i++) {
    const x = X[3 * i], y = X[3 * i + 1], w = X[3 * i + 2];
    const xc = R[0] * x + R[1] * y + R[2] * w + t[0], yc = R[3] * x + R[4] * y + R[5] * w + t[1];
    let zc = R[6] * x + R[7] * y + R[8] * w + t[2]; if (zc < 1e-3) zc = 1e-3;
    uv[2 * i] = c.f * xc / zc + c.cx; uv[2 * i + 1] = c.f * yc / zc + c.cy; z[i] = zc;
  }
  return { uv, z };
}
export function pixelRay(c, u, v) {
  const d = [(u - c.cx) / c.f, (v - c.cy) / c.f, 1];
  return { origin: camPosition(c), dir: unit(m3v(m3T(c.R), d)) };
}

// ------------------------------------------------------------------ keypoints
export function keypointsFromMediapipe(landmarks, W, H, protocol) {
  const K = protocol.coco23.length, uv = new Float64Array(K * 2), conf = new Float64Array(K);
  const idx = Object.fromEntries(protocol.coco23.map((l, i) => [l, i]));
  for (const [mi, label] of Object.entries(protocol.mediapipe33_to_coco23)) {
    const lm = landmarks[+mi]; if (!lm) continue;
    const k = idx[label]; uv[2 * k] = lm.x * W; uv[2 * k + 1] = lm.y * H; conf[k] = lm.visibility ?? 1;
  }
  return { uv, conf, W, H, K };
}
export function swapLR(kp, protocol) {
  const K = kp.K, uv = new Float64Array(K * 2), conf = new Float64Array(K);
  for (let k = 0; k < K; k++) { const s = protocol.swap_lr[k]; uv[2 * k] = kp.uv[2 * s]; uv[2 * k + 1] = kp.uv[2 * s + 1]; conf[k] = kp.conf[s]; }
  return { uv, conf, W: kp.W, H: kp.H, K };
}
function bbox(kp, minConf) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, n = 0;
  for (let k = 0; k < kp.K; k++) if (kp.conf[k] >= minConf) { n++; x0 = Math.min(x0, kp.uv[2 * k]); x1 = Math.max(x1, kp.uv[2 * k]); y0 = Math.min(y0, kp.uv[2 * k + 1]); y1 = Math.max(y1, kp.uv[2 * k + 1]); }
  return n ? [[x0, y0], [x1, y1]] : null;
}

// ------------------------------------------------------------------ the fit
function initCamera(nominal, K3, kp, minConf, K) {
  const { uv: uv0, z: z0 } = project(nominal, K3, K);
  let ws = 0, mu0 = [0, 0], mu1 = [0, 0];
  const w = new Float64Array(K);
  for (let k = 0; k < K; k++) { w[k] = kp.conf[k] >= minConf ? kp.conf[k] : 0; ws += w[k]; mu0[0] += w[k] * uv0[2 * k]; mu0[1] += w[k] * uv0[2 * k + 1]; mu1[0] += w[k] * kp.uv[2 * k]; mu1[1] += w[k] * kp.uv[2 * k + 1]; }
  if (ws < 2) return nominal;
  mu0 = mu0.map(v => v / ws); mu1 = mu1.map(v => v / ws);
  let s1 = 0, s0 = 0, zm = 0;
  for (let k = 0; k < K; k++) { s1 += w[k] * ((kp.uv[2 * k] - mu1[0]) ** 2 + (kp.uv[2 * k + 1] - mu1[1]) ** 2); s0 += w[k] * ((uv0[2 * k] - mu0[0]) ** 2 + (uv0[2 * k + 1] - mu0[1]) ** 2); zm += w[k] * z0[k]; }
  const s = Math.sqrt(s1 / Math.max(s0, 1e-9)); zm /= ws;
  const zn = zm / s, t = nominal.t.slice();
  t[2] += zn - zm;
  t[0] += (mu1[0] - nominal.cx) / nominal.f * zn - (mu0[0] - nominal.cx) / nominal.f * zm;
  t[1] += (mu1[1] - nominal.cy) / nominal.f * zn - (mu0[1] - nominal.cy) / nominal.f * zm;
  return { ...nominal, t };
}

function cholSolve(A, b, n) {                       // A symmetric positive definite (n*n), in place
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i * n + j];
    for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
    if (i === j) { if (s <= 0) return null; L[i * n + i] = Math.sqrt(s); } else L[i * n + j] = s / L[j * n + j];
  }
  const y = new Float64Array(n), x = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return x;
}

/** Levenberg–Marquardt on a residual function with numerical Jacobian. */
function lm(resFn, x0, { maxIter = 80, steps, tol = 1e-9 } = {}) {
  const n = x0.length; let x = Float64Array.from(x0), r = resFn(x), m = r.length;
  let cost = 0; for (const v of r) cost += v * v;
  let lambda = 1e-3, iters = 0;
  const J = new Float64Array(m * n), A = new Float64Array(n * n), g = new Float64Array(n);
  for (let it = 0; it < maxIter; it++) {
    iters = it + 1;
    for (let j = 0; j < n; j++) {                   // forward differences
      const h = steps ? steps[j] : 1e-6, xp = Float64Array.from(x); xp[j] += h;
      const rp = resFn(xp);
      for (let i = 0; i < m; i++) J[i * n + j] = (rp[i] - r[i]) / h;
    }
    A.fill(0); g.fill(0);
    for (let i = 0; i < m; i++) for (let a = 0; a < n; a++) {
      const Jia = J[i * n + a]; if (Jia === 0) continue;
      g[a] -= Jia * r[i];
      for (let b = 0; b <= a; b++) A[a * n + b] += Jia * J[i * n + b];
    }
    for (let a = 0; a < n; a++) for (let b = 0; b < a; b++) A[b * n + a] = A[a * n + b];
    let improved = false;
    for (let tries = 0; tries < 12; tries++) {
      const Ad = Float64Array.from(A);
      for (let a = 0; a < n; a++) Ad[a * n + a] += lambda * (A[a * n + a] + 1e-9);
      const d = cholSolve(Ad, g, n);
      if (d) {
        const xn = Float64Array.from(x); for (let a = 0; a < n; a++) xn[a] += d[a];
        const rn = resFn(xn); let cn = 0; for (const v of rn) cn += v * v;
        if (cn < cost) {
          const rel = (cost - cn) / Math.max(cost, 1e-12);
          x = xn; r = rn; cost = cn; lambda = Math.max(lambda / 3, 1e-9); improved = true;
          if (rel < tol) return { x, cost, iters };
          break;
        }
      }
      lambda *= 4;
    }
    if (!improved) break;
  }
  return { x, cost, iters };
}

/**
 * Fit camera + pose for one shot.  kp: {uv, conf, W, H, K}; pose: protocol pose entry.
 * Returns {accepted, reason, R, t, scale, bt, rmsNorm, reprojPx, missing, swapped, loss, lossAlt, cameraPitchDeg, cameraDevDeg, iters, ms}
 */
export function fitShot(T, fk, protocol, pose, kp, focalPx, opts = {}) {
  const o = { minConf: 0.3, lambdaSwing: 0.1, lambdaTwist: 5.0, lambdaScale: 20.0, lambdaCam: 3.0, swapMargin: 0.8,
              acceptRmsNorm: 0.04, maxMissing: null, trySwap: null, ...opts };
  const t0 = (typeof performance !== "undefined" ? performance : Date).now();
  const nominal = orbitCamera(pose.yaw_deg, pose.distance_m, pose.camera_height, pose.target_z, kp.W, kp.H, 26, focalPx);
  const trySwap = o.trySwap ?? Math.abs(((pose.yaw_deg + 180) % 360) - 180) > 60;
  const candidates = [[kp, false]]; if (trySwap) candidates.push([swapLR(kp, protocol), true]);
  const results = candidates.map(([k, sw]) => ({ ...fitOne(T, fk, protocol, pose, k, nominal, o), swapped: sw, kp: k }));
  let best = results[0];
  if (results.length > 1 && results[1].loss < o.swapMargin * results[0].loss) best = results[1];
  const lossAlt = results.length > 1 ? (best === results[0] ? results[1].loss : results[0].loss) : NaN;
  // residuals / gates
  const K = kp.K, K3 = fk.keypoints(best.bt); for (let i = 0; i < K3.length; i++) K3[i] *= best.scale;
  const cam = { ...nominal, R: best.R, t: best.t };
  const { uv } = project(cam, K3, K);
  const reprojPx = new Float64Array(K).fill(NaN); let se = 0, n = 0;
  for (let k = 0; k < K; k++) if (best.kp.conf[k] >= o.minConf) { const e = Math.hypot(uv[2 * k] - best.kp.uv[2 * k], uv[2 * k + 1] - best.kp.uv[2 * k + 1]); reprojPx[k] = e; se += e * e; n++; }
  const bb = bbox(best.kp, o.minConf), bboxH = bb ? bb[1][1] - bb[0][1] : kp.H;
  const rmsNorm = n ? Math.sqrt(se / n) / Math.max(bboxH, 1) : NaN;
  const idx = Object.fromEntries(protocol.coco23.map((l, i) => [l, i]));
  const missing = pose.required_keypoints.filter(l => best.kp.conf[idx[l]] < 0.5);
  const core = protocol.core_keypoints.filter(l => pose.required_keypoints.includes(l));
  const missingCore = missing.filter(l => core.includes(l));
  const maxMissing = o.maxMissing ?? Math.max(1, Math.floor(0.2 * pose.required_keypoints.length));
  const Rd = m3mul(best.R, m3T(nominal.R));
  const cameraDevDeg = norm(matToRotvec(Rd)) * 180 / Math.PI;
  const cameraPitchDeg = Math.asin(Math.max(-1, Math.min(1, -best.R[8]))) * 180 / Math.PI;
  let reason = "";
  if (missingCore.length) reason = `core keypoints not detected: ${missingCore.join(", ")}`;
  else if (missing.length > maxMissing) reason = `too many required keypoints missing (${missing.length} > ${maxMissing}): ${missing.join(", ")}`;
  else if (rmsNorm > o.acceptRmsNorm) reason = `pose fit residual too high (${(rmsNorm * 100).toFixed(1)}% of body height)`;
  return { accepted: !reason, reason, R: best.R, t: best.t, scale: best.scale, bt: best.bt, camera: cam, keypoints3d: K3, kpUsed: best.kp,
           rmsNorm, reprojPx, missing, swapped: best.swapped, loss: best.loss, lossAlt, cameraPitchDeg, cameraDevDeg, iters: best.iters,
           ms: (typeof performance !== "undefined" ? performance : Date).now() - t0, bboxH };
}

function fitOne(T, fk, protocol, pose, kp, nominal, o) {
  const K = kp.K, FB = protocol.fit_bones, nb = FB.length;
  const std = pose.joint_rotvecs;
  const fixed = {}; for (const [b, r] of Object.entries(std)) if (!FB.includes(b)) fixed[T.boneIndex[b]] = r;
  const theta0 = new Float64Array(nb * 3);
  FB.forEach((b, i) => { const r = std[b] || [0, 0, 0]; theta0[3 * i] = r[0]; theta0[3 * i + 1] = r[1]; theta0[3 * i + 2] = r[2]; });
  const fbIdx = FB.map(b => T.boneIndex[b]);
  const rotOf = theta => { const r = { ...fixed }; for (let i = 0; i < nb; i++) r[fbIdx[i]] = [theta[3 * i], theta[3 * i + 1], theta[3 * i + 2]]; return r; };
  // camera init
  const K30 = fk.keypoints(fk.boneTransforms(rotOf(theta0)));
  const cam0 = initCamera(nominal, K30, kp, o.minConf, K);
  const camR0 = matToRotvec(m3mul(cam0.R, m3T(nominal.R)));
  const w = new Float64Array(K); let ws = 0;
  for (let k = 0; k < K; k++) { w[k] = kp.conf[k] >= o.minConf ? kp.conf[k] : 0; ws += w[k]; } ws = Math.max(ws, 1e-6);
  const zRef = Math.max(norm(camPosition(cam0)), 0.3);
  const sigma = protocol.kp_sigma_m.map(s => Math.max(s * nominal.f / zRef, 2.0));
  const lam = FB.map(b => [o.lambdaSwing, protocol.twist_observable.includes(b) ? o.lambdaSwing : o.lambdaTwist, o.lambdaSwing]);
  const fnh = [nominal.R[6], nominal.R[7]]; const fnn = Math.hypot(fnh[0], fnh[1]); fnh[0] /= fnn; fnh[1] /= fnn;
  const sqrt = Math.sqrt;
  // parameter vector: [camR(3), camT(3), logS(1), theta(nb*3)]
  const x0 = new Float64Array(7 + nb * 3); x0.set(camR0, 0); x0.set(cam0.t, 3); x0[6] = 0; x0.set(theta0, 7);
  const steps = new Float64Array(x0.length).fill(1e-6); steps[3] = steps[4] = steps[5] = 1e-5;
  const nRes = K + nb * 3 + 1 + 2 + 1 + 3 + 1;
  let lastBt = null;
  const residuals = x => {
    const r = new Float64Array(nRes); let p = 0;
    const theta = x.subarray(7);
    const bt = fk.boneTransforms(rotOf(theta)); lastBt = bt;
    const K3 = fk.keypoints(bt); const s = Math.exp(x[6]); for (let i = 0; i < K3.length; i++) K3[i] *= s;
    const R = m3mul(rotvecToMat([x[0], x[1], x[2]]), nominal.R);
    const cam = { ...nominal, R, t: [x[3], x[4], x[5]] };
    const { uv } = project(cam, K3, K);
    for (let k = 0; k < K; k++) {
      const d = Math.hypot(uv[2 * k] - kp.uv[2 * k], uv[2 * k + 1] - kp.uv[2 * k + 1]) / sigma[k];
      const hub = d < 1 ? 0.5 * d * d : d - 0.5;             // Huber; data = Σ w·hub / Σw
      r[p++] = sqrt(2 * hub * w[k] / ws);
    }
    for (let i = 0; i < nb; i++) for (let a = 0; a < 3; a++) r[p++] = sqrt(lam[i][a]) * (theta[3 * i + a] - theta0[3 * i + a]);
    r[p++] = sqrt(o.lambdaScale) * x[6];
    // camera prior: yaw (2 residuals), roll (1), pitch weak (3), pitch wall (1)
    const fh = [R[6], R[7]], cp = Math.hypot(fh[0], fh[1]) + 1e-9, fhu = [fh[0] / cp, fh[1] / cp];
    r[p++] = sqrt(o.lambdaCam) * (fhu[0] - fnh[0]); r[p++] = sqrt(o.lambdaCam) * (fhu[1] - fnh[1]);
    r[p++] = sqrt(o.lambdaCam) * R[2];
    const sp = sqrt(0.1 * o.lambdaCam);
    r[p++] = sp * (R[6] - fhu[0]); r[p++] = sp * (R[7] - fhu[1]); r[p++] = sp * R[8];
    r[p++] = sqrt(50) * Math.max(0, Math.abs(R[8]) - 0.7071);
    return r;
  };
  const { x, cost, iters } = lm(residuals, x0, { steps });
  const r = residuals(x);                                 // refresh lastBt at the solution
  const R = m3mul(rotvecToMat([x[0], x[1], x[2]]), nominal.R);
  return { loss: cost / 2, R, t: [x[3], x[4], x[5]], scale: Math.exp(x[6]), bt: lastBt, iters, theta: Array.from(x.subarray(7)) };
}

// ------------------------------------------------------------------ ray cast → anchors
function faceNormal(V, F, f) {
  const a = 3 * F[3 * f], b = 3 * F[3 * f + 1], c = 3 * F[3 * f + 2];
  const e1 = [V[b] - V[a], V[b + 1] - V[a + 1], V[b + 2] - V[a + 2]], e2 = [V[c] - V[a], V[c + 1] - V[a + 1], V[c + 2] - V[a + 2]];
  return unit(cross(e1, e2));
}
/** Nearest front-facing hit along the ray, plus the depth gap to the next front-facing surface behind it. */
export function castRay(V, F, nF, origin, dir) {
  const hits = [];
  for (let f = 0; f < nF; f++) {
    const a = 3 * F[3 * f], b = 3 * F[3 * f + 1], c = 3 * F[3 * f + 2];
    const e1 = [V[b] - V[a], V[b + 1] - V[a + 1], V[b + 2] - V[a + 2]], e2 = [V[c] - V[a], V[c + 1] - V[a + 1], V[c + 2] - V[a + 2]];
    const p = cross(dir, e2), det = dot(e1, p);
    if (Math.abs(det) < 1e-12) continue;
    const inv = 1 / det, tv = [origin[0] - V[a], origin[1] - V[a + 1], origin[2] - V[a + 2]];
    const u = dot(tv, p) * inv; if (u < 0 || u > 1) continue;
    const q = cross(tv, e1), v = dot(dir, q) * inv; if (v < 0 || u + v > 1) continue;
    const t = dot(e2, q) * inv; if (t <= 1e-6) continue;
    const n = unit(cross(e1, e2)); const cosv = -dot(n, dir);
    hits.push({ f, t, bary: [1 - u - v, u, v], cosView: cosv });
  }
  if (!hits.length) return null;
  hits.sort((a, b) => a.t - b.t);
  const front = hits.filter(h => h.cosView > 0);
  const h = front.length ? front[0] : hits[0];
  let layerGap = Infinity;
  for (const g of front) if (g.t > h.t + 0.02) { layerGap = g.t - h.t; break; }
  return { ...h, cosView: Math.abs(h.cosView), layerGap, xyz: [origin[0] + dir[0] * h.t, origin[1] + dir[1] * h.t, origin[2] + dir[2] * h.t] };
}

/**
 * Lift tapped mole pixels for one shot → anchors with confidence (mirrors lift.py; no silhouette-distance factor).
 * molesPx: [[x,y], ...] in the upright photo.  Returns {fit, anchors:[{anchor:{face_id,barycentric}|null, region, xyz_m, quality}]}
 */
export function liftShot(T, fk, protocol, pose, kp, molesPx, focalPx, opts = {}) {
  const fit = fitShot(T, fk, protocol, pose, kp, focalPx, opts);
  const V = fk.vertices(fit.bt, fit.scale), F = T.faces, nF = T.nF;
  const idx = Object.fromEntries(protocol.coco23.map((l, i) => [l, i]));
  const missing = new Set(fit.missing);
  const anchors = [];
  for (const [x, y] of molesPx) {
    // try the pixel, then snap outward in rings (the tap may sit just off the silhouette)
    let hit = null, snapped = 0;
    outer: for (const rad of [0, 6, 12, 20]) {
      const n = rad ? 8 : 1;
      for (let i = 0; i < n; i++) {
        const a = 2 * Math.PI * i / n, u = x + rad * Math.cos(a), v = y + rad * Math.sin(a);
        const { origin, dir } = pixelRay(fit.camera, u, v);
        hit = castRay(V, F, nF, origin, dir);
        if (hit) { snapped = rad; break outer; }
      }
    }
    if (!hit) { anchors.push({ anchor: null, region: null, xyz_m: null, quality: { confidence: 0, reason: "off body" } }); continue; }
    const region = T.regionLabels[T.face_region[hit.f]];
    const gov = protocol.region_keypoints[region] || [];
    const govRes = gov.map(l => fit.reprojPx[idx[l]] / Math.max(fit.bboxH, 1)).filter(Number.isFinite);
    const localNorm = govRes.length ? Math.sqrt(govRes.reduce((s, v) => s + v * v, 0) / govRes.length) : fit.rmsNorm;
    const govMissing = gov.filter(l => missing.has(l));
    const layered = Number.isFinite(hit.layerGap);
    const inScope = pose.regions.includes(region);
    const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const conf = Math.sqrt(clip(1 - fit.rmsNorm / 0.04, 0, 1)) * Math.sqrt(clip(1 - localNorm / 0.05, 0, 1))
      * clip(hit.cosView / 0.5, 0, 1) * (layered ? 0.5 : 1) * (inScope ? 1 : 0.25) * (govMissing.length ? 0.5 : 1) * (snapped ? 0.6 : 1);
    // template-space position of the anchor (for matching / the viewer): barycentric on the template face
    const a = 3 * F[3 * hit.f], b = 3 * F[3 * hit.f + 1], c = 3 * F[3 * hit.f + 2], tv = T.vertices, B = hit.bary;
    const xyz = [0, 1, 2].map(k => B[0] * tv[a + k] + B[1] * tv[b + k] + B[2] * tv[c + k]);
    anchors.push({ anchor: { face_id: hit.f, barycentric: B }, region, xyz_m: xyz,
      quality: { confidence: +conf.toFixed(3), cos_view: +hit.cosView.toFixed(3), layered, layer_gap_m: layered ? +hit.layerGap.toFixed(3) : null,
                 in_scope: inScope, missing_keypoints: govMissing, local_rms_norm: +localNorm.toFixed(4), snapped_px: snapped,
                 fit_rms_norm: +fit.rmsNorm.toFixed(4), px_per_mm: +(fit.camera.f / hit.t / 1000).toFixed(2) } });
  }
  return { fit, anchors };
}

/** Patient map (step-1 schema, minimal) from a capture export — same shape as scripts/lift_capture.py. */
export function liftExport(T, fk, protocol, exportObj, onShot = null) {
  const poses = Object.fromEntries(protocol.poses.map(p => [p.id, p]));
  const moles = [], shots = [];
  const now = new Date().toISOString();
  for (const shot of exportObj.shots) {
    const pose = poses[shot.pose_id]; if (!pose) continue;
    const [W, H] = shot.image_size;
    const kp = keypointsFromMediapipe(shot.landmarks.map(l => ({ x: l[0], y: l[1], visibility: l[2] })), W, H, protocol);
    const f = focalPxFrom35mm(+(exportObj.focal35 || 26), W, H);
    const res = liftShot(T, fk, protocol, pose, kp, shot.moles || [], f);
    const fit = res.fit;
    shots.push({ pose_id: shot.pose_id, accepted: fit.accepted, reason: fit.reason, rms_norm: fit.rmsNorm, pitch_deg: fit.cameraPitchDeg, swapped: fit.swapped, ms: fit.ms, iters: fit.iters });
    res.anchors.forEach((a, k) => {
      if (!a.anchor) return;
      const [x, y] = shot.moles[k];
      moles.push({ mole_id: `${shot.pose_id}_${String(k + 1).padStart(2, "0")}`,
        anchor: { face_id: a.anchor.face_id, bary: a.anchor.barycentric, uv: [0, 0], xyz_m: a.xyz_m, region_code: a.region, region_name: a.region,
                  source: `keypoints:${shot.pose_id}`, confidence: a.quality.confidence },
        status: "active", merged_into: null, first_seen_visit: "v1", label: null, shot_accepted: fit.accepted,
        observations: [{ visit_id: "v1", detection: { shot: shot.pose_id, px: [x, y] }, measurement: null, thumbnail_sha256: null, neighbour_signature: [], match_confidence: 1.0, notes: null }] });
    });
    if (onShot) onShot(shots[shots.length - 1], res);
  }
  return {
    patient_map: { schema_version: "0.1.0", patient_id: exportObj.patient_id || "capture",
      body_model: { name: "anny", version: "0.6.2", topology: "default", mesh_sha256: T.hash },
      shape: { tier: "generic", phenotypes: null, height_m: exportObj.height_m || null, fitted_at: null },
      visits: [{ visit_id: "v1", date: now.slice(0, 10), capture: "phone-web", device: (typeof navigator !== "undefined" ? navigator.userAgent : "node"), app_version: "0.3.0", notes: null }],
      moles, created_at: now, updated_at: now },
    shots,
  };
}
