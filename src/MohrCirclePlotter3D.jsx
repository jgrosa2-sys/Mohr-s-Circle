import React, { useMemo, useState } from "react";

function Card({ className = "", children }) {
  return <div className={className}>{children}</div>;
}

function CardHeader({ className = "", children }) {
  return <div className={className}>{children}</div>;
}

function CardContent({ className = "", children }) {
  return <div className={className}>{children}</div>;
}

function CardTitle({ className = "", children }) {
  return <h2 className={className}>{children}</h2>;
}

function Input({ className = "", ...props }) {
  return (
    <input
      {...props}
      className={`w-full rounded-xl border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-500 ${className}`}
    />
  );
}

function Button({ className = "", variant = "default", children, ...props }) {
  const base = "rounded-2xl border px-3 py-2 text-sm transition";
  const styles =
    variant === "outline"
      ? "border-slate-300 bg-white text-slate-900 hover:bg-slate-50"
      : "border-slate-900 bg-slate-900 text-white hover:bg-slate-800";
  return (
    <button {...props} className={`${base} ${styles} ${className}`}>
      {children}
    </button>
  );
}

const DEG = Math.PI / 180;

function det3(m) {
  return (
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  );
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize(v) {
  const n = norm(v);
  if (n < 1e-12) return [0, 0, 0];
  return [v[0] / n, v[1] / n, v[2] / n];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function matVec(m, v) {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

function orthonormalize(v, basis = []) {
  let out = [...v];
  basis.forEach((b) => {
    const proj = dot(out, b);
    out = sub(out, scale(b, proj));
  });
  const n = norm(out);
  if (n < 1e-10) return null;
  return scale(out, 1 / n);
}

function fallbackOrthogonal(v) {
  const candidates = [cross(v, [1, 0, 0]), cross(v, [0, 1, 0]), cross(v, [0, 0, 1])];
  const best = candidates.sort((a, b) => norm(b) - norm(a))[0];
  return normalize(best);
}

function eigenvaluesSymmetric3x3(m) {
  const a = m[0][0];
  const b = m[1][1];
  const c = m[2][2];
  const d = m[0][1];
  const e = m[0][2];
  const f = m[1][2];

  const p1 = d * d + e * e + f * f;
  if (Math.abs(p1) < 1e-14) {
    return [a, b, c].sort((x, y) => y - x);
  }

  const q = (a + b + c) / 3;
  const a11 = a - q;
  const b11 = b - q;
  const c11 = c - q;
  const p2 = a11 * a11 + b11 * b11 + c11 * c11 + 2 * p1;
  const p = Math.sqrt(p2 / 6);

  const B = [
    [a11 / p, d / p, e / p],
    [d / p, b11 / p, f / p],
    [e / p, f / p, c11 / p],
  ];

  let r = det3(B) / 2;
  r = Math.max(-1, Math.min(1, r));

  const phi = Math.acos(r) / 3;
  const lambda1 = q + 2 * p * Math.cos(phi);
  const lambda3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  const lambda2 = 3 * q - lambda1 - lambda3;

  return [lambda1, lambda2, lambda3].sort((x, y) => y - x);
}

function eigenvectorForValue(m, lambda) {
  const A = [
    [m[0][0] - lambda, m[0][1], m[0][2]],
    [m[1][0], m[1][1] - lambda, m[1][2]],
    [m[2][0], m[2][1], m[2][2] - lambda],
  ];

  const candidates = [cross(A[0], A[1]), cross(A[0], A[2]), cross(A[1], A[2])].sort((u, v) => norm(v) - norm(u));
  if (norm(candidates[0]) > 1e-10) {
    return normalize(candidates[0]);
  }

  const rows = [...A].sort((u, v) => norm(u) - norm(v));
  if (norm(rows[0]) > 1e-10) {
    return fallbackOrthogonal(normalize(rows[0]));
  }

  return [1, 0, 0];
}

function eigenbasisSymmetric3x3(m, eigenvalues) {
  const v1 = orthonormalize(eigenvectorForValue(m, eigenvalues[0])) || [1, 0, 0];
  let v2 = orthonormalize(eigenvectorForValue(m, eigenvalues[1]), [v1]);
  if (!v2) {
    v2 = orthonormalize(fallbackOrthogonal(v1), [v1]) || [0, 1, 0];
  }
  let v3 = normalize(cross(v1, v2));
  if (norm(v3) < 1e-10) {
    v3 = orthonormalize(eigenvectorForValue(m, eigenvalues[2]), [v1, v2]) || normalize(cross(v1, v2));
  }
  return [v1, v2, v3];
}

function principal2D(a, b, tau) {
  const center = (a + b) / 2;
  const radius = Math.hypot((a - b) / 2, tau);
  return {
    sigma1: center + radius,
    sigma2: center - radius,
    center,
    radius,
    tauMax: radius,
  };
}

function transform2D(a, b, tau, thetaRad) {
  const c = (a + b) / 2;
  const d = (a - b) / 2;
  const c2 = Math.cos(2 * thetaRad);
  const s2 = Math.sin(2 * thetaRad);
  return {
    sxp: c + d * c2 + tau * s2,
    syp: c - d * c2 - tau * s2,
    txpyp: -d * s2 + tau * c2,
  };
}

function wrap180(deg) {
  const out = ((deg % 180) + 180) % 180;
  return Math.abs(out - 180) < 1e-9 ? 0 : out;
}

function principalAngles2D(a, b, tau) {
  let theta = (0.5 * Math.atan2(2 * tau, a - b)) / DEG;
  theta = wrap180(theta);

  const sHere = transform2D(a, b, tau, theta * DEG).sxp;
  const sPerp = transform2D(a, b, tau, (theta + 90) * DEG).sxp;
  if (sHere < sPerp) {
    theta = wrap180(theta + 90);
  }

  return {
    maxPrincipal: theta,
    minPrincipal: wrap180(theta + 90),
    maxShearPositive: wrap180(theta + 45),
    maxShearNegative: wrap180(theta + 135),
  };
}

function transformPrincipal13(sigma1, sigma2, sigma3, thetaDeg) {
  const theta = thetaDeg * DEG;
  const c = (sigma1 + sigma3) / 2;
  const r = (sigma1 - sigma3) / 2;
  return {
    sxp: c + r * Math.cos(2 * theta),
    syp: sigma2,
    szp: c - r * Math.cos(2 * theta),
    txpzp: -r * Math.sin(2 * theta),
  };
}

function rotateBasis13(basis, thetaDeg) {
  const theta = thetaDeg * DEG;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const e1 = basis[0];
  const e2 = basis[1];
  const e3 = basis[2];
  return [add(scale(e1, c), scale(e3, s)), e2, add(scale(e1, -s), scale(e3, c))];
}

function tensorInBasis(tensorGlobal, basis) {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      out[i][j] = dot(basis[i], matVec(tensorGlobal, basis[j]));
    }
  }
  return out;
}

function tensorMaxAbs(t) {
  let m = 0;
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      m = Math.max(m, Math.abs(t[i][j]));
    }
  }
  return Math.max(m, 1);
}

function niceStep(range, targetTicks = 6) {
  const rough = Math.max(range / targetTicks, 1e-9);
  const power = Math.pow(10, Math.floor(Math.log10(rough)));
  const normalized = rough / power;

  let nice;
  if (normalized <= 1) nice = 1;
  else if (normalized <= 2) nice = 2;
  else if (normalized <= 5) nice = 5;
  else nice = 10;

  return nice * power;
}

function buildTicks(min, max, step) {
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step * 0.5; v += step) {
    ticks.push(Number(v.toFixed(10)));
  }
  return ticks;
}

function format(n) {
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1000 || (Math.abs(n) > 0 && Math.abs(n) < 0.01)) {
    return n.toExponential(1);
  }
  return n.toFixed(1);
}

function formatAngle(n) {
  return `${Number(wrap180(n).toFixed(1))}°`;
}

function formatSignedAngle(n) {
  return `${n < 0 ? "-" : ""}${formatAngle(Math.abs(n))}`;
}

function formatVector(v) {
  return `(${format(v[0])}, ${format(v[1])}, ${format(v[2])})`;
}

function isZero(v, tol = 1e-10) {
  return Math.abs(v) < tol;
}

function detectMode(stress) {
  if (isZero(stress.sz) && isZero(stress.txz) && isZero(stress.tyz)) {
    return {
      type: "2d",
      plane: "xy",
      labels: ["X", "Y"],
      axisNames: ["x", "y"],
      a: stress.sx,
      b: stress.sy,
      tau: stress.txy,
    };
  }

  if (isZero(stress.sy) && isZero(stress.txy) && isZero(stress.tyz)) {
    return {
      type: "2d",
      plane: "xz",
      labels: ["X", "Z"],
      axisNames: ["x", "z"],
      a: stress.sx,
      b: stress.sz,
      tau: stress.txz,
    };
  }

  if (isZero(stress.sx) && isZero(stress.txy) && isZero(stress.txz)) {
    return {
      type: "2d",
      plane: "yz",
      labels: ["Y", "Z"],
      axisNames: ["y", "z"],
      a: stress.sy,
      b: stress.sz,
      tau: stress.tyz,
    };
  }

  return { type: "3d" };
}

function mohrShearSignForPlane(plane) {
  return plane === "xz" ? 1 : -1;
}

const examples = {
  "2D XY": { sx: 50, sy: -80, sz: 0, txy: -25, tyz: 0, txz: 0 },
  "2D XZ": { sx: 0, sy: 110, sz: -30, txy: 0, tyz: 40, txz: 0 },
  "2D YZ": { sx: 85, sy: 0, sz: -10, txy: 0, tyz: 0, txz: 28 },
  "Full 3D": { sx: 60, sy: 140, sz: -20, txy: 35, tyz: -15, txz: 20 },
};

function Field({ label, value, onChange }) {
  return (
    <label className="space-y-1">
      <div className="text-sm font-medium text-slate-700">{label}</div>
      <Input type="number" step="any" value={value} onChange={(e) => onChange(Number(e.target.value))} className="bg-white" />
    </label>
  );
}

function TensorTable({ m }) {
  return (
    <div className="overflow-auto rounded-2xl border bg-white">
      <table className="w-full text-sm">
        <tbody>
          {m.map((row, i) => (
            <tr key={i} className="border-b last:border-b-0">
              {row.map((v, j) => (
                <td key={`${i}-${j}`} className="px-4 py-3 text-center font-mono">
                  {format(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatCard({ title, value, subtitle }) {
  return (
    <div className="rounded-2xl border bg-white p-4 shadow-sm">
      <div className="text-sm text-slate-500">{title}</div>
      <div className="mt-1 text-2xl font-semibold text-slate-900">{value}</div>
      {subtitle ? <div className="mt-1 text-xs text-slate-500">{subtitle}</div> : null}
    </div>
  );
}

function AngleSlider({ label, value, onChange, buttons, rotationSense, onRotationSenseChange }) {
  return (
    <div className="space-y-3 rounded-2xl border bg-white p-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="text-sm font-medium text-slate-700">{label}</div>
          <div className="text-xs text-slate-500">The moving diameter on the Mohr circle rotates by 2θ.</div>
        </div>
        <div className="text-2xl font-semibold text-slate-900">{formatAngle(value)}</div>
      </div>
      <input type="range" min="0" max="180" step="0.1" value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full" />
      <div className="flex flex-wrap gap-2">
        <Button variant={rotationSense === "ccw" ? "default" : "outline"} className="rounded-2xl" onClick={() => onRotationSenseChange("ccw")}>
          Counterclockwise
        </Button>
        <Button variant={rotationSense === "cw" ? "default" : "outline"} className="rounded-2xl" onClick={() => onRotationSenseChange("cw")}>
          Clockwise
        </Button>
      </div>
      <div className="flex flex-wrap gap-2">
        {buttons.map((item) => (
          <Button key={item.label} variant="outline" className="rounded-2xl" onClick={() => onChange(item.value)}>
            {item.label}
          </Button>
        ))}
      </div>
    </div>
  );
}

function ViewControls({ azim, elev, setAzim, setElev }) {
  return (
    <div className="space-y-4 rounded-2xl border bg-white p-4">
      <div>
        <div className="text-sm font-medium text-slate-700">3D coordinate view</div>

      </div>
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm text-slate-700">
          <span>Azimuth</span>
          <span>{formatAngle(azim)}</span>
        </div>
        <input type="range" min="-90" max="90" step="1" value={azim} onChange={(e) => setAzim(Number(e.target.value))} className="w-full" />
      </div>
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm text-slate-700">
          <span>Elevation</span>
          <span>{elev.toFixed(0)}°</span>
        </div>
        <input type="range" min="-75" max="75" step="1" value={elev} onChange={(e) => setElev(Number(e.target.value))} className="w-full" />
      </div>
      <Button variant="outline" className="rounded-2xl" onClick={() => { setAzim(0); setElev(0); }}>
        Reset view
      </Button>
    </div>
  );
}

function AxisFrame({ width, height, xmin, xmax, ymin, ymax, children }) {
  const pad = { top: 24, right: 24, bottom: 54, left: 88 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const scaleVal = Math.min(innerW / (xmax - xmin), innerH / (ymax - ymin));
  const usedW = (xmax - xmin) * scaleVal;
  const usedH = (ymax - ymin) * scaleVal;
  const offsetX = pad.left + (innerW - usedW) / 2;
  const offsetY = pad.top + (innerH - usedH) / 2;

  const mapX = (x) => offsetX + (x - xmin) * scaleVal;
  const mapY = (y) => offsetY + (ymax - y) * scaleVal;

  const xStep = niceStep(xmax - xmin, 7);
  const yStep = niceStep(ymax - ymin, 6);
  const xTicks = buildTicks(xmin, xmax, xStep);
  const yTicks = buildTicks(ymin, ymax, yStep);

  const xZeroInView = xmin <= 0 && xmax >= 0;
  const yZeroInView = ymin <= 0 && ymax >= 0;

  return (
    <div className="rounded-2xl border bg-white p-3 shadow-sm">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full">
        <defs>
          <linearGradient id="bgFade" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f8fafc" />
            <stop offset="100%" stopColor="#ffffff" />
          </linearGradient>
        </defs>

        <rect x="0" y="0" width={width} height={height} rx="20" fill="url(#bgFade)" />

        {xTicks.map((t) => (
          <g key={`x-${t}`}>
            <line x1={mapX(t)} x2={mapX(t)} y1={mapY(ymin)} y2={mapY(ymax)} stroke="#e2e8f0" strokeWidth="1" />
            <text x={mapX(t)} y={height - 18} textAnchor="middle" fontSize="12" fill="#475569">
              {format(t)}
            </text>
          </g>
        ))}

        {yTicks.map((t) => (
          <g key={`y-${t}`}>
            <line x1={mapX(xmin)} x2={mapX(xmax)} y1={mapY(t)} y2={mapY(t)} stroke="#e2e8f0" strokeWidth="1" />
            <text x={46} y={mapY(t) + 4} textAnchor="start" fontSize="12" fill="#475569">
              {format(t)}
            </text>
          </g>
        ))}

        {yZeroInView ? <line x1={mapX(xmin)} x2={mapX(xmax)} y1={mapY(0)} y2={mapY(0)} stroke="#0f172a" strokeWidth="1.5" /> : null}
        {xZeroInView ? <line x1={mapX(0)} x2={mapX(0)} y1={mapY(ymin)} y2={mapY(ymax)} stroke="#0f172a" strokeWidth="1.5" /> : null}

        {children({ mapX, mapY })}

        <text x={width / 2} y={height - 4} textAnchor="middle" fontSize="13" fill="#334155">
          Normal stress, σ
        </text>
        <text x="14" y={height / 2} textAnchor="middle" fontSize="13" fill="#334155" transform={`rotate(-90 14 ${height / 2})`}>
          Shear stress, τ
        </text>
      </svg>
    </div>
  );
}

function TwoDMohrSVG({ plane, labels, a, b, tau, thetaDeg }) {
  const width = 1040;
  const height = 620;

  const shearSign = mohrShearSignForPlane(plane);
  const center = (a + b) / 2;
  const radius = Math.hypot((a - b) / 2, tau);
  const originalA = { x: a, y: shearSign * tau, label: labels[0] };
  const originalB = { x: b, y: -shearSign * tau, label: labels[1] };
  const transformed = transform2D(a, b, tau, -thetaDeg * DEG);
  const currentA = { x: transformed.sxp, y: shearSign * transformed.txpyp, label: `${labels[0]}'` };
  const currentB = { x: transformed.syp, y: -shearSign * transformed.txpyp, label: `${labels[1]}'` };
  const sigma1 = center + radius;
  const sigma2 = center - radius;

  let xmin = Math.min(sigma2, originalA.x, originalB.x, currentA.x, currentB.x, 0);
  let xmax = Math.max(sigma1, originalA.x, originalB.x, currentA.x, currentB.x, 0);
  let ymin = Math.min(-radius, originalA.y, originalB.y, currentA.y, currentB.y, 0);
  let ymax = Math.max(radius, originalA.y, originalB.y, currentA.y, currentB.y, 0);

  if (Math.abs(xmax - xmin) < 1e-9) {
    xmin -= 1;
    xmax += 1;
  }
  if (Math.abs(ymax - ymin) < 1e-9) {
    ymin -= 1;
    ymax += 1;
  }

  const xPad = 0.08 * (xmax - xmin || 1);
  const yPad = 0.12 * (ymax - ymin || 1);
  xmin -= xPad;
  xmax += xPad;
  ymin -= yPad;
  ymax += yPad;

  return (
    <AxisFrame width={width} height={height} xmin={xmin} xmax={xmax} ymin={ymin} ymax={ymax}>
      {({ mapX, mapY }) => {
        const rPx = Math.abs(mapX(center + radius) - mapX(center));
        const drawPoint = (p, fill, textColor) => {
          const vx = p.x - center;
          const vy = p.y;
          const mag = Math.hypot(vx, vy) || 1;
          const ux = vx / mag;
          const uy = vy / mag;
          const tangentX = -uy;
          const tangentY = ux;
          const radialPx = -20;
          const tangentSign = p.x >= center ? 1 : -1;
          const tangentPx = tangentSign * 14;
          const tx = mapX(p.x) + radialPx * ux + tangentPx * tangentX;
          const ty = mapY(p.y) - radialPx * uy - tangentPx * tangentY;

          return (
            <g key={p.label}>
              <circle cx={mapX(p.x)} cy={mapY(p.y)} r="5" fill={fill} />
              <text x={tx} y={ty} fontSize="13" fill={textColor}>
                {p.label}
              </text>
            </g>
          );
        };

        return (
          <>
            <ellipse cx={mapX(center)} cy={mapY(0)} rx={rPx} ry={rPx} fill="none" stroke="#0f172a" strokeWidth="2.2" />
            <line x1={mapX(originalA.x)} y1={mapY(originalA.y)} x2={mapX(originalB.x)} y2={mapY(originalB.y)} stroke="#94a3b8" strokeWidth="2" strokeDasharray="8 6" />
            <line x1={mapX(currentA.x)} y1={mapY(currentA.y)} x2={mapX(currentB.x)} y2={mapY(currentB.y)} stroke="#2563eb" strokeWidth="3" />

            {drawPoint(originalA, "#0f172a", "#0f172a")}
            {drawPoint(originalB, "#0f172a", "#0f172a")}
            {drawPoint(currentA, "#2563eb", "#1d4ed8")}
            {drawPoint(currentB, "#2563eb", "#1d4ed8")}

            {[sigma1, sigma2].map((s, i) => (
              <g key={`principal-${i}`}>
                <circle cx={mapX(s)} cy={mapY(0)} r="4.5" fill="#0f172a" />
                <text x={mapX(s) + (i === 0 ? 18 : -18)} y={mapY(0) - 12} textAnchor="middle" fontSize="12" fill="#0f172a">
                  {`σ${i === 0 ? 1 : 2}`}
                </text>
              </g>
            ))}
          </>
        );
      }}
    </AxisFrame>
  );
}

function ThreeDMohrSVG({ sigma1, sigma2, sigma3, thetaDeg }) {
  const width = 1040;
  const height = 620;

  const circles = [
    { name: "σ1–σ2", c: (sigma1 + sigma2) / 2, r: Math.abs(sigma1 - sigma2) / 2 },
    { name: "σ2–σ3", c: (sigma2 + sigma3) / 2, r: Math.abs(sigma2 - sigma3) / 2 },
    { name: "σ1–σ3", c: (sigma1 + sigma3) / 2, r: Math.abs(sigma1 - sigma3) / 2 },
  ];

  const active = transformPrincipal13(sigma1, sigma2, sigma3, thetaDeg);
  const pA = { x: active.sxp, y: active.txpzp, label: "X'" };
  const pB = { x: active.szp, y: -active.txpzp, label: "Z'" };
  const outerCenter = (sigma1 + sigma3) / 2;

  const xCandidates = circles.flatMap(({ c, r }) => [c - r, c + r, c, pA.x, pB.x, 0]);
  const yMaxBase = Math.max(...circles.map((circle) => circle.r), Math.abs(pA.y), Math.abs(pB.y), 1);

  let xmin = Math.min(...xCandidates);
  let xmax = Math.max(...xCandidates);
  let ymin = -1.18 * yMaxBase;
  let ymax = 1.18 * yMaxBase;

  if (Math.abs(xmax - xmin) < 1e-9) {
    xmin -= 1;
    xmax += 1;
  }

  const xPad = 0.08 * (xmax - xmin || 1);
  xmin -= xPad;
  xmax += xPad;

  return (
    <AxisFrame width={width} height={height} xmin={xmin} xmax={xmax} ymin={ymin} ymax={ymax}>
      {({ mapX, mapY }) => (
        <>
          {circles.map((circle) => {
            const rPx = Math.abs(mapX(circle.c + circle.r) - mapX(circle.c));
            return <ellipse key={circle.name} cx={mapX(circle.c)} cy={mapY(0)} rx={rPx} ry={rPx} fill="none" stroke="#0f172a" strokeWidth="2.2" />;
          })}

          <line x1={mapX(sigma1)} y1={mapY(0)} x2={mapX(sigma3)} y2={mapY(0)} stroke="#94a3b8" strokeWidth="2" strokeDasharray="8 6" />
          <line x1={mapX(pA.x)} y1={mapY(pA.y)} x2={mapX(pB.x)} y2={mapY(pB.y)} stroke="#2563eb" strokeWidth="3" />

          {[sigma1, sigma2, sigma3].map((s, i) => (
            <g key={`principal-${i}`}>
              <circle cx={mapX(s)} cy={mapY(0)} r="4.5" fill="#0f172a" />
              <text x={mapX(s) + (i === 0 ? 18 : i === 1 ? 18 : -18)} y={mapY(0) - 12} textAnchor="middle" fontSize="12" fill="#0f172a">
                {`σ${i + 1}`}
              </text>
            </g>
          ))}

          {[pA, pB].map((p) => {
            const vx = p.x - outerCenter;
            const vy = p.y;
            const mag = Math.hypot(vx, vy) || 1;
            const ux = vx / mag;
            const uy = vy / mag;
            const tangentX = -uy;
            const tangentY = ux;
            const radialPx = -20;
            const tangentSign = p.x >= outerCenter ? 1 : -1;
            const tangentPx = tangentSign * 14;
            const tx = mapX(p.x) + radialPx * ux + tangentPx * tangentX;
            const ty = mapY(p.y) - radialPx * uy - tangentPx * tangentY;

            return (
              <g key={p.label}>
                <circle cx={mapX(p.x)} cy={mapY(p.y)} r="5" fill="#2563eb" />
                <text x={tx} y={ty} fontSize="13" fill="#1d4ed8">
                  {p.label}
                </text>
              </g>
            );
          })}
        </>
      )}
    </AxisFrame>
  );
}

function TwoDStressElementSVG({ plane, axisNames, originalState, rotatedState, thetaDeg }) {
  const width = 1040;
  const height = 700;

  const renderPanel = (panelCx, panelCy, thetaLocalDeg, state, names, title, axisColor, useDynamicLengths = false) => {
    const half = 118;
    const theta = thetaLocalDeg * DEG;

    const baseAxes =
      plane === "xy"
        ? { e1: [1, 0], e2: [0, -1] }
        : plane === "xz"
          ? { e1: [1, 0], e2: [0, 1] }
          : { e1: [0, -1], e2: [-1, 0] };

    const ux = [
      Math.cos(theta) * baseAxes.e1[0] - Math.sin(theta) * baseAxes.e2[0],
      Math.cos(theta) * baseAxes.e1[1] - Math.sin(theta) * baseAxes.e2[1],
    ];
    const uy = [
      Math.sin(theta) * baseAxes.e1[0] + Math.cos(theta) * baseAxes.e2[0],
      Math.sin(theta) * baseAxes.e1[1] + Math.cos(theta) * baseAxes.e2[1],
    ];
    const map = (lx, ly) => ({ x: panelCx + lx * ux[0] + ly * uy[0], y: panelCy + lx * ux[1] + ly * uy[1] });
    const corners = [map(-half, -half), map(half, -half), map(half, half), map(-half, half)];

    const innerAxisLen = 64;
    const faceGap = 28;
    const shearGap = 16;
    const tauSign = state.txpyp >= 0 ? 1 : -1;
    const tol = 1e-9;

    const addVec = (p, v, d) => ({ x: p.x + v[0] * d, y: p.y + v[1] * d });

    const maxMag = Math.max(Math.abs(state.sxp), Math.abs(state.syp), Math.abs(state.txpyp), 1);
    const fixedLen = 56;
    const dynamicLen = (value, minLen = 24, maxLen = 64) => {
      if (Math.abs(value) <= tol) return 0;
      return minLen + (maxLen - minLen) * (Math.abs(value) / maxMag);
    };

    const nxLen = useDynamicLengths ? dynamicLen(state.sxp, 26, 66) : (Math.abs(state.sxp) > tol ? fixedLen : 0);
    const nyLen = useDynamicLengths ? dynamicLen(state.syp, 26, 66) : (Math.abs(state.syp) > tol ? fixedLen : 0);
    const tLen = useDynamicLengths ? dynamicLen(state.txpyp, 20, 58) : (Math.abs(state.txpyp) > tol ? fixedLen : 0);

    const arrow = (start, dir, len, stroke, marker, strokeWidth = 3) => {
      if (len <= tol) return null;
      const end = { x: start.x + dir[0] * len, y: start.y + dir[1] * len };
      return <line x1={start.x} y1={start.y} x2={end.x} y2={end.y} stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" markerEnd={`url(#${marker})`} />;
    };

    const sxPos = state.sxp >= 0;
    const syPos = state.syp >= 0;

    const rightFace = addVec(map(half, 0), ux, faceGap);
    const leftFace = addVec(map(-half, 0), [-ux[0], -ux[1]], faceGap);
    const topFace = addVec(map(0, half), uy, faceGap);
    const botFace = addVec(map(0, -half), [-uy[0], -uy[1]], faceGap);

    const rightShear = addVec(map(half, 0), ux, shearGap);
    const leftShear = addVec(map(-half, 0), [-ux[0], -ux[1]], shearGap);
    const topShear = addVec(map(0, half), uy, shearGap);
    const botShear = addVec(map(0, -half), [-uy[0], -uy[1]], shearGap);

    const axisXEnd = addVec({ x: panelCx, y: panelCy }, ux, innerAxisLen);
    const axisYEnd = addVec({ x: panelCx, y: panelCy }, uy, innerAxisLen);

    return (
      <g>
        <text x={panelCx} y={76} textAnchor="middle" fontSize="18" fill="#334155" fontWeight="600">{title}</text>

        <polygon points={corners.map((p) => `${p.x},${p.y}`).join(" ")} fill="#ffffff" stroke="#374151" strokeWidth="3.5" />
        <circle cx={panelCx} cy={panelCy} r="5" fill="#e879f9" opacity="0.55" />

        <line x1={panelCx} y1={panelCy} x2={axisXEnd.x} y2={axisXEnd.y} stroke="#0f172a" strokeWidth="3" markerEnd="url(#axis-head)" />
        <line x1={panelCx} y1={panelCy} x2={axisYEnd.x} y2={axisYEnd.y} stroke="#0f172a" strokeWidth="3" markerEnd="url(#axis-head)" />
        <text x={axisXEnd.x + 10 * ux[0]} y={axisXEnd.y + 10 * ux[1]} fontSize="18" fill={axisColor}>{names[0]}</text>
        <text x={axisYEnd.x + 10 * uy[0]} y={axisYEnd.y + 10 * uy[1]} fontSize="18" fill={axisColor}>{names[1]}</text>

        {sxPos ? arrow(rightFace, ux, nxLen, "#0f172a", "normal-head") : arrow(addVec(rightFace, ux, nxLen), [-ux[0], -ux[1]], nxLen, "#0f172a", "normal-head")}
        {sxPos ? arrow(leftFace, [-ux[0], -ux[1]], nxLen, "#0f172a", "normal-head") : arrow(addVec(leftFace, [-ux[0], -ux[1]], nxLen), ux, nxLen, "#0f172a", "normal-head")}
        {syPos ? arrow(topFace, uy, nyLen, "#0f172a", "normal-head") : arrow(addVec(topFace, uy, nyLen), [-uy[0], -uy[1]], nyLen, "#0f172a", "normal-head")}
        {syPos ? arrow(botFace, [-uy[0], -uy[1]], nyLen, "#0f172a", "normal-head") : arrow(addVec(botFace, [-uy[0], -uy[1]], nyLen), uy, nyLen, "#0f172a", "normal-head")}

        {tauSign >= 0 ? arrow(topShear, ux, tLen, "#6b7280", "shear-head", 2.8) : arrow(addVec(topShear, ux, tLen), [-ux[0], -ux[1]], tLen, "#6b7280", "shear-head", 2.8)}
        {tauSign >= 0 ? arrow(rightShear, uy, tLen, "#6b7280", "shear-head", 2.8) : arrow(addVec(rightShear, uy, tLen), [-uy[0], -uy[1]], tLen, "#6b7280", "shear-head", 2.8)}
        {tauSign >= 0 ? arrow(botShear, [-ux[0], -ux[1]], tLen, "#6b7280", "shear-head", 2.8) : arrow(addVec(botShear, [-ux[0], -ux[1]], tLen), ux, tLen, "#6b7280", "shear-head", 2.8)}
        {tauSign >= 0 ? arrow(leftShear, [-uy[0], -uy[1]], tLen, "#6b7280", "shear-head", 2.8) : arrow(addVec(leftShear, [-uy[0], -uy[1]], tLen), uy, tLen, "#6b7280", "shear-head", 2.8)}
      </g>
    );
  };

  return (
    <div className="rounded-2xl border bg-white p-3 shadow-sm">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full">
        <defs>
          <marker id="normal-head" markerWidth="4" markerHeight="4" refX="3.2" refY="2" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,4 L3.4,2 z" fill="#0f172a" />
          </marker>
          <marker id="shear-head" markerWidth="4" markerHeight="4" refX="3.2" refY="2" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,4 L3.4,2 z" fill="#6b7280" />
          </marker>
          <marker id="axis-head" markerWidth="4" markerHeight="4" refX="3.2" refY="2" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,4 L3.4,2 z" fill="#0f172a" />
          </marker>
        </defs>
        <rect x="0" y="0" width={width} height={height} rx="20" fill="#ffffff" />
        {renderPanel(270, 360, 0, originalState, axisNames, "Original element preview", "#0f172a", false)}
        {renderPanel(770, 360, thetaDeg, rotatedState, [`${axisNames[0]}'`, `${axisNames[1]}'`], "Rotated element", "#2563eb", true)}
      </svg>
    </div>
  );
}

function ThreeDStressElementSVG({ basisReference, basisRotated, tensorReference, tensorRotated, thetaDeg, viewAzim, viewElev }) {
  const width = 1040;
  const height = 760;

  const projectWorld = (v, cx, cy, drawScale) => {
    const a = viewAzim * DEG;
    const e = viewElev * DEG;

    // Standard view target:
    // local axis 0 -> y up on screen
    // local axis 1 -> x coming out of the page
    // local axis 2 -> z to the left
    const out = v[1];
    const up = v[0];
    const left = v[2];

    const outYaw = Math.cos(a) * out - Math.sin(a) * left;
    const leftYaw = Math.sin(a) * out + Math.cos(a) * left;
    const upYaw = up;

    const upPitch = Math.cos(e) * upYaw - Math.sin(e) * leftYaw;
    const leftPitch = Math.sin(e) * upYaw + Math.cos(e) * leftYaw;

    return {
      x: cx + drawScale * (0.85 * outYaw - 0.95 * leftPitch),
      y: cy + drawScale * (-1.0 * upPitch + 0.32 * outYaw),
    };
  };

  const renderPanel = (panelCx, panelCy, basis, tensorLocal, title, axisNames, useDynamicLengths = false) => {
    const cubeHalf = 1;
    const drawScale = 90;
    const tol = 1e-9;

    const worldFromLocal = (lx, ly, lz) => add(add(scale(basis[0], lx), scale(basis[1], ly)), scale(basis[2], lz));
    const projectLocal = (lx, ly, lz) => projectWorld(worldFromLocal(lx, ly, lz), panelCx, panelCy, drawScale);

    const verts = [
      projectLocal(-cubeHalf, -cubeHalf, -cubeHalf),
      projectLocal(cubeHalf, -cubeHalf, -cubeHalf),
      projectLocal(cubeHalf, cubeHalf, -cubeHalf),
      projectLocal(-cubeHalf, cubeHalf, -cubeHalf),
      projectLocal(-cubeHalf, -cubeHalf, cubeHalf),
      projectLocal(cubeHalf, -cubeHalf, cubeHalf),
      projectLocal(cubeHalf, cubeHalf, cubeHalf),
      projectLocal(-cubeHalf, cubeHalf, cubeHalf),
    ];

    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];

    const axisStroke = "#dc2626";
    const displayAxisNames = axisNames;
    const axisXEnd = projectLocal(2.35, 0, 0);
    const axisYEnd = projectLocal(0, 2.35, 0);
    const axisZEnd = projectLocal(0, 0, 2.35);

    const maxMag = tensorMaxAbs(tensorLocal);
    const dynamicLen = (value, minLen, maxLen) => {
      if (Math.abs(value) <= tol) return 0;
      return minLen + (maxLen - minLen) * (Math.abs(value) / maxMag);
    };
    const getNormalLen = (value) => (useDynamicLengths ? dynamicLen(value, 0.24, 0.82) : (Math.abs(value) > tol ? 0.62 : 0));
    const getShearLen = (value) => (useDynamicLengths ? dynamicLen(value, 0.18, 0.68) : (Math.abs(value) > tol ? 0.54 : 0));

    const drawArrow = (startLocal, dirLocal, len, stroke, marker, strokeWidth = 3.2) => {
      if (len <= tol || (dirLocal[0] === 0 && dirLocal[1] === 0 && dirLocal[2] === 0)) return null;
      const start = projectLocal(startLocal[0], startLocal[1], startLocal[2]);
      const end = projectLocal(
        startLocal[0] + dirLocal[0] * len,
        startLocal[1] + dirLocal[1] * len,
        startLocal[2] + dirLocal[2] * len
      );
      return (
        <line
          x1={start.x}
          y1={start.y}
          x2={end.x}
          y2={end.y}
          stroke={stroke}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          markerEnd={`url(#${marker})`}
        />
      );
    };

    const drawCenteredArrow = (centerLocal, dirLocal, len, stroke, marker, strokeWidth = 2.8) => {
      if (len <= tol || (dirLocal[0] === 0 && dirLocal[1] === 0 && dirLocal[2] === 0)) return null;
      const halfLen = len / 2;
      const startLocal = [
        centerLocal[0] - dirLocal[0] * halfLen,
        centerLocal[1] - dirLocal[1] * halfLen,
        centerLocal[2] - dirLocal[2] * halfLen,
      ];
      return drawArrow(startLocal, dirLocal, len, stroke, marker, strokeWidth);
    };

    const faceConfigs = [
      { axis: 0, sign: 1, tangents: [1, 2] },
      { axis: 0, sign: -1, tangents: [1, 2] },
      { axis: 1, sign: 1, tangents: [0, 2] },
      { axis: 1, sign: -1, tangents: [0, 2] },
      { axis: 2, sign: 1, tangents: [0, 1] },
      { axis: 2, sign: -1, tangents: [0, 1] },
    ];

    return (
      <g>
        <text x={panelCx} y={68} textAnchor="middle" fontSize="18" fill="#334155" fontWeight="600">{title}</text>
        {edges.map(([i, j], idx) => (
          <line key={idx} x1={verts[i].x} y1={verts[i].y} x2={verts[j].x} y2={verts[j].y} stroke="#0f172a" strokeWidth="3.4" />
        ))}
        <circle cx={panelCx} cy={panelCy} r="5" fill="#e879f9" opacity="0.55" />

        <line x1={panelCx} y1={panelCy} x2={axisXEnd.x} y2={axisXEnd.y} stroke={axisStroke} strokeWidth="3.4" markerEnd="url(#axis3-head)" />
        <line x1={panelCx} y1={panelCy} x2={axisYEnd.x} y2={axisYEnd.y} stroke={axisStroke} strokeWidth="3.4" markerEnd="url(#axis3-head)" />
        <line x1={panelCx} y1={panelCy} x2={axisZEnd.x} y2={axisZEnd.y} stroke={axisStroke} strokeWidth="3.4" markerEnd="url(#axis3-head)" />
        <text x={axisXEnd.x + 18} y={axisXEnd.y + 4} fontSize="20" fill={axisStroke}>{displayAxisNames[0]}</text>
        <text x={axisYEnd.x + 18} y={axisYEnd.y + 4} fontSize="20" fill={axisStroke}>{displayAxisNames[1]}</text>
        <text x={axisZEnd.x - 26} y={axisZEnd.y + 2} fontSize="20" fill={axisStroke}>{displayAxisNames[2]}</text>

        {faceConfigs.map((face, idx) => {
          const faceCenter = [0, 0, 0];
          faceCenter[face.axis] = face.sign * 1.02;
          const faceAnchor = [0, 0, 0];
          faceAnchor[face.axis] = face.sign * 1.18;
          const faceNormal = [0, 0, 0];
          faceNormal[face.axis] = face.sign;

          const traction = [
            face.sign * tensorLocal[0][face.axis],
            face.sign * tensorLocal[1][face.axis],
            face.sign * tensorLocal[2][face.axis],
          ];

          const sigmaNormal = tensorLocal[face.axis][face.axis];
          const normalLen = getNormalLen(sigmaNormal);
          const normalDir = sigmaNormal >= 0
            ? faceNormal
            : [-faceNormal[0], -faceNormal[1], -faceNormal[2]];
          const normalStart = sigmaNormal >= 0
            ? faceAnchor
            : [
                faceAnchor[0] + faceNormal[0] * normalLen,
                faceAnchor[1] + faceNormal[1] * normalLen,
                faceAnchor[2] + faceNormal[2] * normalLen,
              ];

          return (
            <g key={`face-${idx}`}>
              {drawArrow(normalStart, normalDir, normalLen, "#0f172a", "normal3-head")}
              {face.tangents.map((tangentAxis) => {
                const shearValue = traction[tangentAxis];
                const shearDir = [0, 0, 0];
                shearDir[tangentAxis] = Math.sign(shearValue) || 0;
                return (
                  <g key={`shear-${idx}-${tangentAxis}`}>
                    {drawCenteredArrow(faceCenter, shearDir, getShearLen(shearValue), "#64748b", "shear3-head", 2.8)}
                  </g>
                );
              })}
            </g>
          );
        })}
      </g>
    );
  };

  return (
    <div className="rounded-2xl border bg-white p-3 shadow-sm">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full">
        <defs>
          <marker id="normal3-head" markerWidth="4" markerHeight="4" refX="3.2" refY="2" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,4 L3.4,2 z" fill="#0f172a" />
          </marker>
          <marker id="shear3-head" markerWidth="4" markerHeight="4" refX="3.2" refY="2" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,4 L3.4,2 z" fill="#64748b" />
          </marker>
          <marker id="axis3-head" markerWidth="4" markerHeight="4" refX="3.2" refY="2" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,4 L3.4,2 z" fill="#dc2626" />
          </marker>
        </defs>
        <rect x="0" y="0" width={width} height={height} rx="20" fill="#ffffff" />
        {renderPanel(280, 420, basisReference, tensorReference, "Original 3D stress cube", ["y", "x", "z"], false)}
        {renderPanel(
          760,
          420,
          basisRotated,
          tensorRotated,
          `Rotated 3D stress cube (${formatSignedAngle(thetaDeg)})`,
          ["y'", "x'", "z'"],
          wrap180(Math.abs(thetaDeg)) > 5
        )}
      </svg>
    </div>
  );
}

export default function MohrCirclePlotter3D() {
  const [stress, setStress] = useState({
    sx: 140,
    sy: 60,
    sz: -20,
    txy: 35,
    tyz: 20,
    txz: -15,
  });
  const [theta2D, setTheta2D] = useState(0);
  const [theta3D, setTheta3D] = useState(0);
  const [rotationSense2D, setRotationSense2D] = useState("ccw");
  const [rotationSense3D, setRotationSense3D] = useState("ccw");
  const [viewAzim, setViewAzim] = useState(0);
  const [viewElev, setViewElev] = useState(0);

  const displayStress = useMemo(
    () => ({
      sx: stress.sy,
      sy: stress.sx,
      sz: stress.sz,
      txy: stress.txy,
      txz: stress.tyz,
      tyz: stress.txz,
    }),
    [stress]
  );

  const tensor = useMemo(
    () => [
      [displayStress.sx, displayStress.txy, displayStress.txz],
      [displayStress.txy, displayStress.sy, displayStress.tyz],
      [displayStress.txz, displayStress.tyz, displayStress.sz],
    ],
    [displayStress]
  );

  const mode = useMemo(() => detectMode(displayStress), [displayStress]);
  const principal3D = useMemo(() => eigenvaluesSymmetric3x3(tensor), [tensor]);
  const eigenbasis = useMemo(() => eigenbasisSymmetric3x3(tensor, principal3D), [tensor, principal3D]);

  const plane2D = useMemo(() => {
    if (mode.type !== "2d") return null;
    return principal2D(mode.a, mode.b, mode.tau);
  }, [mode]);

  const angles2D = useMemo(() => {
    if (mode.type !== "2d") return null;
    return principalAngles2D(mode.a, mode.b, mode.tau);
  }, [mode]);

  const signedTheta2D = rotationSense2D === "cw" ? theta2D : -theta2D;
  const signedTheta3D = rotationSense3D === "cw" ? -theta3D : theta3D;

  const original2D = useMemo(() => {
    if (mode.type !== "2d") return null;
    return { sxp: mode.a, syp: mode.b, txpyp: mode.tau };
  }, [mode]);

  const mohrTauSign2D = useMemo(() => {
    if (mode.type !== "2d") return 1;
    return mohrShearSignForPlane(mode.plane);
  }, [mode]);

  const current2D = useMemo(() => {
    if (mode.type !== "2d") return null;
    return transform2D(mode.a, mode.b, mode.tau, signedTheta2D * DEG);
  }, [mode, signedTheta2D]);

  const reference3D = useMemo(() => transformPrincipal13(principal3D[0], principal3D[1], principal3D[2], 0), [principal3D]);
  const current3D = useMemo(() => transformPrincipal13(principal3D[0], principal3D[1], principal3D[2], signedTheta3D), [principal3D, signedTheta3D]);
  const basisReference3D = useMemo(() => [[1, 0, 0], [0, 1, 0], [0, 0, 1]], []);
  const currentBasis3D = useMemo(() => rotateBasis13(basisReference3D, signedTheta3D), [basisReference3D, signedTheta3D]);
  const referenceTensor3D = useMemo(() => tensor, [tensor]);
  const currentTensor3D = useMemo(() => tensorInBasis(tensor, currentBasis3D), [tensor, currentBasis3D]);

  const circles3D = useMemo(() => {
    const [sigma1, sigma2, sigma3] = principal3D;
    return [
      { name: "σ1–σ2", center: (sigma1 + sigma2) / 2, radius: Math.abs(sigma1 - sigma2) / 2 },
      { name: "σ2–σ3", center: (sigma2 + sigma3) / 2, radius: Math.abs(sigma2 - sigma3) / 2 },
      { name: "σ1–σ3", center: (sigma1 + sigma3) / 2, radius: Math.abs(sigma1 - sigma3) / 2 },
    ];
  }, [principal3D]);

  const setField = (key, value) => {
    setStress((prev) => ({ ...prev, [key]: Number.isFinite(value) ? value : 0 }));
  };

  const buttons2D = angles2D
    ? [
        { label: "Max principal", value: angles2D.maxPrincipal },
        { label: "Min principal", value: angles2D.minPrincipal },
        { label: "+ Max shear", value: angles2D.maxShearPositive },
        { label: "- Max shear", value: angles2D.maxShearNegative },
      ]
    : [];

  const buttons3D = [
    { label: "σ1 direction", value: 0 },
    { label: "σ3 direction", value: 90 },
    { label: "+ τmax", value: 45 },
    { label: "- τmax", value: 135 },
  ];

  return (
    <div className="min-h-screen bg-slate-50 p-6 text-slate-900">
      <div className="mx-auto grid max-w-[1800px] gap-6 lg:grid-cols-[320px_1fr]">
        <div className="space-y-6">
          <Card className="rounded-3xl shadow-sm">
            <CardHeader>
              <CardTitle className="text-2xl">Adaptive 2D / 3D Mohr Circle Plotter</CardTitle>
              <p className="text-sm text-slate-600">
                In 2D mode the app shows the original and rotated stress elements side by side. In 3D mode it shows a reference coordinate system and the rotated one, while the Mohr circles remain linked to the active rotation.
              </p>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="grid grid-cols-2 gap-4">
                <Field label="σx" value={stress.sy} onChange={(v) => setField("sy", v)} />
                <Field label="σy" value={stress.sx} onChange={(v) => setField("sx", v)} />
                <Field label="σz" value={stress.sz} onChange={(v) => setField("sz", v)} />
                <Field label="τxy" value={stress.txy} onChange={(v) => setField("txy", v)} />
                <Field label="τyz" value={stress.txz} onChange={(v) => setField("txz", v)} />
                <Field label="τxz" value={stress.tyz} onChange={(v) => setField("tyz", v)} />
              </div>

              <div>
                <div className="mb-2 text-sm font-medium text-slate-700">Examples</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(examples).map(([label, vals]) => (
                    <Button key={label} variant="outline" className="rounded-2xl" onClick={() => setStress(vals)}>
                      {label}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border bg-white p-4">
                <div className="text-sm font-medium text-slate-700">Active mode</div>
                <div className="mt-2 text-lg font-semibold text-slate-900">
                  {mode.type === "2d" ? `2D Mohr circle in the ${mode.plane.toUpperCase()} plane` : "Full 3D Mohr circles"}
                </div>

              </div>

              {mode.type === "2d" && angles2D ? (
                <AngleSlider
                  label={`Rotation angle in the ${mode.plane.toUpperCase()} plane`}
                  value={theta2D}
                  onChange={setTheta2D}
                  buttons={buttons2D}
                  rotationSense={rotationSense2D}
                  onRotationSenseChange={setRotationSense2D}
                />
              ) : (
                <AngleSlider
                  label="Rotation in the principal 1–3 plane"
                  value={theta3D}
                  onChange={setTheta3D}
                  buttons={buttons3D}
                  rotationSense={rotationSense3D}
                  onRotationSenseChange={setRotationSense3D}
                />
              )}

              {mode.type !== "2d" ? <ViewControls azim={viewAzim} elev={viewElev} setAzim={setViewAzim} setElev={setViewElev} /> : null}

              <div>
                <div className="mb-2 text-sm font-medium text-slate-700">Stress tensor</div>
                <TensorTable m={tensor} />
                <p className="mt-2 text-xs text-slate-500">Since this is a Cauchy stress tensor, it is symmetric by definition.</p>
              </div>
            </CardContent>
          </Card>

          {mode.type !== "2d" ? (
            <div className="rounded-2xl border bg-white p-4 shadow-sm">
              <div className="text-sm text-slate-500">Principal directions</div>
              <div className="mt-3 grid gap-3">
                <div className="rounded-xl border p-3">
                  <div className="text-sm text-slate-500">σ1 direction</div>
                  <div className="font-mono text-sm">{formatVector(eigenbasis[0])}</div>
                </div>
                <div className="rounded-xl border p-3">
                  <div className="text-sm text-slate-500">σ2 direction</div>
                  <div className="font-mono text-sm">{formatVector(eigenbasis[1])}</div>
                </div>
                <div className="rounded-xl border p-3">
                  <div className="text-sm text-slate-500">σ3 direction</div>
                  <div className="font-mono text-sm">{formatVector(eigenbasis[2])}</div>
                </div>
                <div className="text-xs text-slate-500">The Mohr circle view only shows the σ1–σ3 principal-plane exploration.</div>
              </div>
            </div>
          ) : null}
        </div>

        <div className="space-y-6">
          <Card className="rounded-3xl shadow-sm">
            <CardHeader>
              <CardTitle className="text-xl">Mohr circle and stress element explorer</CardTitle>
              <p className="text-sm text-slate-600">
                The dashed line is the reference diameter. The blue line is the currently rotated diameter and it stays synchronized with the stress element.
              </p>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6">
                <div>
                  {mode.type === "2d" && plane2D ? (
                    <TwoDMohrSVG plane={mode.plane} labels={mode.labels} a={mode.a} b={mode.b} tau={mode.tau} thetaDeg={signedTheta2D} />
                  ) : (
                    <ThreeDMohrSVG sigma1={principal3D[0]} sigma2={principal3D[1]} sigma3={principal3D[2]} thetaDeg={signedTheta3D} />
                  )}
                </div>
                <div>
                  {mode.type === "2d" && current2D && original2D ? (
                    <TwoDStressElementSVG plane={mode.plane} axisNames={mode.axisNames} originalState={original2D} rotatedState={current2D} thetaDeg={signedTheta2D} />
                  ) : (
                    <ThreeDStressElementSVG
                      basisReference={basisReference3D}
                      basisRotated={currentBasis3D}
                      tensorReference={referenceTensor3D}
                      tensorRotated={currentTensor3D}
                      thetaDeg={signedTheta3D}
                      viewAzim={viewAzim}
                      viewElev={viewElev}
                    />
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-3xl shadow-sm">
            <CardHeader>
              <CardTitle className="text-xl">Circle data</CardTitle>
            </CardHeader>
            <CardContent>
              {mode.type === "2d" && plane2D && current2D ? (
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
                  <div className="rounded-2xl border bg-white p-4">
                    <div className="text-sm text-slate-500">Original positive face point {mode.labels[0]}</div>
                    <div className="mt-2 space-y-1 text-sm">
                      <div><span className="font-medium">σ:</span> <span className="font-mono">{format(mode.a)}</span></div>
                      <div><span className="font-medium">τ:</span> <span className="font-mono">{format(mohrTauSign2D * mode.tau)}</span></div>
                    </div>
                  </div>
                  <div className="rounded-2xl border bg-white p-4">
                    <div className="text-sm text-slate-500">Original positive face point {mode.labels[1]}</div>
                    <div className="mt-2 space-y-1 text-sm">
                      <div><span className="font-medium">σ:</span> <span className="font-mono">{format(mode.b)}</span></div>
                      <div><span className="font-medium">τ:</span> <span className="font-mono">{format(-mohrTauSign2D * mode.tau)}</span></div>
                    </div>
                  </div>
                  <div className="rounded-2xl border bg-white p-4">
                    <div className="text-sm text-slate-500">Current rotated point {mode.labels[0]}'</div>
                    <div className="mt-2 space-y-1 text-sm">
                      <div><span className="font-medium">σ:</span> <span className="font-mono">{format(current2D.sxp)}</span></div>
                      <div><span className="font-medium">τ:</span> <span className="font-mono">{format(mohrTauSign2D * current2D.txpyp)}</span></div>
                    </div>
                  </div>
                  <div className="rounded-2xl border bg-white p-4">
                    <div className="text-sm text-slate-500">Current rotated point {mode.labels[1]}'</div>
                    <div className="mt-2 space-y-1 text-sm">
                      <div><span className="font-medium">σ:</span> <span className="font-mono">{format(current2D.syp)}</span></div>
                      <div><span className="font-medium">τ:</span> <span className="font-mono">{format(-mohrTauSign2D * current2D.txpyp)}</span></div>
                    </div>
                  </div>
                  <div className="rounded-2xl border bg-white p-4">
                    <div className="text-sm text-slate-500">Circle properties</div>
                    <div className="mt-2 space-y-1 text-sm">
                      <div><span className="font-medium">Center:</span> <span className="font-mono">{format(plane2D.center)}</span></div>
                      <div><span className="font-medium">Radius:</span> <span className="font-mono">{format(plane2D.radius)}</span></div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="grid gap-4 md:grid-cols-3">
                  {circles3D.map((circle) => (
                    <div key={circle.name} className="rounded-2xl border bg-white p-4">
                      <div className="text-sm text-slate-500">{circle.name}</div>
                      <div className="mt-2 space-y-1 text-sm">
                        <div><span className="font-medium">Center:</span> <span className="font-mono">{format(circle.center)}</span></div>
                        <div><span className="font-medium">Radius:</span> <span className="font-mono">{format(circle.radius)}</span></div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="rounded-3xl shadow-sm">
            <CardHeader>
              <CardTitle className="text-xl">Stress results</CardTitle>
            </CardHeader>
            <CardContent>
              {mode.type === "2d" && plane2D && current2D && angles2D ? (
                <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                  <StatCard title="σ1" value={format(plane2D.sigma1)} subtitle={`Rotate to ${formatAngle(angles2D.maxPrincipal)}`} />
                  <StatCard title="σ2" value={format(plane2D.sigma2)} subtitle={`Rotate to ${formatAngle(angles2D.minPrincipal)}`} />
                  <StatCard title="τmax" value={format(plane2D.tauMax)} subtitle={`At ${formatAngle(angles2D.maxShearPositive)} and ${formatAngle(angles2D.maxShearNegative)}`} />
                  <StatCard title={`Current σ${mode.axisNames[0]}'`} value={format(current2D.sxp)} subtitle={`at θ = ${formatSignedAngle(signedTheta2D)}`} />
                  <StatCard title={`Current σ${mode.axisNames[1]}'`} value={format(current2D.syp)} subtitle={`at θ = ${formatSignedAngle(signedTheta2D)}`} />
                  <StatCard title={`Current τ${mode.axisNames[0]}'${mode.axisNames[1]}'`} value={format(current2D.txpyp)} subtitle="Moving blue diameter" />
                </div>
              ) : (
                <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                  <StatCard title="σ1" value={format(principal3D[0])} subtitle="Maximum principal stress" />
                  <StatCard title="σ2" value={format(principal3D[1])} subtitle="Intermediate principal stress" />
                  <StatCard title="σ3" value={format(principal3D[2])} subtitle="Minimum principal stress" />
                  <StatCard title="τmax" value={format(Math.abs(principal3D[0] - principal3D[2]) / 2)} subtitle="Largest circle: (σ1 − σ3) / 2" />
                  <StatCard title="Current σx'" value={format(current3D.sxp)} subtitle={`θ₁₃ = ${formatSignedAngle(signedTheta3D)}`} />
                  <StatCard title="Current τx'z'" value={format(current3D.txpzp)} subtitle="Moving blue diameter on σ1–σ3 circle" />
                </div>
              )}
            </CardContent>
          </Card>

          {mode.type === "2d" && angles2D ? (
            <Card className="rounded-3xl shadow-sm">
              <CardHeader>
                <CardTitle className="text-xl">Principal and shear angles</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="rounded-xl border bg-white p-3">
                    <div className="text-sm text-slate-500">Max principal</div>
                    <div className="font-semibold">{formatAngle(angles2D.maxPrincipal)}</div>
                  </div>
                  <div className="rounded-xl border bg-white p-3">
                    <div className="text-sm text-slate-500">Min principal</div>
                    <div className="font-semibold">{formatAngle(angles2D.minPrincipal)}</div>
                  </div>
                  <div className="rounded-xl border bg-white p-3">
                    <div className="text-sm text-slate-500">+ Max shear</div>
                    <div className="font-semibold">{formatAngle(angles2D.maxShearPositive)}</div>
                  </div>
                  <div className="rounded-xl border bg-white p-3">
                    <div className="text-sm text-slate-500">- Max shear</div>
                    <div className="font-semibold">{formatAngle(angles2D.maxShearNegative)}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : null}
        </div>
      </div>
    </div>
  );
}
