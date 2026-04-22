/**
 * Stress-test scale helpers matching index.html accChartPadRange + plot mapping.
 * Run: node stt_cnn_lstm/tools/stress_accuracy_chart_bounds.js
 */
function accChartPadRange(minV, maxV, padRatio) {
  if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return [0, 1];
  if (minV === maxV) {
    const d = Math.max(Math.abs(minV) * 0.12, 0.5);
    return [minV - d, maxV + d];
  }
  const span = (maxV - minV) * padRatio;
  return [minV - span, maxV + span];
}

const padL = 46,
  padR = 14,
  pw = 400;

for (let k = 0; k < 25_000; k++) {
  const n = 2 + Math.floor(Math.random() * 120);
  const dbs = [];
  const accs = [];
  for (let i = 0; i < n; i++) {
    dbs.push(-75 + Math.random() * 55);
    accs.push(Math.random());
  }
  const minDb = Math.min(...dbs),
    maxDb = Math.max(...dbs);
  const minA = Math.min(...accs),
    maxA = Math.max(...accs);
  const xr = accChartPadRange(minDb, maxDb, 0.1);
  const yr = accChartPadRange(minA, maxA, 0.1);
  if (!(xr[0] < xr[1] && yr[0] < yr[1])) {
    throw new Error(`bad range k=${k} xr=${xr} yr=${yr}`);
  }
  for (let j = 0; j < n; j++) {
    const nx = (dbs[j] - xr[0]) / (xr[1] - xr[0]);
    const ny = (accs[j] - yr[0]) / (yr[1] - yr[0]);
    if (nx < -1e-6 || nx > 1 + 1e-6 || ny < -1e-6 || ny > 1 + 1e-6) {
      throw new Error(`map k=${k} j=${j} nx=${nx} ny=${ny}`);
    }
    const xPix = padL + nx * pw;
    if (xPix < padL - 1 || xPix > padL + pw + 1) {
      throw new Error(`pixel k=${k}`);
    }
  }
}

console.log("stress_accuracy_chart_bounds ok (25000 batches)");
