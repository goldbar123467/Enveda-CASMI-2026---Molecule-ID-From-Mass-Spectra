"use strict";
const $ = id => document.getElementById(id);
const set = (id, value) => { $(id).textContent = value; };
const num = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const title = value => value ? value.charAt(0).toUpperCase() + value.slice(1).replaceAll('_', ' ') : '—';
const modelLabel = model => model.label || `${title(model.family)} · fold ${model.fold}`;
const clock = value => value ? new Date(value).toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'}) : '—';
const duration = seconds => {
  if (!Number.isFinite(seconds)) return '—';
  seconds = Math.max(0, seconds);
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor(seconds % 3600 / 60)}m`;
};
const percent = (id, value) => { $(id).style.width = `${Math.min(100, Math.max(0, value || 0))}%`; };
let current = null, fetching = false, lastSuccess = 0;

function refreshTimes() {
  if (!current) return;
  const age = current.updated_utc ? (Date.now() - Date.parse(current.updated_utc)) / 1000 : null;
  set('heartbeat', age === null ? '—' : `${duration(age)} ago`);
  const eta = current.progress.eta_utc;
  set('remaining', eta ? `${duration((Date.parse(eta) - Date.now()) / 1000)} remaining` : current.state === 'complete' ? 'Training finished' : 'ETA unavailable');
  const watchdog = current.watchdog;
  const next = watchdog.next_check_utc;
  const terminal = ['complete', 'paused', 'budget_reached', 'needs_attention', 'attention_missing_status'].includes(watchdog.action);
  set('watchdog-next', next && !terminal ? `${clock(next)} · ${Date.parse(next) < Date.now() ? 'due' : 'in ' + duration((Date.parse(next) - Date.now()) / 1000)}` : '—');
  if (lastSuccess && Date.now() - lastSuccess > 15000) {
    set('connection', 'Connection lost');
    $('connection').classList.add('offline');
    $('notice').hidden = false;
    set('notice', 'The dashboard connection is interrupted. Values below are from the last successful refresh.');
  }
}

function render(data) {
  current = data;
  const p = data.progress, a = data.active, gpu = data.gpu || {}, ram = data.memory || {};
  const frozen = data.kind === 'frozen_retrieval', unit = p.unit || 'epochs';
  const runLive = ['training', 'finalizing'].includes(data.state);
  set('run-id', data.run_id);
  set('run-description', data.description || `${p.total_models} models · ${a.epochs} epochs each · paired control / external data`);
  set('pace-unit', `s / ${unit === 'steps' ? 'step' : 'epoch'}`);
  set('chart-description', `Training loss by ${unit === 'steps' ? 'step' : 'epoch'}`);
  set('queue-description', frozen ? 'Two seeds · learned heads on frozen encoders' : 'Paired control and external-data models · shared folds');
  set('model-index-heading', frozen ? 'SEED' : 'FOLD');
  const labels = {training: 'Training active', finalizing: 'Finalizing OOF', complete: 'Training complete', waiting: 'Waiting for run', stale: 'Heartbeat stale', stopped: 'Trainer stopped', error: 'Training error', paused: 'Project paused'};
  set('run-state', labels[data.state] || title(data.state));
  $('run-state').className = 'status-pill' + (['error', 'stopped'].includes(data.state) ? ' error-state' : ['stale', 'paused'].includes(data.state) ? ' warning-state' : '');
  set('overall', num(p.percent));
  percent('overall-bar', p.percent);
  set('epoch-total', `${p.completed_epochs.toLocaleString()} / ${p.total_epochs.toLocaleString()} ${unit}`);
  set('model-counter', `${p.completed_models} / ${p.total_models} models complete`);
  set('queue-count', `${p.completed_models} / ${p.total_models}`);
  set('active-model', data.state === 'complete' ? 'All training models complete' : frozen ? data.stage.label : a.family ? `${title(a.family)} · fold ${a.fold}` : 'Waiting for first epoch');
  set('active-epoch', frozen && data.stage.total ? `${data.stage.completed.toLocaleString()} / ${data.stage.total.toLocaleString()}` : a.epoch && data.state !== 'complete' ? `${title(unit === 'steps' ? 'step' : 'epoch')} ${a.epoch} / ${a.epochs}` : '');
  const eta = p.eta_utc ? new Date(p.eta_utc) : null;
  set('eta', eta ? clock(p.eta_utc) : data.state === 'complete' ? 'Finished' : '—');
  set('eta-date', eta ? eta.toLocaleDateString([], {weekday: 'short', month: 'short', day: 'numeric', timeZoneName: undefined}) + ' · your local time' : data.state === 'complete' ? 'Ready for validation' : frozen ? `Budget ends ${new Date(data.stage.deadline_utc).toLocaleString()}` : 'Awaiting a healthy training heartbeat');
  set('pace', num(p.seconds_per_epoch, frozen ? 3 : 1));
  set('loss', num(data.resources.loss, 4));
  $('segments').replaceChildren(...data.models.map(model => {
    const segment = document.createElement('div'), fill = document.createElement('div');
    segment.className = 'segment' + (model.active ? ' active' : '');
    segment.title = `${modelLabel(model)}: ${model.epoch}/${model.epochs} ${unit}`;
    fill.style.width = `${100 * model.epoch / model.epochs}%`;
    segment.append(fill); return segment;
  }));
  set('gpu-name', gpu.name || 'GPU telemetry unavailable');
  set('gpu-util', Number.isFinite(gpu.utilization) ? `${num(gpu.utilization, 0)}%` : '—');
  percent('gpu-bar', gpu.utilization);
  set('vram', gpu.available && Number.isFinite(gpu.used_mib) && Number.isFinite(gpu.total_mib) ? `${num(gpu.used_mib / 1024, 2)} / ${num(gpu.total_mib / 1024, 1)} GiB` : '—');
  percent('vram-bar', gpu.total_mib ? gpu.used_mib / gpu.total_mib * 100 : 0);
  set('ram', Number.isFinite(ram.free_gib) && ram.total_gib ? `${num(ram.total_gib - ram.free_gib)} / ${num(ram.total_gib)} GiB` : '—');
  percent('ram-bar', ram.total_gib ? (ram.total_gib - ram.free_gib) / ram.total_gib * 100 : 0);
  set('temperature', Number.isFinite(gpu.temperature) ? `${num(gpu.temperature, 0)} °C` : '—');
  set('ram-free', Number.isFinite(ram.free_gib) ? `${num(ram.free_gib)} GiB` : '—');
  set('torch-memory', Number.isFinite(data.resources.gpu_peak_reserved_gib) ? `${num(data.resources.gpu_peak_reserved_gib, 2)} GiB` : '—');
  $('model-rows').replaceChildren(...data.models.map(model => {
    const row = document.createElement('tr');
    row.className = model.active && runLive ? 'current-row' : '';
    const cells = Array.from({length: 5}, () => document.createElement('td'));
    cells[0].textContent = title(model.family);
    cells[1].textContent = frozen ? String(model.seed) : String(model.fold).padStart(2, '0');
    cells[1].title = frozen ? `Random seed ${model.seed}` : `Zero-based fold index ${model.fold}`;
    const progress = document.createElement('div'), meter = document.createElement('div'), fill = document.createElement('div'), text = document.createElement('span');
    progress.className = 'table-progress'; meter.className = 'mini-meter'; fill.style.width = `${100 * model.epoch / model.epochs}%`;
    text.textContent = `${model.epoch}/${model.epochs}`; meter.append(fill); progress.append(meter, text); cells[2].append(progress);
    const badge = document.createElement('span');
    const state = model.complete ? 'complete' : model.active && runLive ? 'current' : model.epoch ? 'pending' : 'queued';
    badge.className = `row-state ${state}`;
    badge.textContent = model.complete ? 'Complete' : model.active && runLive ? model.epoch === model.epochs ? 'Evaluating' : 'Training' : model.epoch ? 'Incomplete' : 'Queued';
    cells[3].append(badge);
    cells[4].textContent = num(model.metrics?.mrr25, 4);
    row.append(...cells); return row;
  }));
  const selected = $('chart-model').value;
  $('chart-model').replaceChildren(...[['current', 'Current model'], ...data.models.map(model => [model.id, modelLabel(model)])].map(([value, label]) => {
    const option = document.createElement('option'); option.value = value; option.textContent = label; return option;
  }));
  $('chart-model').value = selected || 'current';
  const watchdog = data.watchdog;
  const terminal = ['complete', 'paused', 'budget_reached', 'needs_attention', 'attention_missing_status'].includes(watchdog.action);
  const offline = watchdog.alive === false && !terminal;
  set('watchdog-state', offline ? 'Supervisor offline' : watchdog.overdue ? 'Check overdue' : title(watchdog.action || 'Not configured'));
  $('watchdog-dot').className = 'watchdog-dot ' + (offline || watchdog.overdue || ['needs_attention', 'attention_missing_status'].includes(watchdog.action) ? 'bad' : watchdog.action === 'healthy' ? 'good' : '');
  set('watchdog-last', clock(watchdog.checked_utc));
  set('watchdog-restarts', watchdog.restarts ?? '—');
  const reason = watchdog.reason || watchdog.error;
  $('watchdog-reason').hidden = !reason;
  set('watchdog-reason', reason || '');
  set('log-output', data.log.length ? data.log.join('\n') : 'No training log yet.');
  set('error-output', data.errors.join('\n'));
  $('error-details').hidden = !data.errors.length;
  let notice = '';
  if (data.state === 'complete') notice = 'Training is complete. Review the validation results and qualify the production notebook before submission.';
  else if (['stale', 'stopped', 'error', 'paused'].includes(data.state)) notice = `${labels[data.state]}. ${data.failure?.error || 'Check the watchdog status and recent logs below.'}`;
  else if (offline || watchdog.overdue) notice = 'The training supervisor needs attention. See the watchdog panel below.';
  $('notice').hidden = !notice;
  set('notice', notice);
  set('last-updated', `Updated ${new Date(data.sampled_utc).toLocaleTimeString()} · refresh 5s`);
  refreshTimes(); drawChart();
}

function drawChart() {
  if (!current) return;
  const choice = $('chart-model').value;
  const model = choice === 'current' ? current.models.find(model => model.active) || [...current.models].reverse().find(model => model.series.length) : current.models.find(model => model.id === choice);
  const canvas = $('loss-chart'), rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
  const width = rect.width, height = rect.height, left = 43, right = 15, top = 18, bottom = 27;
  const plotW = width - left - right, plotH = height - top - bottom;
  ctx.font = '9px Consolas, monospace'; ctx.lineWidth = 1;
  const series = model?.series || [];
  if (!series.length) {
    canvas.setAttribute('aria-label', 'Training loss: awaiting optimizer records');
    ctx.fillStyle = '#94a088'; ctx.textAlign = 'center'; ctx.fillText('Waiting for training records', width / 2, height / 2);
    set('chart-label', current.kind === 'frozen_retrieval' ? 'Encoder preparation precedes head training' : 'Waiting for epoch records'); return;
  }
  let low = Math.min(...series.map(point => point.loss)), high = Math.max(...series.map(point => point.loss));
  const pad = Math.max((high - low) * .1, .05); low -= pad; high += pad;
  const maxEpoch = Math.max(2, model.epochs), x = epoch => left + (epoch - 1) / (maxEpoch - 1) * plotW;
  const y = loss => top + (high - loss) / (high - low) * plotH;
  for (let i = 0; i <= 4; i++) {
    const value = low + (high - low) * i / 4, yy = y(value);
    ctx.strokeStyle = '#edf1e7'; ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(width - right, yy); ctx.stroke();
    ctx.fillStyle = '#a2ad98'; ctx.textAlign = 'right'; ctx.fillText(value.toFixed(2), left - 9, yy + 3);
  }
  ctx.textAlign = 'center'; ctx.fillStyle = '#a2ad98';
  [1, Math.round(maxEpoch / 4), Math.round(maxEpoch / 2), Math.round(maxEpoch * .75), maxEpoch].forEach(epoch => ctx.fillText(String(epoch), x(epoch), height - 8));
  const gradient = ctx.createLinearGradient(0, top, 0, height - bottom); gradient.addColorStop(0, '#e9f0e4'); gradient.addColorStop(1, '#ffffff');
  ctx.beginPath(); ctx.moveTo(x(series[0].epoch), height - bottom); series.forEach(point => ctx.lineTo(x(point.epoch), y(point.loss))); ctx.lineTo(x(series.at(-1).epoch), height - bottom); ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();
  ctx.beginPath(); series.forEach((point, index) => index ? ctx.lineTo(x(point.epoch), y(point.loss)) : ctx.moveTo(x(point.epoch), y(point.loss))); ctx.lineWidth = 1.8; ctx.lineJoin = 'round'; ctx.strokeStyle = '#648451'; ctx.stroke();
  const last = series.at(-1); ctx.beginPath(); ctx.arc(x(last.epoch), y(last.loss), 3, 0, Math.PI * 2); ctx.fillStyle = '#426c37'; ctx.fill();
  const first = series[0];
  set('chart-label', `${modelLabel(model)} · loss ${num(first.loss, 3)} → ${num(last.loss, 3)}`);
  canvas.setAttribute('aria-label', `Training loss for ${modelLabel(model)}, from ${num(first.loss, 3)} at ${first.epoch} to ${num(last.loss, 3)} at ${last.epoch}.`);
}

async function refresh() {
  if (fetching) return;
  fetching = true; $('refresh').disabled = true;
  try {
    const response = await fetch('/api/status', {cache: 'no-store', signal: AbortSignal.timeout(6000)});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    lastSuccess = Date.now();
    render(data); set('connection', 'Live · 5s'); $('connection').classList.remove('offline');
  } catch (error) {
    set('connection', 'Disconnected'); $('connection').classList.add('offline');
    $('notice').hidden = false;
    set('notice', 'Cannot reach the local monitor. Start training-monitor.cmd, then refresh this page. Training runs independently.');
  } finally {
    fetching = false; $('refresh').disabled = false;
  }
}
$('refresh').addEventListener('click', refresh);
$('chart-model').addEventListener('change', drawChart);
new ResizeObserver(drawChart).observe($('loss-chart').parentElement);
setInterval(refresh, 5000); setInterval(refreshTimes, 1000); refresh();
