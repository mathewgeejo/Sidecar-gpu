// Browser-executable TypeScript: intentionally uses no syntax requiring transpilation.
const el = (id) => document.getElementById(id);
let token = '';
let selected = null;
let jobs = [];
let loading = false;
const bytes = (n) => n == null ? 'Unavailable' : n >= 1073741824 ? `${(n/1073741824).toFixed(2)} GiB` : n >= 1048576 ? `${(n/1048576).toFixed(1)} MiB` : `${n.toLocaleString()} B`;
const duration = (n) => n == null ? 'Unavailable' : n < 1 ? `${(n*1000).toFixed(2)} ms` : `${n.toFixed(2)} s`;
const text = (id, value) => { el(id).textContent = String(value); };
const node = (tag, value, className = '') => { const n = document.createElement(tag); n.textContent = value; n.className = className; return n; };

async function api(path, options = {}) {
  const response = await fetch(`/api/v1${path}`, {...options, headers: {'Authorization': `Bearer ${token}`, ...(options.headers || {})}});
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error?.message || `HTTP ${response.status}`);
  }
  return response;
}

function connection(message, live) {
  text('connection-label', message);
  el('connection-label').parentElement.classList.toggle('live', live);
}

function details(job) {
  selected = job.id;
  text('detail-id', job.id.slice(0, 14));
  text('job-detail', JSON.stringify(job, null, 2));
  text('timing-job', job.id.slice(0, 14));
  text('worker-total', duration(job.timings.worker_total_s));
  const container = el('timing-bars');
  container.replaceChildren();
  const labels = [['Queue wait', 'queue_wait_s'], ['Model loading', 'model_load_s'], ['Input loading', 'input_load_s'], ['Host → device', 'host_to_device_s'], ['Compute', 'compute_s'], ['Device → host', 'device_to_host_s'], ['Serialization', 'serialization_s'], ['Result commit', 'result_commit_s']];
  const maximum = Math.max(...labels.map(([, key]) => job.timings[key] || 0), 0.000001);
  for (const [label, key] of labels) {
    const row = node('div', '', `timing-row ${key === 'compute_s' ? 'compute' : ''}`);
    row.append(node('span', label));
    const track = node('div', '', 'bar-track');
    const fill = node('div', '', 'bar-fill');
    fill.style.width = `${Math.min(100, (job.timings[key] || 0) / maximum * 100)}%`;
    track.append(fill);
    row.append(track, node('span', duration(job.timings[key]), 'timing-value'));
    container.append(row);
  }
  document.querySelectorAll('tr[data-job]').forEach(row => row.classList.toggle('selected', row.dataset.job === selected));
}

async function act(action) {
  try { text('action-message', ''); await action(); await refresh(); }
  catch (error) { text('action-message', error.message); }
}

function renderJobs() {
  const body = el('job-rows');
  body.replaceChildren();
  text('job-count', `${jobs.length} jobs`);
  el('jobs-empty').hidden = jobs.length > 0;
  for (const job of jobs) {
    const row = node('tr', ''); row.dataset.job = job.id;
    const identity = node('td', '');
    const button = node('button', `${job.id.slice(0, 14)}…`, 'job-id');
    button.title = job.id; button.addEventListener('click', () => details(job));
    identity.append(button, node('small', job.operation));
    const state = node('td', ''); state.append(node('span', job.state, `state ${job.state}`));
    const queue = node('td', duration(job.timings.queue_wait_s));
    const execution = node('td', duration(job.timings.worker_total_s));
    const cache = node('td', job.result ? (job.result.cache_hit ? 'Hit' : 'New object') : '—');
    const actions = node('td', ''); const controls = node('div', '', 'actions');
    if (['queued', 'running', 'cancel_requested'].includes(job.state)) {
      const cancel = node('button', job.state === 'cancel_requested' ? 'Stopping…' : 'Cancel');
      cancel.disabled = job.state === 'cancel_requested';
      cancel.addEventListener('click', () => act(() => api(`/jobs/${job.id}/cancel`, {method:'POST'})));
      controls.append(cancel);
    }
    if (job.state === 'succeeded') {
      const download = node('button', '↓ Result');
      download.addEventListener('click', () => act(async () => {
        const response = await api(`/jobs/${job.id}/result/download`);
        const url = URL.createObjectURL(await response.blob());
        const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${job.id}.npy`;
        anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 30000);
      }));
      controls.append(download);
    }
    actions.append(controls);
    row.append(identity, state, queue, execution, cache, actions);
    body.append(row);
  }
  const choice = jobs.find(job => job.id === selected) || jobs[0];
  if (choice) details(choice);
}

async function refresh() {
  if (!token || loading) return;
  loading = true;
  try {
    const [cap, metrics, recent] = await Promise.all([api('/capabilities').then(r => r.json()), api('/metrics').then(r => r.json()), api('/jobs').then(r => r.json())]);
    const device = cap.device;
    connection(cap.ready ? 'Authenticated · worker ready' : 'Authenticated · executor unavailable', cap.ready);
    text('device-type', device.device === 'cuda' ? 'CUDA' : device.device === 'cpu' ? 'CPU' : 'Unavailable');
    text('device-name', device.device_name || 'Unavailable');
    text('queue-depth', metrics.queue_depth);
    text('queue-limit', `${cap.limits.queue_limit} waiting slots · 1 executing`);
    text('object-count', metrics.object_count);
    text('storage-size', `${bytes(metrics.storage_bytes)} stored on disk`);
    text('cache-hits', metrics.counters.object_lookup_hits || 0);
    text('executor-status', cap.ready ? 'READY' : 'UNAVAILABLE');
    text('hardware-name', device.device_name || 'Unavailable');
    text('torch-version', device.torch_version ? `PyTorch ${device.torch_version} · executor ${device.executor_pid}` : 'Unavailable');
    text('gpu-memory', device.gpu_total_bytes ? `${bytes(device.gpu_free_bytes)} / ${bytes(device.gpu_total_bytes)}` : 'Unavailable');
    text('host-memory', bytes(device.host_memory_total_bytes));
    text('cuda-version', device.cuda_version || 'Unavailable');
    text('telemetry-time', device.sampled_at ? `Last device sample: ${new Date(device.sampled_at*1000).toLocaleTimeString()}` : 'Device telemetry unavailable.');
    el('storage-bar').style.width = `${Math.min(100, metrics.storage_bytes / metrics.disk_budget_bytes * 100)}%`;
    text('storage-budget', `${bytes(metrics.storage_bytes)} used of ${bytes(metrics.disk_budget_bytes)}`);
    text('storage-reservations', `${bytes(metrics.reserved_bytes)} reserved for uploads and accepted outputs`);
    text('model-state', `Encoder: ${device.model_state || 'not loaded'} · ${device.model_cache?.length || 0} model(s) in executor memory`);
    jobs = recent.jobs;
    renderJobs();
    text('last-updated', `Updated ${new Date().toLocaleTimeString()}`);
    el('pair-form').hidden = true;
    el('console').hidden = false;
    el('refresh').disabled = false;
  } catch (error) {
    connection('Connection unavailable', false);
    text('pair-error', error.message);
    text('action-message', error.message);
    throw error;
  } finally { loading = false; }
}

el('pair-form').addEventListener('submit', async event => {
  event.preventDefault(); token = el('token').value.trim();
  try { await refresh(); el('token').value = ''; text('pair-error', ''); }
  catch { token = ''; }
});
el('refresh').addEventListener('click', () => refresh().catch(() => {}));
el('disconnect').addEventListener('click', () => {
  token = ''; jobs = []; selected = null;
  el('console').hidden = true; el('pair-form').hidden = false; el('refresh').disabled = true;
  text('pair-error', ''); connection('Disconnected', false);
});
setInterval(() => refresh().catch(() => {}), 2000);
fetch('/api/v1/health').then(r => r.json()).then(health => connection(`Worker ${health.status} · pairing required`, false)).catch(() => connection('Worker unreachable', false));
