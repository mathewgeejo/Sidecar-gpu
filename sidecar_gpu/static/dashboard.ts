// Buildless module: browser-executable JavaScript in the existing TypeScript entrypoint.
import {sha256, makeMatrices, validateMatrix} from '/workload.js';
const el = id => document.getElementById(id);
const text = (id, value) => {
  const element=el(id), next=String(value);
  if(element.textContent===next) return;
  element.textContent=next;
  if(element.parentElement?.classList.contains('stat')) {
    element.classList.remove('value-changed');
    requestAnimationFrame(()=>element.classList.add('value-changed'));
  }
};
const node = (tag, value = '', className = '') => { const n = document.createElement(tag); n.textContent = value; n.className = className; return n; };
const icon = name => { const svg = document.createElementNS('http://www.w3.org/2000/svg','svg'); const use = document.createElementNS(svg.namespaceURI,'use'); use.setAttribute('href','#i-'+name); svg.setAttribute('aria-hidden','true'); svg.append(use); return svg; };
const bytes = n => n == null ? 'Not available' : n >= 1073741824 ? (n/1073741824).toFixed(2)+' GiB' : n >= 1048576 ? (n/1048576).toFixed(1)+' MiB' : n >= 1024 ? (n/1024).toFixed(1)+' KiB' : n+' B';
const duration = n => n == null ? '—' : n < 1 ? (n*1000).toFixed(2)+' ms' : n.toFixed(2)+' s';
let token = '', selected = null, jobs = [], loading = false, ready = false, busy = false;
let epoch = 0, session = new AbortController(), requests = 0, phase = 'idle';
let rowsSignature = '', detailSignature = '', toastTimer, receipt = null;
const observed = new Map(), events = [], responseSamples=[];
let guidedStep=-1, guideTimer=null;
try { receipt = JSON.parse(sessionStorage.getItem('sidecar-submission') || 'null'); } catch {}
if (receipt?.origin !== location.origin) receipt = null;

function connection(message, live) {
  text('connection-label', message);
  el('connection-label').parentElement.classList.toggle('live',live);
}
function event(message) {
  events.unshift({message,time:new Date()}); events.splice(8);
  el('event-list').replaceChildren(...events.map(entry => {
    const item = node('li'); item.append(node('span','','event-dot'),node('span',entry.message));
    const time = node('time',entry.time.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}));
    time.dateTime = entry.time.toISOString(); item.append(time); return item;
  }));
}
function toast(message) {
  text('toast',message); el('toast').hidden=false;
  clearTimeout(toastTimer); toastTimer=setTimeout(() => { el('toast').hidden=true; },4500);
}
function updateFlow() {
  if(guidedStep>=0) return;
  const active = jobs.find(job => ['running','cancel_requested'].includes(job.state));
  const mode = ['upload','download'].includes(phase) ? phase : phase === 'execute' && active ? 'running' : requests ? 'request' : active ? 'running' : 'idle';
  el('flow-panel').dataset.flow = mode;
  const labels = {upload:'Uploading inputs to the worker',download:'Retrieving a worker result',running:'Worker reports an executing job',request:'Request in flight to the worker',idle:ready?'Idle · ready for the next job':'Executor unavailable'};
  text('flow-caption',labels[mode]);
}
async function api(path, options = {}) {
  const controller = new AbortController(), current = session;
  const abort = () => controller.abort();
  current.signal.addEventListener('abort',abort,{once:true});
  const timer = setTimeout(abort,30000);
  requests++; updateFlow();
  try {
    const response = await fetch('/api/v1'+path,{...options,cache:'no-store',signal:controller.signal,headers:{'Authorization':'Bearer '+token,...options.headers}});
    if (!response.ok) {
      let message = 'Worker returned HTTP '+response.status;
      try { message = (await response.json()).error?.message || message; } catch {}
      const error = new Error(message); error.status=response.status; throw error;
    }
    return response;
  } finally { clearTimeout(timer); current.signal.removeEventListener('abort',abort); requests--; updateFlow(); }
}
const json = async (path, options) => (await api(path,options)).json();
function saveReceipt(value) {
  receipt=value;
  try { if(value) sessionStorage.setItem('sidecar-submission',JSON.stringify(value)); else sessionStorage.removeItem('sidecar-submission'); } catch {}
  el('recover-submission').hidden = !value;
  el('run-workload').disabled = !token || !ready || busy || !!value;
}
function details(job) {
  selected=job.id;
  text('detail-id',job.id.slice(0,16)); text('job-detail',JSON.stringify(job,null,2));
  text('timing-job',job.id.slice(0,15)); text('worker-total',duration(job.timings.worker_total_s));
  document.querySelectorAll('tr[data-job]').forEach(row => row.classList.toggle('selected',row.dataset.job===selected));
  const signature=JSON.stringify([job.id,job.timings]);
  if(signature===detailSignature) return;
  detailSignature=signature;
  const labels=[['Queue wait','queue_wait_s'],['Model load','model_load_s'],['Input load','input_load_s'],['To device','host_to_device_s'],['Compute','compute_s'],['From device','device_to_host_s'],['Serialization','serialization_s'],['Result commit','result_commit_s']];
  const available=labels.filter(([,key]) => typeof job.timings[key]==='number');
  if(!available.length) { el('timing-bars').replaceChildren(node('p','Timings arrive after the worker completes this job.','empty')); return; }
  const maximum=Math.max(...available.map(([,key])=>job.timings[key]),0.000001);
  el('timing-bars').replaceChildren(...available.map(([label,key]) => {
    const row=node('div','','timing-row'+(key==='compute_s'?' compute':''));
    const track=node('div','','bar-track'), fill=node('div','','bar-fill');
    fill.style.width=Math.min(100,job.timings[key]/maximum*100)+'%'; track.append(fill);
    row.append(node('span',label),track,node('span',duration(job.timings[key]),'timing-value')); return row;
  }));
}
async function act(action) {
  try { text('action-message',''); await action(); await refresh(); }
  catch(error) { text('action-message',error.message); toast(error.message); }
}
async function verifiedDownload(jobId) {
  const metadata=await json('/jobs/'+jobId+'/result');
  const response=await api('/objects/'+metadata.handle+'/download');
  const buffer=await response.arrayBuffer();
  if(buffer.byteLength!==metadata.bytes || await sha256(buffer)!==metadata.sha256) throw new Error('Result integrity check failed. The file was not saved.');
  return buffer;
}
function saveFile(buffer,name) {
  const url=URL.createObjectURL(new Blob([buffer],{type:'application/octet-stream'}));
  const anchor=node('a'); anchor.href=url; anchor.download=name; document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(()=>URL.revokeObjectURL(url),30000);
}
function renderJobs(force=false) {
  const search=el('job-search').value.toLowerCase(), filter=el('job-filter').value;
  const visible=jobs.filter(job => (filter==='all'||job.state===filter) && (job.id+' '+job.operation+' '+job.state).toLowerCase().includes(search));
  const signature=JSON.stringify([visible,search,filter]);
  if(!force && signature===rowsSignature) return;
  rowsSignature=signature;
  text('job-count',visible.length); el('jobs-empty').hidden=visible.length>0;
  if(!visible.length && jobs.length) el('jobs-empty').replaceChildren(node('h3','No matching jobs.'),node('p','Try a different search or state.'));
  else if(!jobs.length) el('jobs-empty').replaceChildren(icon('jobs'),node('h3','A little quiet in here.'),node('p','Run a workload to see its journey from inputs to results.'));
  el('job-rows').replaceChildren(...visible.map(job => {
    const row=node('tr'); row.dataset.job=job.id;
    const identity=node('td'), button=node('button',job.id.slice(0,16)+'…','job-id');
    button.title=job.id; button.addEventListener('click',()=>details(job));
    identity.append(button,node('small',job.operation==='matrix.multiply'?'Matrix multiplication':'Image embeddings'));
    const state=node('td'); state.append(node('span',job.state.replaceAll('_',' '),'state '+job.state));
    const result=node('td',job.result ? job.result.cache_hit?'Reused object':'Stored object':'—');
    const actions=node('div','','actions');
    if(['queued','running','cancel_requested'].includes(job.state)) {
      const cancel=node('button',job.state==='cancel_requested'?'Stopping…':'Cancel');
      cancel.disabled=job.state==='cancel_requested';
      cancel.addEventListener('click',()=>act(async()=>{ await json('/jobs/'+job.id+'/cancel',{method:'POST'}); event('Cancellation requested · '+job.id.slice(0,12)); }));
      actions.append(cancel);
    }
    if(job.state==='succeeded') {
      const download=node('button'); download.append(icon('down'),node('span','Result'));
      download.addEventListener('click',()=>act(async()=>{
        download.disabled=true; phase='download'; updateFlow();
        try { const buffer=await verifiedDownload(job.id); saveFile(buffer,job.id+'.npy'); event('Result hash verified · '+job.id.slice(0,12)); toast('Result verified and downloaded.'); }
        finally { phase='idle'; download.disabled=false; updateFlow(); }
      }));
      actions.append(download);
    }
    const controls=node('td'); controls.append(actions);
    row.append(identity,state,node('td',duration(job.timings.queue_wait_s)),node('td',duration(job.timings.worker_total_s)),result,controls);
    row.classList.toggle('selected',job.id===selected); return row;
  }));
}
async function refresh() {
  if(!token || loading) return;
  loading=true;
  const generation=epoch, started=performance.now();
  try {
    let responseMs=0;
    const [cap,metrics,recent]=await Promise.all([
      json('/capabilities').then(value=>{responseMs=performance.now()-started;return value;}),
      json('/metrics'),json('/jobs'),
    ]);
    if(generation!==epoch || !token) return;
    ready=cap.ready;
    const device=cap.device, cuda=device.device==='cuda', stateCounts=metrics.jobs||{};
    connection(ready?'Worker connected':'Executor unavailable',ready);
    text('mode-label',cuda?'CUDA execution':'CPU execution');
    text('mode-description',cuda?'Jobs execute on this worker’s NVIDIA CUDA device.':'This worker computes on its CPU. CUDA offloading is not active.');
    text('rail-mode',cuda?'CUDA worker connected':'CPU worker connected'); text('rail-session-address',location.host);
    text('device-type',cuda?'NVIDIA CUDA':'CPU RUNTIME'); text('hardware-name',cuda?device.device_name:'CPU executor');
    text('device-name',cuda?device.device_name:'Persistent PyTorch execution process');
    text('execution-mode',cuda?'CUDA':'CPU'); text('flow-worker-name',cuda?'CUDA worker':'CPU worker');
    text('flow-worker-description','Persistent execution process'); text('flow-device',cuda?'NVIDIA · CUDA':'CPU · PyTorch');
    text('worker-address',location.host);
    text('transport-description',['localhost','127.0.0.1','[::1]'].includes(location.hostname)?'Loopback address':'Network worker address');
    text('api-latency',responseMs.toFixed(0)+' ms response'); recordResponse(responseMs);
    text('dialog-device',cuda?device.device_name:'CPU · no CUDA execution');
    text('total-jobs',Object.values(stateCounts).reduce((sum,n)=>sum+n,0));
    text('completed-count',(stateCounts.succeeded||0)+' completed successfully');
    text('active-jobs',(stateCounts.queued||0)+(stateCounts.running||0)+(stateCounts.cancel_requested||0));
    text('queue-depth',metrics.queue_depth); text('queue-limit',metrics.queue_depth+' waiting · '+cap.limits.queue_limit+' queue slots');
    text('object-count',metrics.object_count); text('storage-size',bytes(metrics.storage_bytes)+' stored on disk');
    text('cache-hits',metrics.counters.object_lookup_hits||0); text('executor-status',ready?'READY':'UNAVAILABLE');
    text('host-memory',bytes(device.host_memory_total_bytes));
    text('gpu-memory',cuda?bytes(device.gpu_free_bytes)+' / '+bytes(device.gpu_total_bytes):'Not in use');
    text('cuda-version',cuda?(device.cuda_version||'Not reported'):'Not active');
    text('torch-version','PyTorch '+(device.torch_version||'—')+' · process '+(device.executor_pid||'—'));
    text('telemetry-time',device.sampled_at?'Device sampled '+new Date(device.sampled_at*1000).toLocaleTimeString():'Device telemetry unavailable');
    const percent=metrics.disk_budget_bytes?metrics.storage_bytes/metrics.disk_budget_bytes*100:0;
    el('storage-bar').style.width=Math.min(100,percent)+'%';
    text('storage-used',bytes(metrics.storage_bytes)); text('storage-total','of '+bytes(metrics.disk_budget_bytes));
    text('storage-percent',percent<1?'< 1%':percent.toFixed(1)+'%');
    text('storage-budget','Immutable objects within the configured disk budget.');
    text('storage-reservations',bytes(metrics.reserved_bytes));
    text('model-state',device.model_state==='loaded'?(device.model_cache?.length||0)+' model cached':(device.model_state||'Not loaded').replaceAll('_',' '));
    jobs=recent.jobs;
    for(const job of jobs) {
      const previous=observed.get(job.id);
      if(previous && previous!==job.state) event(job.id.slice(0,12)+' · '+job.state.replaceAll('_',' '));
      observed.set(job.id,job.state);
    }
    text('nav-job-count',jobs.length); renderJobs();
    const choice=jobs.find(job=>job.id===selected)||jobs[0]; if(choice) details(choice);
    el('pair-form').hidden=true; el('console').hidden=false; el('refresh').disabled=false;
    el('run-workload').disabled=!ready||busy||!!receipt;
    if(receipt) { el('run-panel').hidden=false; el('recover-submission').hidden=false; }
    text('last-updated','Updated '+new Date().toLocaleTimeString());
  } catch(error) {
    if(generation===epoch) { connection('Connection unavailable',false); text('pair-error',error.message); text('action-message',error.message); ready=false; el('run-workload').disabled=true; }
    throw error;
  } finally { if(generation===epoch) loading=false; updateFlow(); }
}
function setStage(stage,message) {
  phase=stage; el('run-panel').hidden=false;
  const steps=['prepare','upload','execute','download','verify'], index=steps.indexOf(stage);
  document.querySelectorAll('.run-steps li').forEach((step,i)=>{step.classList.toggle('active',i===index);step.classList.toggle('done',stage==='complete'||i<index);});
  text('run-message',message); updateFlow();
}
async function finishRun(job, matrices) {
  setStage('execute','Job accepted. Waiting for the worker’s reported state.');
  selected=job.id;
  const deadline=Date.now()+180000;
  while(!['succeeded','failed','cancelled','interrupted'].includes(job.state)) {
    if(!token) throw new Error('Disconnected. Reconnect to recover the accepted job.');
    if(Date.now()>deadline) throw new Error('The job is still accepted. Recover it using its saved identity.');
    await new Promise(resolve=>setTimeout(resolve,250));
    job=await json('/jobs/'+job.id);
    text('run-message','Worker reports: '+job.state.replaceAll('_',' ')+'.'); await refresh();
  }
  if(job.state!=='succeeded') { saveReceipt(null); throw new Error(job.error?.message||'Job ended: '+job.state); }
  setStage('download','Downloading the worker’s result and checking its SHA-256.');
  const buffer=await verifiedDownload(job.id);
  setStage('verify','Comparing every result value against the known matrix reference.');
  const maxError=validateMatrix(buffer,matrices);
  saveReceipt(null); setStage('complete','Verified '+(matrices.size**2).toLocaleString()+' values · result available in Recent jobs.');
  text('run-title',matrices.size+' × '+matrices.size+' matrix job complete');
  text('run-outcome','Verified'); el('run-outcome').dataset.state='verified';
  el('run-outcome').title='SHA-256 checked. Maximum absolute numerical error: '+maxError;
  text('return-label','Result downloaded & verified');
  event('Browser workload verified · '+job.id.slice(0,12)); toast('Real worker result received and verified.');
  await refresh();
}
async function runWorkload(size) {
  busy=true; el('run-workload').disabled=true; el('run-dialog').close();
  text('run-title',size+' × '+size+' matrix multiplication'); text('run-outcome','In progress');
  el('run-outcome').dataset.state='running'; text('run-uploaded','0 B'); el('run-uploaded').dataset.bytes='0';
  let uploaded=0;
  try {
    setStage('prepare','Preparing two dense float32 matrices in this browser.');
    await new Promise(resolve=>requestAnimationFrame(resolve));
    const matrices=makeMatrices(size), objects=[];
    for(const [label,buffer] of [['A',matrices.a],['B',matrices.b]]) {
      const hash=await sha256(buffer);
      setStage('upload','Checking whether input '+label+' is already cached.');
      const lookup=await json('/objects/lookup/'+hash);
      let object;
      if(lookup.exists) { object=lookup.object; event('Input '+label+' reused · zero payload bytes'); }
      else { setStage('upload','Uploading input '+label+' to the worker.'); object=await json('/objects',{method:'POST',headers:{'X-Content-SHA256':hash,'Content-Type':'application/octet-stream'},body:buffer}); uploaded+=buffer.byteLength; event('Input '+label+' uploaded · '+bytes(buffer.byteLength)); }
      objects.push(object); text('run-uploaded',bytes(uploaded)); el('run-uploaded').dataset.bytes=String(uploaded);
    }
    const key='browser-'+Array.from(crypto.getRandomValues(new Uint8Array(16)),n=>n.toString(16).padStart(2,'0')).join('');
    const parameters={operation:'matrix.multiply',a:objects[0].handle,b:objects[1].handle,a_shape:[size,size],b_shape:[size,size],dtype:'float32',retain_output:true,retry_of:null};
    saveReceipt({origin:location.origin,key,parameters,size,jobId:null});
    setStage('execute','Submitting the job with a durable idempotency key.');
    let job;
    try { job=await json('/jobs',{method:'POST',headers:{'Content-Type':'application/json','Idempotency-Key':key},body:JSON.stringify(parameters)}); }
    catch(error) { if(error.status) { saveReceipt(null); throw error; } job=await json('/jobs/by-key/'+key); }
    saveReceipt({...receipt,jobId:job.id});
    event('Job accepted · '+job.id.slice(0,12));
    await finishRun(job,matrices);
  } catch(error) {
    text('run-outcome',receipt?'Recoverable':'Failed'); el('run-outcome').dataset.state='error';
    text('run-message',error.message+(receipt?' The submission identity is saved in this tab.':''));
    text('action-message',error.message); phase='idle'; updateFlow();
  } finally { busy=false; el('run-workload').disabled=!token||!ready||!!receipt; }
}
el('pair-form').addEventListener('submit',async eventObject=>{
  eventObject.preventDefault(); session.abort(); session=new AbortController(); epoch++; loading=false;
  token=el('token').value.trim();
  try { await refresh(); el('token').value=''; text('pair-error',''); event('Authenticated connection to worker established.'); }
  catch { token=''; }
});
el('refresh').addEventListener('click',()=>refresh().catch(()=>{}));
el('disconnect').addEventListener('click',()=>{
  stopGuide(); epoch++; session.abort(); session=new AbortController(); token=''; loading=false; ready=false; jobs=[]; phase='idle';
  el('console').hidden=true; el('pair-form').hidden=false; el('refresh').disabled=true; el('run-workload').disabled=true;
  text('pair-error',''); connection('Disconnected',false); text('rail-mode','Awaiting connection'); text('rail-session-address','Not connected'); updateFlow();
});
el('job-search').addEventListener('input',()=>renderJobs(true));
el('job-filter').addEventListener('change',()=>renderJobs(true));
el('run-workload').addEventListener('click',()=>{ stopGuide(); el('run-dialog').showModal(); });
el('close-run-dialog').addEventListener('click',()=>el('run-dialog').close());
el('run-form').addEventListener('submit',eventObject=>{eventObject.preventDefault();if(!busy)runWorkload(Number(el('matrix-size').value));});
el('recover-submission').addEventListener('click',async()=>{
  if(!receipt || busy) return;
  busy=true; el('recover-submission').disabled=true;
  try { const job=await json(receipt.jobId?'/jobs/'+receipt.jobId:'/jobs/by-key/'+receipt.key); await finishRun(job,makeMatrices(receipt.size)); }
  catch(error) { text('run-message','Recovery: '+error.message+'. Your original identity is retained.'); }
  finally { busy=false; el('recover-submission').disabled=false; el('run-workload').disabled=!ready||!!receipt; }
});
document.querySelectorAll('.nav-link').forEach(link=>link.addEventListener('click',()=>{
  document.querySelectorAll('.nav-link').forEach(item=>item.classList.toggle('active',item===link));
}));

function recordResponse(milliseconds) {
  responseSamples.push(milliseconds);
  if(responseSamples.length>24) responseSamples.shift();
  text('trace-value',Math.round(milliseconds)+' ms');
  if(responseSamples.length<2) return;
  const min=Math.min(...responseSamples), max=Math.max(...responseSamples,min+5);
  el('api-trace').setAttribute('d',responseSamples.map((value,index)=>{
    const x=2+index/(responseSamples.length-1)*86, y=19-(value-min)/(max-min)*16;
    return (index?'L':'M')+x.toFixed(1)+' '+y.toFixed(1);
  }).join(' '));
}
const guideSteps=[
  {node:'client',flow:'upload',caption:'01 / The client prepares float32 inputs and checks their hashes.'},
  {node:'transport',flow:'upload',caption:'02 / Authenticated HTTP carries inputs and a job request to the worker.'},
  {node:'worker',flow:'running',caption:'03 / The persistent PyTorch process executes on its configured CPU or CUDA device.'},
  {node:'result',flow:'download',caption:'04 / The client retrieves the result and verifies its hash and values.'},
];
function stopGuide() {
  clearTimeout(guideTimer); guidedStep=-1;
  el('flow-panel').dataset.guide='';
  el('flow-live').classList.add('active'); el('flow-live').setAttribute('aria-pressed','true');
  el('flow-tour').classList.remove('active'); el('flow-tour').setAttribute('aria-pressed','false');
  text('flow-mode-label','LIVE TELEMETRY'); text('flow-context','Observed requests & job states');
  updateFlow();
}
function showGuideStep() {
  const step=guideSteps[guidedStep];
  el('flow-panel').dataset.guide=step.node;
  el('flow-panel').dataset.flow=step.flow;
  text('flow-mode-label','WALKTHROUGH / '+(guidedStep+1)+' OF 4');
  text('flow-caption',step.caption);
  text('flow-context','Illustration only · no job submitted');
  guideTimer=setTimeout(()=>{
    guidedStep++;
    if(guidedStep>=guideSteps.length) stopGuide(); else showGuideStep();
  },2600);
}
el('flow-live').addEventListener('click',stopGuide);
el('flow-tour').addEventListener('click',()=>{
  clearTimeout(guideTimer); guidedStep=0;
  el('flow-live').classList.remove('active'); el('flow-live').setAttribute('aria-pressed','false');
  el('flow-tour').classList.add('active'); el('flow-tour').setAttribute('aria-pressed','true');
  showGuideStep();
});
let tiltFrame=0;
const diagram=el('flow-panel');
diagram.addEventListener('pointermove',event=>{
  if(event.pointerType==='touch'||matchMedia('(prefers-reduced-motion: reduce)').matches||tiltFrame) return;
  tiltFrame=requestAnimationFrame(()=>{
    const box=diagram.getBoundingClientRect();
    const x=(event.clientX-box.left)/box.width-.5, y=(event.clientY-box.top)/box.height-.5;
    diagram.style.setProperty('--tilt-x',(-y*6).toFixed(2)+'deg');
    diagram.style.setProperty('--tilt-y',(x*8).toFixed(2)+'deg');
    tiltFrame=0;
  });
});
diagram.addEventListener('pointerleave',()=>{
  cancelAnimationFrame(tiltFrame); tiltFrame=0;
  diagram.style.setProperty('--tilt-x','0deg'); diagram.style.setProperty('--tilt-y','0deg');
});
document.addEventListener('visibilitychange',()=>{ if(document.hidden)stopGuide(); });

setInterval(()=>{if(!document.hidden)refresh().catch(()=>{});},2000);
fetch('/api/v1/health',{cache:'no-store'}).then(response=>response.json()).then(health=>connection(health.status==='ready'?'Worker ready · pairing required':'Worker unavailable',false)).catch(()=>connection('Worker unreachable',false));
