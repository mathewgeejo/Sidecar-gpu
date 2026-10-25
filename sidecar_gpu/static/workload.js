// Self-contained browser input/result helpers. No worker computation happens here.
const hex = data => Array.from(data, n => n.toString(16).padStart(2, '0')).join('');
const rotate = (x, n) => (x >>> n) | (x << (32 - n));
const constants = new Uint32Array([
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
]);

export async function sha256(input) {
  const source = input instanceof ArrayBuffer ? new Uint8Array(input) : input;
  if (globalThis.crypto?.subtle) return hex(new Uint8Array(await crypto.subtle.digest('SHA-256', source)));
  // Standard SHA-256 fallback for HTTP LAN origins where Web Crypto is unavailable.
  const padded = new Uint8Array(Math.ceil((source.length + 9) / 64) * 64);
  padded.set(source); padded[source.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, Math.floor(source.length / 0x20000000));
  view.setUint32(padded.length - 4, (source.length * 8) >>> 0);
  const hash = new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]);
  const words = new Uint32Array(64);
  for (let offset = 0; offset < padded.length; offset += 64) {
    for (let i = 0; i < 16; i++) words[i] = view.getUint32(offset + i * 4);
    for (let i = 16; i < 64; i++) {
      const x = words[i - 15], y = words[i - 2];
      words[i] = (words[i-16] + (rotate(x,7)^rotate(x,18)^(x>>>3)) + words[i-7] + (rotate(y,17)^rotate(y,19)^(y>>>10))) >>> 0;
    }
    let [a,b,c,d,e,f,g,h] = hash;
    for (let i = 0; i < 64; i++) {
      const t1 = (h + (rotate(e,6)^rotate(e,11)^rotate(e,25)) + ((e&f)^(~e&g)) + constants[i] + words[i]) >>> 0;
      const t2 = ((rotate(a,2)^rotate(a,13)^rotate(a,22)) + ((a&b)^(a&c)^(b&c))) >>> 0;
      h=g; g=f; f=e; e=(d+t1)>>>0; d=c; c=b; b=a; a=(t1+t2)>>>0;
    }
    [a,b,c,d,e,f,g,h].forEach((n,i) => { hash[i]=(hash[i]+n)>>>0; });
    if (offset && offset % 262144 === 0) await new Promise(resolve => setTimeout(resolve, 0));
  }
  return Array.from(hash, n => n.toString(16).padStart(8,'0')).join('');
}

function npy(size) {
  const definition = `{'descr': '<f4', 'fortran_order': False, 'shape': (${size}, ${size}), }`;
  const padding = (64 - (10 + definition.length + 1) % 64) % 64;
  const header = new TextEncoder().encode(definition + ' '.repeat(padding) + '\n');
  const buffer = new ArrayBuffer(10 + header.length + size * size * 4);
  const bytes = new Uint8Array(buffer);
  bytes.set([0x93,78,85,77,80,89,1,0]);
  new DataView(buffer).setUint16(8, header.length, true);
  bytes.set(header, 10);
  return {buffer, values: new Float32Array(buffer, 10 + header.length)};
}

export function makeMatrices(size) {
  if (![128,256,512,1024].includes(size)) throw new Error('Choose a supported matrix size.');
  const a = npy(size), b = npy(size), expectedRows = new Float64Array(size);
  let seed = 73129;
  for (let row = 0; row < size; row++) {
    let sum = 0;
    for (let col = 0; col < size; col++) {
      seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
      const index = row * size + col;
      a.values[index] = seed / 4294967296 - 0.5;
      sum += a.values[index];
      b.values[index] = 1 / size;
    }
    expectedRows[row] = sum / size;
  }
  return {a: a.buffer, b: b.buffer, expectedRows, size};
}

export function validateMatrix(buffer, matrices) {
  const bytes = new Uint8Array(buffer), view = new DataView(buffer);
  if (bytes.length < 12 || hex(bytes.slice(0,6)) !== '934e554d5059') throw new Error('Result is not a valid NPY file.');
  const version = bytes[6];
  if (version !== 1 && version !== 2) throw new Error('Unsupported result encoding.');
  const prefix = version === 1 ? 10 : 12;
  const length = version === 1 ? view.getUint16(8,true) : view.getUint32(8,true);
  const header = new TextDecoder().decode(bytes.slice(prefix,prefix+length));
  const shape = header.match(/'shape':\s*\((\d+),\s*(\d+)\)/);
  if (!header.includes("'<f4'") || !header.includes('False') || !shape || +shape[1] !== matrices.size || +shape[2] !== matrices.size || prefix+length+matrices.size**2*4 !== buffer.byteLength) throw new Error('Result shape or dtype differs from the submitted workload.');
  let maximum = 0;
  for (let row = 0; row < matrices.size; row++) for (let col = 0; col < matrices.size; col++) {
    const actual = view.getFloat32(prefix+length+(row*matrices.size+col)*4,true);
    const expected = matrices.expectedRows[row], error = Math.abs(actual-expected);
    if (!Number.isFinite(actual) || error > 1e-4+1e-4*Math.abs(expected)) throw new Error('Numerical verification failed.');
    maximum = Math.max(maximum,error);
  }
  return maximum;
}
