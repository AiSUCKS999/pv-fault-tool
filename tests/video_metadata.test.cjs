// Exercise the metadata component without a browser or reading a large file.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const packets = [];
let onChange;
const file = {name: 'drone-flight.mp4', type: 'video/mp4', size: 20 * 1024 ** 3, lastModified: 1793200000000};
const picker = {files: [file], addEventListener: (name, fn) => { assert.equal(name, 'change'); onChange = fn; }};
const label = {};
let revoked = false;
const video = {duration: 127.5, videoWidth: 3840, videoHeight: 2160, removeAttribute() {}, load() {}};
Object.defineProperty(video, 'src', {set() { assert.equal(video.preload, 'metadata'); video.onloadedmetadata(); }});
const context = {
  parent: {postMessage: data => packets.push(data)},
  document: {getElementById: id => id === 'videoFile' ? picker : label, createElement: type => { assert.equal(type, 'video'); return video; }},
  URL: {createObjectURL: value => { assert.equal(value, file); return 'blob:local-file'; }, revokeObjectURL: () => { revoked = true; }},
  crypto: {randomUUID: () => 'browser-file-selection'}, setTimeout: () => 1, clearTimeout() {},
};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../pv_video_metadata/metadata.js'), 'utf8'), context);
onChange();
const value = packets.find(p => p.type === 'streamlit:setComponentValue').value;
assert.equal(value.name, file.name);
assert.equal(value.size_bytes, 20 * 1024 ** 3);
assert.equal(value.duration_seconds, 127.5);
assert.equal(value.width, 3840);
assert.equal(value.height, 2160);
assert.ok(revoked);
assert.ok(!('contents' in value));
assert.ok(!('data' in value));
console.log('video metadata: PASS (20 GiB virtual file; no FileReader, upload or content read)');
