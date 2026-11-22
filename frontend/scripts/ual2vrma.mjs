// Usage: node scripts/ual2vrma.mjs UAL1_Standard.glb Idle_Loop public/vrm/motions/ual_idle.vrma [fps]
// Offline UAL (UE mannequin, T-pose rest, +Z forward, meters) -> VRMA 1.0.
import fs from "fs";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
const [src, clipName, out, fpsArg] = process.argv.slice(2);
const FPS = +(fpsArg || 30);
const MAP = { hips: "pelvis", spine: "spine_01", chest: "spine_02", upperChest: "spine_03", neck: "neck_01", head: "Head" };
const PARENT = { spine: "hips", chest: "spine", upperChest: "chest", neck: "upperChest", head: "neck" };
for (const [S, s] of [["left", "l"], ["right", "r"]]) {
  Object.assign(MAP, { [`${S}Shoulder`]: `clavicle_${s}`, [`${S}UpperArm`]: `upperarm_${s}`, [`${S}LowerArm`]: `lowerarm_${s}`, [`${S}Hand`]: `hand_${s}`,
    [`${S}UpperLeg`]: `thigh_${s}`, [`${S}LowerLeg`]: `calf_${s}`, [`${S}Foot`]: `foot_${s}`, [`${S}Toes`]: `ball_${s}` });
  Object.assign(PARENT, { [`${S}Shoulder`]: "upperChest", [`${S}UpperArm`]: `${S}Shoulder`, [`${S}LowerArm`]: `${S}UpperArm`, [`${S}Hand`]: `${S}LowerArm`,
    [`${S}UpperLeg`]: "hips", [`${S}LowerLeg`]: `${S}UpperLeg`, [`${S}Foot`]: `${S}LowerLeg`, [`${S}Toes`]: `${S}Foot` });
  const fingers = { Thumb: ["Metacarpal", "Proximal", "Distal"], Index: ["Proximal", "Intermediate", "Distal"], Middle: ["Proximal", "Intermediate", "Distal"],
    Ring: ["Proximal", "Intermediate", "Distal"], Little: ["Proximal", "Intermediate", "Distal"] };
  const ue = { Thumb: "thumb", Index: "index", Middle: "middle", Ring: "ring", Little: "pinky" };
  for (const [F, segs] of Object.entries(fingers)) segs.forEach((seg, i) => {
    const b = `${S}${F}${seg}`; MAP[b] = `${ue[F]}_0${i + 1}_${s}`; PARENT[b] = i ? `${S}${F}${segs[i - 1]}` : `${S}Hand`;
  });
}
const BONES = Object.keys(MAP);

const buf = fs.readFileSync(src);
const gltf = await new GLTFLoader().parseAsync(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), "");
const scene = gltf.scene;
const clip = gltf.animations.find(a => a.name === clipName);
if (!clip) throw new Error("clip not found: " + clipName);
const node = b => { const o = scene.getObjectByName(MAP[b]); if (!o) throw new Error("missing " + MAP[b]); return o; };
const wq = b => node(b).getWorldQuaternion(new THREE.Quaternion());
const wp = b => node(b).getWorldPosition(new THREE.Vector3());

scene.updateMatrixWorld(true);
const restQ = {}, restP = {};
for (const b of BONES) { restQ[b] = wq(b); restP[b] = wp(b); }

const mixer = new THREE.AnimationMixer(scene);
mixer.clipAction(clip).play();
const n = Math.round(clip.duration * FPS) + 1;
const times = [], rot = Object.fromEntries(BONES.map(b => [b, []])), hipsPos = [];
for (let i = 0; i < n; i++) {
  const t = Math.min(i / FPS, clip.duration);
  mixer.setTime(t); scene.updateMatrixWorld(true);
  times.push(t);
  const N = {};
  for (const b of BONES) N[b] = wq(b).multiply(restQ[b].clone().invert());
  for (const b of BONES) {
    const local = PARENT[b] ? N[PARENT[b]].clone().invert().multiply(N[b]) : N[b];
    rot[b].push(...local.normalize().toArray());
  }
  hipsPos.push(...wp("hips").toArray());
}

// glTF/VRMA: one node per bone, identity rest rotations (T-pose), rest offsets from the source skeleton.
const idx = Object.fromEntries(BONES.map((b, i) => [b, i]));
const nodes = BONES.map(b => {
  const off = PARENT[b] ? restP[b].clone().sub(restP[PARENT[b]]) : restP[b];
  const children = BONES.filter(c => PARENT[c] === b).map(c => idx[c]);
  return { name: b, translation: off.toArray(), ...(children.length ? { children } : {}) };
});
const chunks = [], accessors = [], bufferViews = [];
let offset = 0;
const addAccessor = (arr, type, count, minmax) => {
  const f = new Float32Array(arr); const bytes = Buffer.from(f.buffer);
  bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: bytes.length });
  chunks.push(bytes); offset += bytes.length;
  accessors.push({ bufferView: bufferViews.length - 1, componentType: 5126, count, type, ...(minmax || {}) });
  return accessors.length - 1;
};
const input = addAccessor(times, "SCALAR", n, { min: [times[0]], max: [times[n - 1]] });
const samplers = [], channels = [];
for (const b of BONES) {
  samplers.push({ input, output: addAccessor(rot[b], "VEC4", n), interpolation: "LINEAR" });
  channels.push({ sampler: samplers.length - 1, target: { node: idx[b], path: "rotation" } });
}
samplers.push({ input, output: addAccessor(hipsPos, "VEC3", n), interpolation: "LINEAR" });
channels.push({ sampler: samplers.length - 1, target: { node: idx.hips, path: "translation" } });

const json = {
  asset: { version: "2.0", generator: "mercy-protocol ual2vrma (offline)" },
  extensionsUsed: ["VRMC_vrm_animation"],
  extensions: { VRMC_vrm_animation: { specVersion: "1.0", humanoid: { humanBones: Object.fromEntries(BONES.map(b => [b, { node: idx[b] }])) } } },
  scene: 0, scenes: [{ nodes: [idx.hips] }], nodes,
  buffers: [{ byteLength: offset }], bufferViews, accessors,
  animations: [{ name: clipName, samplers, channels }],
};
const pad = (b, c) => Buffer.concat([b, Buffer.alloc((4 - (b.length % 4)) % 4, c)]);
const jb = pad(Buffer.from(JSON.stringify(json)), 0x20), bb = pad(Buffer.concat(chunks), 0);
const header = Buffer.alloc(12); header.writeUInt32LE(0x46546c67, 0); header.writeUInt32LE(2, 4); header.writeUInt32LE(12 + 8 + jb.length + 8 + bb.length, 8);
const ch = (len, type) => { const h = Buffer.alloc(8); h.writeUInt32LE(len, 0); h.writeUInt32LE(type, 4); return h; };
fs.writeFileSync(out, Buffer.concat([header, ch(jb.length, 0x4e4f534a), jb, ch(bb.length, 0x004e4942), bb]));
console.log(out, clipName, `${clip.duration.toFixed(2)}s`, n, "frames", fs.statSync(out).size, "bytes");
