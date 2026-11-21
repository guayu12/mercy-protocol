import * as THREE from "three";
import type { MToonMaterial, VRM } from "@pixiv/three-vrm";

export type LightingPreset = "studio" | "warm" | "cool" | "basic";
export const LIGHTING_PRESETS: LightingPreset[] = ["studio", "warm", "cool", "basic"];

interface LightSpec { color: number; intensity: number; pos: [number, number, number] }
interface Backdrop { center: number; edge: number; glow: number; glowStrength: number; bokeh: number; bokehColor: number }

export interface LookSpec {
    toneMapping: THREE.ToneMapping;
    exposure: number;
    hemi: { sky: number; ground: number; intensity: number };
    /** key, fill, rim, (optional second rim) — positions relative to the face. */
    lights: LightSpec[];
    backdrop: Backdrop | null;
    rim: { color: number; strength: number; power: number; lift: number } | null;
    shadeTint: { color: number; amount: number } | null;
    outline: { widthScale: number; color: number; amount: number } | null;
    bloom: { strength: number; radius: number; threshold: number } | null;
}

export const LOOKS: Record<LightingPreset, LookSpec> = {
    studio: {
        toneMapping: THREE.ACESFilmicToneMapping,
        exposure: 1.0,
        hemi: { sky: 0xfff3e6, ground: 0x3a2e2a, intensity: 0.46 },
        lights: [
            { color: 0xfff0dd, intensity: 1.18, pos: [-1.0, 1.0, 2.2] },
            { color: 0xd8e4ff, intensity: 0.38, pos: [1.6, 0.3, 1.6] },
            { color: 0xffcf9a, intensity: 1.34, pos: [1.0, 1.2, -2.0] },
            { color: 0x9fbfff, intensity: 0.67, pos: [-1.2, 0.8, -2.0] },
        ],
        backdrop: { center: 0x3a2d25, edge: 0x0f0d0d, glow: 0xffa860, glowStrength: 0.16, bokeh: 0.5, bokehColor: 0xffc890 },
        rim: { color: 0xffd2a8, strength: 0.35, power: 3.5, lift: 0.02 },
        shadeTint: { color: 0x9a6a78, amount: 0.18 },
        outline: { widthScale: 0.8, color: 0x2a1c1b, amount: 0.6 },
        bloom: { strength: 0.22, radius: 0.35, threshold: 0.95 },
    },
    warm: {
        toneMapping: THREE.ACESFilmicToneMapping,
        exposure: 1.1,
        hemi: { sky: 0xffe2c4, ground: 0x40281c, intensity: 0.42 },
        lights: [
            { color: 0xffc98a, intensity: 1.34, pos: [-1.4, 0.8, 1.8] },
            { color: 0xffe9d6, intensity: 0.29, pos: [1.6, 0.2, 1.6] },
            { color: 0xff9a50, intensity: 1.68, pos: [1.2, 1.0, -1.8] },
        ],
        backdrop: { center: 0x4a2e1c, edge: 0x140c08, glow: 0xff8a3c, glowStrength: 0.28, bokeh: 0.8, bokehColor: 0xffb060 },
        rim: { color: 0xffb070, strength: 0.5, power: 3.0, lift: 0.03 },
        shadeTint: { color: 0xb06a58, amount: 0.25 },
        outline: { widthScale: 0.8, color: 0x3a1e14, amount: 0.6 },
        bloom: { strength: 0.3, radius: 0.4, threshold: 0.95 },
    },
    cool: {
        toneMapping: THREE.ACESFilmicToneMapping,
        exposure: 1.1,
        hemi: { sky: 0xdfe8ff, ground: 0x1c2030, intensity: 0.42 },
        lights: [
            { color: 0xf2f5ff, intensity: 1.09, pos: [-1.0, 1.0, 2.2] },
            { color: 0xffd6f0, intensity: 0.34, pos: [1.6, 0.3, 1.6] },
            { color: 0x40d8ff, intensity: 1.51, pos: [1.1, 1.0, -2.0] },
            { color: 0xff5ab8, intensity: 1.01, pos: [-1.2, 0.8, -2.0] },
        ],
        backdrop: { center: 0x14203a, edge: 0x05070d, glow: 0x3ad0ff, glowStrength: 0.18, bokeh: 0.7, bokehColor: 0x7fd8ff },
        rim: { color: 0x9ee8ff, strength: 0.45, power: 3.2, lift: 0.02 },
        shadeTint: { color: 0x6a6aa8, amount: 0.22 },
        outline: { widthScale: 0.8, color: 0x141428, amount: 0.6 },
        bloom: { strength: 0.32, radius: 0.4, threshold: 0.95 },
    },
    basic: {
        toneMapping: THREE.NoToneMapping,
        exposure: 1,
        hemi: { sky: 0xffffff, ground: 0xffffff, intensity: 0.6 },
        lights: [{ color: 0xffffff, intensity: 1.2, pos: [0.6, 0.2, 2] }],
        backdrop: null, rim: null, shadeTint: null, outline: null, bloom: null,
    },
};

/** Original MToon values, so presets can be switched back and forth. */
interface MToonBase { shade: THREE.Color; rim: THREE.Color; power: number; lift: number; outlineWidth: number; outlineColor: THREE.Color }
const bases = new WeakMap<MToonMaterial, MToonBase>();

export function applyMToonLook(vrm: VRM, look: LookSpec) {
    vrm.scene.traverse(obj => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh) return;
        for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
            const mt = m as MToonMaterial;
            if (!mt.isMToonMaterial) continue;
            let b = bases.get(mt);
            if (!b) {
                b = {
                    shade: mt.shadeColorFactor.clone(), rim: mt.parametricRimColorFactor.clone(),
                    power: mt.parametricRimFresnelPowerFactor, lift: mt.parametricRimLiftFactor,
                    outlineWidth: mt.outlineWidthFactor, outlineColor: mt.outlineColorFactor.clone(),
                };
                bases.set(mt, b);
            }
            mt.shadeColorFactor.copy(b.shade);
            if (look.shadeTint) mt.shadeColorFactor.lerp(new THREE.Color(look.shadeTint.color), look.shadeTint.amount);
            if (look.rim) {
                mt.parametricRimColorFactor.copy(b.rim).add(new THREE.Color(look.rim.color).multiplyScalar(look.rim.strength));
                mt.parametricRimFresnelPowerFactor = look.rim.power;
                mt.parametricRimLiftFactor = look.rim.lift;
            } else {
                mt.parametricRimColorFactor.copy(b.rim);
                mt.parametricRimFresnelPowerFactor = b.power;
                mt.parametricRimLiftFactor = b.lift;
            }
            mt.outlineWidthFactor = b.outlineWidth * (look.outline?.widthScale ?? 1);
            mt.outlineColorFactor.copy(b.outlineColor);
            if (look.outline) mt.outlineColorFactor.lerp(new THREE.Color(look.outline.color), look.outline.amount);
        }
    });
}

const BACKDROP_FRAG = /* glsl */ `
uniform vec3 uCenter, uEdge, uGlow, uBokehColor;
uniform float uGlowStrength, uBokeh, uTime, uAspect;
varying vec2 vUv;
float hash(float n) { return fract(sin(n) * 43758.5453); }
void main() {
    vec2 p = vUv - vec2(0.5, 0.56);
    float d = length(p * vec2(min(uAspect, 1.6), 1.25));
    vec3 col = mix(uCenter, uEdge, smoothstep(0.02, 0.62, d));
    col += uGlow * uGlowStrength * exp(-d * d * 14.0);
    for (int i = 0; i < 14; i++) {
        float fi = float(i);
        vec2 c = vec2(hash(fi * 7.1), 0.15 + 0.8 * hash(fi * 3.7));
        vec2 asp = vec2(uAspect, 1.0);
        c.x += 0.015 * sin(uTime * (0.05 + 0.04 * hash(fi)) + fi);
        c.y += 0.01 * cos(uTime * (0.04 + 0.03 * hash(fi * 1.3)) + fi);
        float r = 0.02 + 0.05 * hash(fi * 5.3);
        float k = 1.0 - smoothstep(r * 0.65, r, length((vUv - c) * asp));
        col += uBokehColor * k * uBokeh * (0.05 + 0.12 * hash(fi * 9.9)) * smoothstep(0.1, 0.45, abs(vUv.x - 0.5));
    }
    col *= 1.0 - 0.35 * smoothstep(0.35, 0.85, d);
    col += (hash(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) - 0.5) / 255.0;
    gl_FragColor = vec4(col, 1.0);
}`;

/** Studio backdrop card: radial gradient, soft glow behind the head and out-of-focus light dots. */
export function createBackdrop() {
    const uniforms = {
        uCenter: { value: new THREE.Color() }, uEdge: { value: new THREE.Color() },
        uGlow: { value: new THREE.Color() }, uBokehColor: { value: new THREE.Color() },
        uGlowStrength: { value: 0 }, uBokeh: { value: 0 }, uTime: { value: 0 }, uAspect: { value: 1 },
    };
    const mat = new THREE.ShaderMaterial({
        uniforms,
        vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
        fragmentShader: BACKDROP_FRAG,
        depthWrite: false,
        toneMapped: false,
    });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
    mesh.renderOrder = -1;
    const set = (b: Backdrop | null) => {
        mesh.visible = !!b;
        if (!b) return;
        uniforms.uCenter.value.set(b.center); uniforms.uEdge.value.set(b.edge);
        uniforms.uGlow.value.set(b.glow); uniforms.uBokehColor.value.set(b.bokehColor);
        uniforms.uGlowStrength.value = b.glowStrength; uniforms.uBokeh.value = b.bokeh;
    };
    return { mesh, set, uniforms };
}
