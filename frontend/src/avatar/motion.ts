import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { VRM, VRMHumanBoneName } from "@pixiv/three-vrm";
import { VRMAnimationLoaderPlugin, createVRMAnimationClip, type VRMAnimation } from "@pixiv/three-vrm-animation";

export type Emotion = "neutral" | "happy" | "sad" | "angry" | "surprised" | "shy" | "relaxed";
export const EMOTIONS: Emotion[] = ["neutral", "happy", "sad", "angry", "surprised", "shy", "relaxed"];

export type Gesture = "greet" | "nod" | "shake" | "think";
export const GESTURES: Gesture[] = ["greet", "nod", "shake", "think"];

/** VRM preset expressions driven by the emotion layer. */
export const EMOTION_KEYS = ["happy", "angry", "sad", "relaxed", "surprised"] as const;
export type EmotionKey = (typeof EMOTION_KEYS)[number];

export const EMOTION_EXPRESSIONS: Record<Emotion, Partial<Record<EmotionKey, number>>> = {
    neutral: {},
    happy: { happy: 1 },
    sad: { sad: 1 },
    angry: { angry: 1 },
    surprised: { surprised: 1 },
    relaxed: { relaxed: 1 },
    // No blush expression on the sample models: soft smile + lowered gaze and tilted head.
    shy: { happy: 0.45, relaxed: 0.3 },
};

export interface EmotionPose { headTilt: number; headPitch: number; lookY: number }
export const EMOTION_POSES: Record<Emotion, EmotionPose> = {
    neutral: { headTilt: 0, headPitch: 0, lookY: 0 },
    happy: { headTilt: 0.05, headPitch: -0.03, lookY: 0 },
    sad: { headTilt: 0, headPitch: 0.1, lookY: -0.25 },
    angry: { headTilt: 0, headPitch: 0.05, lookY: 0 },
    surprised: { headTilt: 0, headPitch: -0.06, lookY: 0.05 },
    relaxed: { headTilt: 0.06, headPitch: 0, lookY: 0 },
    shy: { headTilt: 0.12, headPitch: 0.1, lookY: -0.35 },
};

export async function loadVRMAClip(url: string, vrm: VRM): Promise<THREE.AnimationClip> {
    const loader = new GLTFLoader();
    loader.register(parser => new VRMAnimationLoaderPlugin(parser));
    const gltf = await loader.loadAsync(url);
    const anim = (gltf.userData.vrmAnimations as VRMAnimation[] | undefined)?.[0];
    if (!anim) throw new Error(`no VRMA animation in ${url}`);
    return createVRMAnimationClip(anim, vrm);
}

type Euler3 = [number, number, number];
type Key = [number, Partial<Record<VRMHumanBoneName, Euler3>>];

/** Build a clip on normalized bones from Euler keys (VRM1 axes; `axis` = -1 mirrors x/z for VRM0). */
function keyClip(vrm: VRM, name: string, keys: Key[], axis: number): THREE.AnimationClip {
    const tracks: THREE.KeyframeTrack[] = [];
    const bones = new Set(keys.flatMap(([, b]) => Object.keys(b) as VRMHumanBoneName[]));
    const q = new THREE.Quaternion(), e = new THREE.Euler();
    for (const bone of bones) {
        const node = vrm.humanoid.getNormalizedBoneNode(bone);
        if (!node) continue;
        const times: number[] = [], values: number[] = [];
        for (const [t, b] of keys) {
            const r = b[bone];
            if (!r) continue;
            q.setFromEuler(e.set(r[0] * axis, r[1], r[2] * axis));
            times.push(t);
            values.push(q.x, q.y, q.z, q.w);
        }
        tracks.push(new THREE.QuaternionKeyframeTrack(`${node.name}.quaternion`, times, values));
    }
    return new THREE.AnimationClip(name, -1, tracks);
}

const ARMS_REST: Key[1] = {
    leftUpperArm: [0, 0, -1.2], leftLowerArm: [0, -0.25, 0],
    rightUpperArm: [0, 0, 1.2], rightLowerArm: [0, 0.25, 0],
};

export function gestureClips(vrm: VRM, axis: number): Partial<Record<Gesture, THREE.AnimationClip>> {
    const greetUp: Key[1] = { rightUpperArm: [0, 0.9, 0.95], rightLowerArm: [0, 0.2, -2.3] };
    const greet = keyClip(vrm, "greet", [
        [0, ARMS_REST],
        [0.35, { ...ARMS_REST, ...greetUp }],
        [0.6, { rightLowerArm: [0, 0.2, -2.0] }],
        [0.85, { rightLowerArm: [0, 0.2, -2.55] }],
        [1.1, { rightLowerArm: [0, 0.2, -2.0] }],
        [1.35, { rightLowerArm: [0, 0.2, -2.55] }],
        [1.6, { ...ARMS_REST, ...greetUp }],
        [2.1, ARMS_REST],
    ], axis);
    const chin: Key[1] = { rightUpperArm: [0, 0.9, 1.25], rightLowerArm: [0, 2.5, 0], rightHand: [0, 0, -0.3] };
    const think = keyClip(vrm, "think", [
        [0, { ...ARMS_REST, rightHand: [0, 0, 0] }],
        [0.5, { ...ARMS_REST, ...chin }],
        [2.5, { ...ARMS_REST, ...chin }],
        [3.0, { ...ARMS_REST, rightHand: [0, 0, 0] }],
    ], axis);
    return { greet, think };
}
