import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { VRM, VRMLoaderPlugin, VRMUtils, type VRMHumanBoneName } from "@pixiv/three-vrm";
import { VISEMES, visemeTargets, type Viseme, type VisemeCue } from "./lipsync";
import { LOOKS, applyMToonLook, createBackdrop, type LightingPreset } from "./look";
import { VRMLookAtQuaternionProxy } from "@pixiv/three-vrm-animation";
import {
    EMOTION_EXPRESSIONS, EMOTION_KEYS, EMOTION_POSES, gestureClips, loadVRMAClip,
    type Emotion, type EmotionKey, type Gesture,
} from "./motion";

export { EMOTIONS, GESTURES, type Emotion, type Gesture } from "./motion";

const IDLE_VRMA = "/vrm/motions/idle_loop.vrma";
const GESTURE_FADE = 0.3;
const PROC_GESTURE_SECONDS = 1.1;

export { LIGHTING_PRESETS, type LightingPreset } from "./look";

export type AvatarState = "idle" | "listening" | "thinking" | "speaking";
export const AVATAR_STATES: AvatarState[] = ["idle", "listening", "thinking", "speaking"];

export type Quality = "high" | "balanced";
export const QUALITIES: Quality[] = ["high", "balanced"];

/** face: thumbnail head shot; bust: head + shoulders; upper: waist up; auto picks from container size. */
export type Framing = "auto" | "face" | "bust" | "upper";
export const FRAMINGS: Framing[] = ["auto", "face", "bust", "upper"];
const FRAMES = {
    face: { viewH: 0.34, viewW: 0.3, faceFromTop: 0.5 },
    bust: { viewH: 0.62, viewW: 0.62, faceFromTop: 0.36 },
    upper: { viewH: 1.05, viewW: 0.75, faceFromTop: 0.24 },
};

export interface AvatarStats { fps: number; targetFps: number; drawCalls: number; triangles: number; textures: number; geometries: number }

/** Body-language parameters blended per state (radians / normalized offsets). */
interface Pose { lean: number; headTilt: number; headPitch: number; lookX: number; lookY: number; pointerFollow: number }
const POSES: Record<AvatarState, Pose> = {
    idle: { lean: 0, headTilt: 0, headPitch: 0, lookX: 0, lookY: 0, pointerFollow: 1 },
    listening: { lean: 0.06, headTilt: 0.1, headPitch: 0.04, lookX: 0, lookY: 0, pointerFollow: 0.3 },
    thinking: { lean: -0.02, headTilt: -0.08, headPitch: -0.1, lookX: -0.5, lookY: 0.45, pointerFollow: 0 },
    speaking: { lean: 0.03, headTilt: 0, headPitch: 0, lookX: 0, lookY: 0, pointerFollow: 0.4 },
};

export class AvatarEngine {
    vrm: VRM | null = null;
    readonly renderer: THREE.WebGLRenderer;
    readonly scene = new THREE.Scene();
    readonly camera = new THREE.PerspectiveCamera(20, 1, 0.1, 30);

    private container: HTMLElement;
    private timer = new THREE.Timer();
    private ro: ResizeObserver;
    private face = new THREE.Vector3(0, 1.4, 0);
    /** -1 for VRM0: rotateVRM0 turns the model 180deg, flipping x/z on normalized bones. */
    private axis = 1;
    private camBase = new THREE.Vector3();
    private camAim = new THREE.Vector3();
    private viewH = 1;
    private framing: Framing = "auto";
    private quality: Quality = "balanced";
    private onScreen = true;
    private io: IntersectionObserver;
    private lastFrame = 0;
    private lighting: LightingPreset = "studio";
    private composer: EffectComposer | null = null;
    private bloom: UnrealBloomPass | null = null;
    private backdrop = createBackdrop();
    private lookTarget = new THREE.Object3D();
    private pointer = new THREE.Vector2();
    private lights: THREE.Light[] = [];

    private state: AvatarState = "idle";
    private pose: Pose = { ...POSES.idle };
    private reactUntil = 0;
    private happy = 0;
    private nextBlink = 2;
    private blinkStart = -1;

    private audio = new Audio();
    private cues: VisemeCue[] = [];
    private clock: (() => number | null) | null = null;
    private onSpeechEnd: (() => void) | null = null;
    private mouth: Record<Viseme, number> = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
    private overrides = new Map<string, number>();
    private emotion: Emotion = "neutral";
    private speechEmotion: Emotion | null = null;
    private emo: Record<EmotionKey, number> = { happy: 0, angry: 0, sad: 0, relaxed: 0, surprised: 0 };
    /** Peak weight of emotion expressions (tunable from the debug panel). */
    emotionWeight = 0.7;
    private mixer: THREE.AnimationMixer | null = null;
    private baseAction: THREE.AnimationAction | null = null;
    private clips: Partial<Record<Gesture, THREE.AnimationClip>> = {};
    private gestureAction: THREE.AnimationAction | null = null;
    private fadingGesture: THREE.AnimationAction | null = null;
    private procGesture: { name: "nod" | "shake"; t0: number } | null = null;
    private qTmp = new THREE.Quaternion();
    private eTmp = new THREE.Euler();

    private frames = 0;
    private fpsWindowStart = 0;
    readonly stats: AvatarStats = { fps: 0, targetFps: 30, drawCalls: 0, triangles: 0, textures: 0, geometries: 0 };

    /** Throws if WebGL is unavailable. */
    constructor(container: HTMLElement) {
        this.container = container;
        this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "low-power" });
        this.renderer.info.autoReset = false;
        const s = this.renderer.domElement.style;
        s.position = "absolute"; s.inset = "0"; s.width = "100%"; s.height = "100%";
        container.appendChild(this.renderer.domElement);

        this.scene.add(this.lookTarget);
        this.scene.add(this.backdrop.mesh);
        this.setLighting(this.lighting);
        this.setQuality(this.quality);

        this.ro = new ResizeObserver(this.resize);
        this.ro.observe(container);
        this.resize();

        window.addEventListener("pointermove", this.onPointerMove);
        this.renderer.domElement.addEventListener("pointerdown", this.react);
        this.audio.addEventListener("ended", this.endSpeech);

        this.timer.connect(document);
        this.io = new IntersectionObserver(([e]) => { this.onScreen = e.isIntersecting; this.updateLoop(); });
        this.io.observe(container);
        document.addEventListener("visibilitychange", this.updateLoop);
        this.updateLoop();
    }

    async load(url: string) {
        const loader = new GLTFLoader();
        loader.register(parser => new VRMLoaderPlugin(parser));
        const gltf = await loader.loadAsync(url);
        const vrm = gltf.userData.vrm as VRM;
        VRMUtils.removeUnnecessaryVertices(gltf.scene);
        VRMUtils.combineSkeletons(gltf.scene);
        VRMUtils.rotateVRM0(vrm);
        this.axis = vrm.meta.metaVersion === "0" ? -1 : 1;
        vrm.scene.traverse(o => { o.frustumCulled = false; });
        this.scene.add(vrm.scene);
        if (vrm.lookAt) vrm.lookAt.target = this.lookTarget;
        this.vrm = vrm;

        applyMToonLook(vrm, LOOKS[this.lighting]);
        // Emotions must not mute the mouth: visemes are scaled down instead (see animate()).
        for (const name of EMOTION_KEYS) {
            const ex = vrm.expressionManager?.getExpression(name);
            if (ex) ex.overrideMouth = "none";
        }
        if (vrm.lookAt) {
            const proxy = new VRMLookAtQuaternionProxy(vrm.lookAt);
            proxy.name = "VRMLookAtQuaternionProxy";
            vrm.scene.add(proxy);
        }
        this.mixer = new THREE.AnimationMixer(vrm.scene);
        this.clips = gestureClips(vrm, this.axis);
        loadVRMAClip(IDLE_VRMA, vrm).then(clip => {
            if (this.vrm !== vrm || !this.mixer) return;
            this.baseAction = this.mixer.clipAction(clip);
            this.baseAction.play();
        }).catch(err => console.warn("[VRM] idle motion unavailable, using procedural idle:", err));
        const head = vrm.humanoid.getNormalizedBoneNode("head");
        if (head) {
            vrm.scene.updateMatrixWorld(true);
            head.getWorldPosition(this.face);
            this.face.y += 0.06;
            this.placeLights();
            this.resize();
        }
    }

    // ---- public controls ----

    setState(state: AvatarState) { this.state = state; }
    getState() { return this.state; }

    react = () => { this.reactUntil = this.timer.getElapsed() + 1.6; };

    /** Base emotion; while speaking, the sentence's own emotion (if any) wins. */
    setEmotion(emotion: Emotion) { this.emotion = emotion; }
    getEmotion() { return this.emotion; }
    get motionSource(): "vrma" | "procedural" { return this.baseAction ? "vrma" : "procedural"; }

    /** One-shot gesture; ignored until the model is loaded. */
    gesture(name: Gesture) {
        if (!this.vrm || !this.mixer) return;
        const t = this.timer.getElapsed();
        if (name === "nod" || name === "shake") { this.procGesture = { name, t0: t }; return; }
        const clip = this.clips[name];
        if (!clip) return;
        this.gestureAction?.stop();
        const action = this.mixer.clipAction(clip);
        action.reset().setLoop(THREE.LoopOnce, 1);
        action.fadeIn(GESTURE_FADE).play();
        this.baseAction?.fadeOut(GESTURE_FADE);
        this.gestureAction = action;
    }

    /** Play audio with a timed viseme track; resolves when playback ends or is interrupted. */
    speak(audioUrl: string, cues: VisemeCue[], onEnd?: () => void, emotion?: Emotion) {
        this.stopSpeech();
        this.speechEmotion = emotion ?? null;
        this.cues = cues;
        this.onSpeechEnd = onEnd ?? null;
        this.audio.src = audioUrl;
        this.clock = () => (!this.audio.paused && !this.audio.ended ? this.audio.currentTime : null);
        this.audio.play().catch(e => console.warn("[TTS] autoplay blocked:", e));
    }

    /** Drive the mouth from a viseme track on a silent clock (no audio). */
    speakSilent(cues: VisemeCue[], duration: number, onEnd?: () => void, emotion?: Emotion) {
        this.stopSpeech();
        this.speechEmotion = emotion ?? null;
        this.cues = cues;
        this.onSpeechEnd = onEnd ?? null;
        const t0 = performance.now();
        this.clock = () => {
            const t = (performance.now() - t0) / 1000;
            if (t > duration) { queueMicrotask(this.endSpeech); return null; }
            return t;
        };
    }

    stopSpeech() {
        this.audio.pause();
        this.audio.removeAttribute("src");
        this.cues = [];
        this.clock = null;
        this.onSpeechEnd = null;
        this.speechEmotion = null;
    }

    isSpeaking() { return this.clock !== null; }

    get expressionNames(): string[] {
        return this.vrm?.expressionManager?.expressions.map(e => e.expressionName) ?? [];
    }

    /** Pin an expression weight (debug); null releases it. */
    setExpressionOverride(name: string, value: number | null) {
        if (value !== null) { this.overrides.set(name, value); return; }
        this.overrides.delete(name);
        this.vrm?.expressionManager?.setValue(name, 0);
    }

    getLighting() { return this.lighting; }
    getQuality() { return this.quality; }
    getFraming() { return this.framing; }

    setLighting(preset: LightingPreset) {
        this.lighting = preset;
        const look = LOOKS[preset];
        for (const l of this.lights) { this.scene.remove(l); l.dispose(); }
        this.lights = [new THREE.HemisphereLight(look.hemi.sky, look.hemi.ground, look.hemi.intensity)];
        for (const spec of look.lights) {
            const l = new THREE.DirectionalLight(spec.color, spec.intensity);
            l.userData.offset = new THREE.Vector3(...spec.pos);
            this.lights.push(l);
        }
        for (const l of this.lights) this.scene.add(l);
        this.placeLights();
        this.renderer.toneMapping = look.toneMapping;
        this.renderer.toneMappingExposure = look.exposure;
        this.backdrop.set(look.backdrop);
        if (this.vrm) applyMToonLook(this.vrm, look);
        this.updateBloom();
    }

    /**
     * balanced (default, tablet-friendly): direct render, <=1.5x pixel ratio, 30 fps unless speaking.
     * high: HalfFloat MSAA target + bloom, <=2x pixel ratio, 60 fps always.
     */
    setQuality(quality: Quality) {
        this.quality = quality;
        this.disposeComposer();
        if (quality === "high") {
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            const rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
            const composer = new EffectComposer(this.renderer, rt);
            composer.addPass(new RenderPass(this.scene, this.camera));
            this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0, 0, 1);
            composer.addPass(this.bloom);
            composer.addPass(new OutputPass());
            this.composer = composer;
            this.updateBloom();
        } else {
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
        }
        this.resize();
    }

    setFraming(framing: Framing) { this.framing = framing; this.resize(); }

    dispose() {
        this.renderer.setAnimationLoop(null);
        this.io.disconnect();
        document.removeEventListener("visibilitychange", this.updateLoop);
        this.stopSpeech();
        this.audio.removeEventListener("ended", this.endSpeech);
        this.timer.dispose();
        this.mixer?.stopAllAction();
        this.disposeComposer();
        this.backdrop.mesh.geometry.dispose();
        (this.backdrop.mesh.material as THREE.Material).dispose();
        this.ro.disconnect();
        window.removeEventListener("pointermove", this.onPointerMove);
        this.renderer.domElement.removeEventListener("pointerdown", this.react);
        if (this.vrm) VRMUtils.deepDispose(this.vrm.scene);
        this.renderer.dispose();
        this.renderer.domElement.remove();
    }

    // ---- internals ----

    private endSpeech = () => {
        const cb = this.onSpeechEnd;
        this.cues = [];
        this.clock = null;
        this.onSpeechEnd = null;
        cb?.();
    };

    /** EffectComposer.dispose() only frees its own targets, not the passes' (bloom mips, materials). */
    private disposeComposer() {
        if (!this.composer) return;
        for (const pass of this.composer.passes) pass.dispose();
        this.composer.dispose();
        this.composer = null;
        this.bloom = null;
    }

    /** Hand back to the idle loop shortly before a one-shot clip ends. */
    private updateGesture() {
        const a = this.gestureAction;
        if (!a) return;
        const left = a.getClip().duration - a.time;
        if (left <= GESTURE_FADE || !a.isRunning()) {
            if (this.baseAction) { this.baseAction.enabled = true; this.baseAction.fadeIn(GESTURE_FADE); }
            a.fadeOut(Math.max(left, 0.01));
            this.fadingGesture = a;
            this.gestureAction = null;
        }
    }

    /** Render loop only runs while the tab is visible and the canvas is on screen. */
    private updateLoop = () => {
        const run = this.onScreen && !document.hidden;
        this.renderer.setAnimationLoop(run ? this.tick : null);
    };

    private updateBloom() {
        const b = LOOKS[this.lighting].bloom;
        if (!this.bloom) return;
        this.bloom.enabled = !!b;
        if (b) { this.bloom.strength = b.strength; this.bloom.radius = b.radius; this.bloom.threshold = b.threshold; }
    }

    private placeLights() {
        for (const l of this.lights) {
            const off = l.userData.offset as THREE.Vector3 | undefined;
            if (!off || !(l instanceof THREE.DirectionalLight)) continue;
            l.position.copy(this.face).add(off);
            l.target.position.copy(this.face);
            l.target.updateMatrixWorld();
        }
    }

    private resolveFraming(w: number, h: number): Exclude<Framing, "auto"> {
        if (this.framing !== "auto") return this.framing;
        if (h < 160) return "face";
        return w / h >= 0.6 ? "bust" : "upper";
    }

    private resize = () => {
        const w = this.container.clientWidth, h = this.container.clientHeight;
        if (!w || !h) return;
        this.renderer.setSize(w, h, false);
        this.composer?.setPixelRatio(this.renderer.getPixelRatio());
        this.composer?.setSize(w, h);
        const cam = this.camera;
        cam.aspect = w / h;
        cam.updateProjectionMatrix();

        const f = FRAMES[this.resolveFraming(w, h)];
        const tanV = Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));
        const viewH = Math.max(f.viewH, f.viewW / cam.aspect);
        this.viewH = viewH;
        const centerY = this.face.y - (0.5 - f.faceFromTop) * viewH;
        this.camAim.set(this.face.x, centerY, this.face.z);
        this.camBase.set(this.face.x, centerY, this.face.z + viewH / 2 / tanV);

        // Backdrop fills the frustum 1.2 m behind the avatar.
        const depth = this.camBase.z - (this.face.z - 1.2);
        const bh = 2 * depth * tanV * 1.15;
        this.backdrop.mesh.scale.set(bh * cam.aspect, bh, 1);
        this.backdrop.uniforms.uAspect.value = cam.aspect;
        this.backdrop.mesh.position.set(this.face.x, centerY, this.face.z - 1.2);
        this.updateCamera(this.timer.getElapsed());
    };

    /** Slow, sub-centimetre drift so the shot breathes without drawing attention. */
    private updateCamera(t: number) {
        const v = this.viewH;
        this.camera.position.set(
            this.camBase.x + Math.sin(t * 0.23) * 0.015 * v,
            this.camBase.y + Math.sin(t * 0.31 + 1) * 0.008 * v,
            this.camBase.z + Math.sin(t * 0.17 + 2) * 0.025 * v,
        );
        this.camera.lookAt(this.camAim);
    }

    private onPointerMove = (e: PointerEvent) => {
        const r = this.container.getBoundingClientRect();
        this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -(((e.clientY - r.top) / r.height) * 2 - 1));
    };

    private tick = (now: number) => {
        const target = this.quality === "high" || this.state === "speaking" || this.clock ? 60 : 30;
        this.stats.targetFps = target;
        // rAF runs at display rate; skip frames to hit the target (small slack for vsync jitter).
        if (target < 60 && now - this.lastFrame < 1000 / target - 4) return;
        this.lastFrame = now;
        this.timer.update(now);
        const dt = Math.min(this.timer.getDelta(), 0.1);
        const t = this.timer.getElapsed();
        this.renderer.info.reset();
        this.updateCamera(t);
        this.backdrop.uniforms.uTime.value = t;
        if (this.vrm) this.animate(this.vrm, t, dt);
        if (this.composer) this.composer.render(dt);
        else this.renderer.render(this.scene, this.camera);
        this.sampleStats(t);
    };

    private animate(vrm: VRM, t: number, dt: number) {
        const emotion = (this.clock ? this.speechEmotion : null) ?? this.emotion;
        const target = POSES[this.state], ep = EMOTION_POSES[emotion];
        const k = Math.min(1, dt * 4);
        for (const key of Object.keys(this.pose) as (keyof Pose)[]) {
            const goal = target[key] + (key in ep ? ep[key as keyof typeof ep] : 0);
            this.pose[key] += (goal - this.pose[key]) * k;
        }
        const pose = this.pose;

        const bone = (name: VRMHumanBoneName) => vrm.humanoid.getNormalizedBoneNode(name);
        const reacting = t < this.reactUntil;
        const talk = this.mouth.aa + this.mouth.oh + this.mouth.ee * 0.6 + this.mouth.ih * 0.4 + this.mouth.ou * 0.5;
        const s = this.axis;
        const neck = bone("neck"), chest = bone("chest") ?? bone("spine"), spine = bone("spine");

        // Procedural offsets are layered on top of whatever the mixer wrote this frame.
        for (const n of [neck, chest, spine]) n?.quaternion.identity();
        this.updateGesture();
        this.mixer?.update(dt);
        const add = (node: THREE.Object3D | null, x: number, y: number, z: number) => {
            node?.quaternion.multiply(this.qTmp.setFromEuler(this.eTmp.set(x * s, y, z * s)));
        };

        const breath = Math.sin(t * 1.6);
        if (this.fadingGesture && !(this.fadingGesture.isRunning() && this.fadingGesture.getEffectiveWeight() > 0)) {
            this.fadingGesture = null;
        }
        // Without the VRMA base loop, let a gesture finish fading before procedural arms take over.
        if (!this.baseAction && !this.gestureAction && !this.fadingGesture) {
            const lu = bone("leftUpperArm"), ru = bone("rightUpperArm");
            const ll = bone("leftLowerArm"), rl = bone("rightLowerArm");
            if (lu) lu.rotation.set(0, 0, s * (-1.2 - breath * 0.02));
            if (ru) ru.rotation.set(0, 0, s * (1.2 + breath * 0.02));
            if (ll) ll.rotation.set(0, -0.25, 0);
            if (rl) rl.rotation.set(0, 0.25, 0);
        }
        add(chest, breath * 0.015 + pose.lean, 0, 0);
        if (spine !== chest) add(spine, 0, Math.sin(t * 0.5) * 0.04, 0);

        let nod = reacting ? Math.sin((this.reactUntil - t) * 7) * 0.08 : Math.sin(t * 0.7) * 0.02;
        let shake = 0;
        if (this.procGesture) {
            const p = (t - this.procGesture.t0) / PROC_GESTURE_SECONDS;
            if (p >= 1) this.procGesture = null;
            else {
                const wave = Math.sin(p * Math.PI * 4) * Math.sin(p * Math.PI);
                if (this.procGesture.name === "nod") nod += wave * 0.14; else shake = wave * 0.22;
            }
        }
        add(neck, nod + pose.headPitch + talk * 0.03, shake, Math.sin(t * 0.43) * 0.04 + pose.headTilt);

        const cam = this.camera.position;
        this.lookTarget.position.set(
            cam.x + this.pointer.x * 0.6 * pose.pointerFollow + pose.lookX,
            cam.y + this.pointer.y * 0.4 * pose.pointerFollow + pose.lookY,
            cam.z,
        );

        const em = vrm.expressionManager;
        if (em) {
            if (this.blinkStart < 0 && t > this.nextBlink) this.blinkStart = t;
            let blink = 0;
            if (this.blinkStart >= 0) {
                const p = (t - this.blinkStart) / 0.16;
                blink = p < 1 ? Math.sin(p * Math.PI) : 0;
                if (p >= 1) { this.blinkStart = -1; this.nextBlink = t + 2 + Math.random() * 4; }
            }
            this.happy += ((reacting ? 1 : 0) - this.happy) * Math.min(1, dt * 8);
            const goal = EMOTION_EXPRESSIONS[emotion];
            const ke = Math.min(1, dt * 6);
            let emoTotal = 0;
            for (const name of EMOTION_KEYS) {
                this.emo[name] += ((goal[name] ?? 0) * this.emotionWeight - this.emo[name]) * ke;
                const w = name === "happy" ? Math.max(this.emo.happy, this.happy * 0.8) : this.emo[name];
                em.setValue(name, w);
                emoTotal += w;
            }
            // Smiling/relaxed already narrows the eyes; a full blink on top looks like a squint.
            const eyesNarrowed = Math.min(1, Math.max(this.emo.happy, this.happy * 0.8) + this.emo.relaxed);
            em.setValue("blink", blink * (1 - eyesNarrowed * 0.85));
            const mouthScale = 1 - 0.3 * Math.min(1, emoTotal);

            const at = this.clock?.() ?? null;
            const targets = at !== null ? visemeTargets(this.cues, at) : null;
            for (const v of VISEMES) {
                this.mouth[v] += ((targets ? targets[v] : 0) - this.mouth[v]) * Math.min(1, dt * 30);
                em.setValue(v, this.mouth[v] * mouthScale);
            }
            for (const [name, value] of this.overrides) em.setValue(name, value);
        }
        vrm.update(dt);
    }

    private sampleStats(t: number) {
        this.frames++;
        if (t - this.fpsWindowStart >= 0.5) {
            this.stats.fps = Math.round(this.frames / (t - this.fpsWindowStart));
            this.frames = 0;
            this.fpsWindowStart = t;
        }
        const info = this.renderer.info;
        this.stats.drawCalls = info.render.calls;
        this.stats.triangles = info.render.triangles;
        this.stats.textures = info.memory.textures;
        this.stats.geometries = info.memory.geometries;
    }
}
