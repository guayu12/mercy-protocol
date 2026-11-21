import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRM, VRMLoaderPlugin, VRMUtils, type VRMHumanBoneName } from "@pixiv/three-vrm";
import { VISEMES, visemeTargets, type Viseme, type VisemeCue } from "./lipsync";

export type AvatarState = "idle" | "listening" | "thinking" | "speaking";
export const AVATAR_STATES: AvatarState[] = ["idle", "listening", "thinking", "speaking"];

export type LightingPreset = "basic";
export const LIGHTING_PRESETS: LightingPreset[] = ["basic"];

export interface AvatarStats { fps: number; drawCalls: number; triangles: number; textures: number; geometries: number }

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
    readonly camera = new THREE.PerspectiveCamera(28, 1, 0.1, 20);

    private container: HTMLElement;
    private timer = new THREE.Timer();
    private ro: ResizeObserver;
    private focus = new THREE.Vector3(0, 1.35, 0);
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

    private frames = 0;
    private fpsWindowStart = 0;
    readonly stats: AvatarStats = { fps: 0, drawCalls: 0, triangles: 0, textures: 0, geometries: 0 };

    /** Throws if WebGL is unavailable. */
    constructor(container: HTMLElement) {
        this.container = container;
        this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "low-power" });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        const s = this.renderer.domElement.style;
        s.position = "absolute"; s.inset = "0"; s.width = "100%"; s.height = "100%";
        container.appendChild(this.renderer.domElement);

        this.scene.add(this.lookTarget);
        this.setLighting("basic");

        this.ro = new ResizeObserver(this.resize);
        this.ro.observe(container);
        this.resize();

        window.addEventListener("pointermove", this.onPointerMove);
        this.renderer.domElement.addEventListener("pointerdown", this.react);
        this.audio.addEventListener("ended", this.endSpeech);

        this.timer.connect(document);
        this.renderer.setAnimationLoop(this.tick);
    }

    async load(url: string) {
        const loader = new GLTFLoader();
        loader.register(parser => new VRMLoaderPlugin(parser));
        const gltf = await loader.loadAsync(url);
        const vrm = gltf.userData.vrm as VRM;
        VRMUtils.removeUnnecessaryVertices(gltf.scene);
        VRMUtils.combineSkeletons(gltf.scene);
        VRMUtils.rotateVRM0(vrm);
        vrm.scene.traverse(o => { o.frustumCulled = false; });
        this.scene.add(vrm.scene);
        if (vrm.lookAt) vrm.lookAt.target = this.lookTarget;
        this.vrm = vrm;

        const head = vrm.humanoid.getNormalizedBoneNode("head");
        if (head) {
            vrm.scene.updateMatrixWorld(true);
            head.getWorldPosition(this.focus);
            this.focus.y -= 0.12;
            this.resize();
        }
    }

    // ---- public controls ----

    setState(state: AvatarState) { this.state = state; }
    getState() { return this.state; }

    react = () => { this.reactUntil = this.timer.getElapsed() + 1.6; };

    /** Play audio with a timed viseme track; resolves when playback ends or is interrupted. */
    speak(audioUrl: string, cues: VisemeCue[], onEnd?: () => void) {
        this.stopSpeech();
        this.cues = cues;
        this.onSpeechEnd = onEnd ?? null;
        this.audio.src = audioUrl;
        this.clock = () => (!this.audio.paused && !this.audio.ended ? this.audio.currentTime : null);
        this.audio.play().catch(e => console.warn("[TTS] autoplay blocked:", e));
    }

    /** Drive the mouth from a viseme track on a silent clock (no audio). */
    speakSilent(cues: VisemeCue[], duration: number, onEnd?: () => void) {
        this.stopSpeech();
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

    setLighting(_preset: LightingPreset) {
        for (const l of this.lights) this.scene.remove(l);
        const ambient = new THREE.AmbientLight(0xffffff, 0.6);
        const key = new THREE.DirectionalLight(0xffffff, 1.2);
        key.position.set(0.6, 1.5, 2);
        this.lights = [ambient, key];
        for (const l of this.lights) this.scene.add(l);
    }

    dispose() {
        this.renderer.setAnimationLoop(null);
        this.stopSpeech();
        this.audio.removeEventListener("ended", this.endSpeech);
        this.timer.dispose();
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

    private resize = () => {
        const w = this.container.clientWidth, h = this.container.clientHeight;
        if (!w || !h) return;
        this.renderer.setSize(w, h, false);
        const cam = this.camera;
        cam.aspect = w / h;
        const viewHeight = 1.05; // metres of body visible vertically
        const dist = viewHeight / 2 / Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));
        cam.position.set(this.focus.x, this.focus.y + 0.02, this.focus.z + Math.max(dist, 0.8 / cam.aspect));
        cam.lookAt(this.focus);
        cam.updateProjectionMatrix();
    };

    private onPointerMove = (e: PointerEvent) => {
        const r = this.container.getBoundingClientRect();
        this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -(((e.clientY - r.top) / r.height) * 2 - 1));
    };

    private tick = () => {
        this.timer.update();
        const dt = Math.min(this.timer.getDelta(), 0.1);
        const t = this.timer.getElapsed();
        if (this.vrm) this.animate(this.vrm, t, dt);
        this.renderer.render(this.scene, this.camera);
        this.sampleStats(t);
    };

    private animate(vrm: VRM, t: number, dt: number) {
        const target = POSES[this.state];
        const k = Math.min(1, dt * 4);
        for (const key of Object.keys(this.pose) as (keyof Pose)[]) this.pose[key] += (target[key] - this.pose[key]) * k;
        const pose = this.pose;

        const bone = (name: VRMHumanBoneName) => vrm.humanoid.getNormalizedBoneNode(name);
        const reacting = t < this.reactUntil;
        const talk = this.mouth.aa + this.mouth.oh + this.mouth.ee * 0.6 + this.mouth.ih * 0.4 + this.mouth.ou * 0.5;

        const breath = Math.sin(t * 1.6);
        const lu = bone("leftUpperArm"), ru = bone("rightUpperArm");
        const ll = bone("leftLowerArm"), rl = bone("rightLowerArm");
        if (lu) lu.rotation.set(0, 0, -1.2 - breath * 0.02);
        if (ru) ru.rotation.set(0, 0, 1.2 + breath * 0.02);
        if (ll) ll.rotation.set(0, -0.25, 0);
        if (rl) rl.rotation.set(0, 0.25, 0);
        const chest = bone("chest") ?? bone("spine");
        if (chest) chest.rotation.x = breath * 0.015 + pose.lean;
        const spine = bone("spine");
        if (spine) spine.rotation.y = Math.sin(t * 0.5) * 0.04;

        const neck = bone("neck");
        if (neck) {
            const nod = reacting ? Math.sin((this.reactUntil - t) * 7) * 0.08 : Math.sin(t * 0.7) * 0.02;
            neck.rotation.x = nod + pose.headPitch + talk * 0.03;
            neck.rotation.z = Math.sin(t * 0.43) * 0.04 + pose.headTilt;
        }

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
            em.setValue("happy", this.happy * 0.8);
            em.setValue("blink", Math.max(blink, this.happy * 0.6));

            const at = this.clock?.() ?? null;
            const targets = at !== null ? visemeTargets(this.cues, at) : null;
            for (const v of VISEMES) {
                this.mouth[v] += ((targets ? targets[v] : 0) - this.mouth[v]) * Math.min(1, dt * 30);
                em.setValue(v, this.mouth[v]);
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
