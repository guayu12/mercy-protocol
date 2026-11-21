import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRM, VRMLoaderPlugin, VRMUtils, type VRMHumanBoneName } from "@pixiv/three-vrm";

// pixiv VRM 1.0 sample (redistribution, modification and commercial use allowed — see public/vrm/LICENSE.md)
const MODEL_URL = "/vrm/avatar.vrm";

export type Viseme = "aa" | "ih" | "ou" | "ee" | "oh";
const VISEMES: Viseme[] = ["aa", "ih", "ou", "ee", "oh"];

export interface VisemeCue { start: number; end: number; v: Viseme }
export interface Speech { audioUrl: string; visemes: VisemeCue[] }

interface VRMAvatarProps {
    onLoaded?: () => void;
    speech?: Speech | null;
    onAudioEnd?: () => void;
}

const MOUTH_PEAK = 0.85;
const MOUTH_RAMP = 0.06; // seconds to open/close at the edges of each cue

function visemeTargets(cues: VisemeCue[], t: number): Record<Viseme, number> {
    const out: Record<Viseme, number> = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
    for (const c of cues) {
        if (t < c.start - MOUTH_RAMP) break;
        if (t > c.end + MOUTH_RAMP) continue;
        const rampIn = (t - (c.start - MOUTH_RAMP)) / MOUTH_RAMP;
        const rampOut = (c.end + MOUTH_RAMP - t) / MOUTH_RAMP;
        const w = Math.max(0, Math.min(1, rampIn, rampOut)) * MOUTH_PEAK;
        out[c.v] = Math.max(out[c.v], w);
    }
    return out;
}

export function VRMAvatar({ onLoaded, speech, onAudioEnd }: VRMAvatarProps) {
    const [error, setError] = useState<string | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const cuesRef = useRef<VisemeCue[]>([]);

    const onLoadedRef = useRef(onLoaded);
    useEffect(() => { onLoadedRef.current = onLoaded; }, [onLoaded]);
    const onAudioEndRef = useRef(onAudioEnd);
    useEffect(() => { onAudioEndRef.current = onAudioEnd; }, [onAudioEnd]);

    // ---- three.js scene + VRM (once) ----
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;
        let disposed = false;
        let vrm: VRM | null = null;

        let renderer: THREE.WebGLRenderer;
        try {
            renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "low-power" });
        } catch (err: any) {
            console.error("[VRM] WebGL init failed:", err);
            setError(String(err?.message || err));
            onLoadedRef.current?.();
            return;
        }
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.domElement.style.position = "absolute";
        renderer.domElement.style.inset = "0";
        container.appendChild(renderer.domElement);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(28, 1, 0.1, 20);
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
        keyLight.position.set(0.6, 1.5, 2);
        scene.add(keyLight);

        // Upper-body framing around the head; refined once the model is loaded
        const focus = new THREE.Vector3(0, 1.35, 0);
        const resize = () => {
            const w = container.clientWidth, h = container.clientHeight;
            if (!w || !h) return;
            renderer.setSize(w, h, false);
            camera.aspect = w / h;
            const viewHeight = 1.05; // metres of body visible vertically
            const dist = viewHeight / 2 / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
            camera.position.set(focus.x, focus.y + 0.02, focus.z + Math.max(dist, 0.8 / camera.aspect));
            camera.lookAt(focus);
            camera.updateProjectionMatrix();
        };
        const ro = new ResizeObserver(resize);
        ro.observe(container);
        resize();

        // Eyes follow the pointer, softly
        const lookTarget = new THREE.Object3D();
        scene.add(lookTarget);
        const pointer = new THREE.Vector2();
        const onPointerMove = (e: PointerEvent) => {
            const r = container.getBoundingClientRect();
            pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -(((e.clientY - r.top) / r.height) * 2 - 1));
        };
        window.addEventListener("pointermove", onPointerMove);

        // Click → smile + nod
        let reactUntil = 0;
        const onClick = () => { reactUntil = timer.getElapsed() + 1.6; };
        renderer.domElement.addEventListener("pointerdown", onClick);

        const loader = new GLTFLoader();
        loader.register(parser => new VRMLoaderPlugin(parser));
        loader.loadAsync(MODEL_URL).then(gltf => {
            if (disposed) return;
            vrm = gltf.userData.vrm as VRM;
            VRMUtils.removeUnnecessaryVertices(gltf.scene);
            VRMUtils.combineSkeletons(gltf.scene);
            VRMUtils.rotateVRM0(vrm);
            vrm.scene.traverse(o => { o.frustumCulled = false; });
            scene.add(vrm.scene);
            if (vrm.lookAt) vrm.lookAt.target = lookTarget;

            const head = vrm.humanoid.getNormalizedBoneNode("head");
            if (head) {
                vrm.scene.updateMatrixWorld(true);
                head.getWorldPosition(focus);
                focus.y -= 0.12;
                resize();
            }
            onLoadedRef.current?.();
        }).catch(err => {
            console.error("[VRM] Load failed:", err);
            if (!disposed) { setError(String(err?.message || err)); onLoadedRef.current?.(); }
        });

        const timer = new THREE.Timer();
        timer.connect(document);
        const mouth: Record<Viseme, number> = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
        let nextBlink = 2, blinkStart = -1;
        let happy = 0;

        renderer.setAnimationLoop(() => {
            timer.update();
            const dt = Math.min(timer.getDelta(), 0.1);
            const t = timer.getElapsed();
            if (vrm) {
                const humanoid = vrm.humanoid;
                const bone = (name: VRMHumanBoneName) => humanoid.getNormalizedBoneNode(name);

                // Relaxed idle pose with breathing and a gentle sway
                const breath = Math.sin(t * 1.6);
                const lu = bone("leftUpperArm"), ru = bone("rightUpperArm");
                const ll = bone("leftLowerArm"), rl = bone("rightLowerArm");
                if (lu) lu.rotation.set(0, 0, -1.2 - breath * 0.02);
                if (ru) ru.rotation.set(0, 0, 1.2 + breath * 0.02);
                if (ll) ll.rotation.set(0, -0.25, 0);
                if (rl) rl.rotation.set(0, 0.25, 0);
                const chest = bone("chest") ?? bone("spine");
                if (chest) chest.rotation.x = breath * 0.015;
                const spine = bone("spine");
                if (spine) spine.rotation.y = Math.sin(t * 0.5) * 0.04;

                const reacting = t < reactUntil;
                const neck = bone("neck");
                if (neck) {
                    neck.rotation.x = reacting ? Math.sin((reactUntil - t) * 7) * 0.08 : Math.sin(t * 0.7) * 0.02;
                    neck.rotation.z = Math.sin(t * 0.43) * 0.04;
                }

                lookTarget.position.set(
                    camera.position.x + pointer.x * 0.6,
                    camera.position.y + pointer.y * 0.4,
                    camera.position.z,
                );

                const em = vrm.expressionManager;
                if (em) {
                    // Blink
                    if (blinkStart < 0 && t > nextBlink) blinkStart = t;
                    let blink = 0;
                    if (blinkStart >= 0) {
                        const p = (t - blinkStart) / 0.16;
                        blink = p < 1 ? Math.sin(p * Math.PI) : 0;
                        if (p >= 1) { blinkStart = -1; nextBlink = t + 2 + Math.random() * 4; }
                    }
                    happy += ((reacting ? 1 : 0) - happy) * Math.min(1, dt * 8);
                    em.setValue("happy", happy * 0.8);
                    em.setValue("blink", Math.max(blink, happy * 0.6));

                    // Lip-sync from the timed viseme track
                    const audio = audioRef.current;
                    const playing = !!audio && !audio.paused && !audio.ended;
                    const targets = playing ? visemeTargets(cuesRef.current, audio!.currentTime) : null;
                    for (const v of VISEMES) {
                        const target = targets ? targets[v] : 0;
                        mouth[v] += (target - mouth[v]) * Math.min(1, dt * 30);
                        em.setValue(v, mouth[v]);
                    }
                }
                vrm.update(dt);
            }
            renderer.render(scene, camera);
        });

        return () => {
            disposed = true;
            renderer.setAnimationLoop(null);
            timer.dispose();
            ro.disconnect();
            window.removeEventListener("pointermove", onPointerMove);
            renderer.domElement.removeEventListener("pointerdown", onClick);
            if (vrm) VRMUtils.deepDispose(vrm.scene);
            renderer.dispose();
            renderer.domElement.remove();
        };
    }, []);

    // ---- Speech playback ----
    useEffect(() => {
        if (!speech) return;
        const audio = audioRef.current ?? (audioRef.current = new Audio());
        cuesRef.current = speech.visemes;
        audio.onended = () => { onAudioEndRef.current?.(); };
        audio.src = speech.audioUrl;
        audio.play().catch(e => console.warn("[TTS] autoplay blocked:", e));
        return () => { audio.pause(); audio.removeAttribute("src"); cuesRef.current = []; };
    }, [speech]);

    return (
        <div ref={containerRef} className="absolute inset-0 w-full h-full overflow-hidden" style={{ zIndex: 5 }}>
            {error && (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-red-500/90 text-white px-6 py-4 rounded-xl shadow-2xl backdrop-blur-md max-w-sm text-center font-mono text-xs z-50">
                    <strong>Avatar Error:</strong><br />{error}
                </div>
            )}
        </div>
    );
}
