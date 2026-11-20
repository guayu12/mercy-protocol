import { useEffect, useRef, useState } from "react";
import * as PIXI from "pixi.js";
import { Live2DModel } from "pixi-live2d-display";

// Register PIXI to window for the Live2D plugin
(window as any).PIXI = PIXI;

// Shizuku — Live2D official sample model (Cubism 2)
// 21 motions: idle(3), tap_body(3), pinch_in(3), pinch_out(3), shake(3), flick_head(3)
// 4 expressions, 3 hit areas (head, mouth, body)
const MODEL_URL = "/live2d/shizuku.model.json";

// All available motion groups for random cycling
const MOTION_GROUPS = ["idle", "tap_body", "pinch_in", "pinch_out", "shake", "flick_head"];

interface Live2DAvatarProps {
    onLoaded?: () => void;
    audioUrl?: string | null;
    onAudioEnd?: () => void;
}

export function Live2DAvatar({ onLoaded, audioUrl, onAudioEnd }: Live2DAvatarProps) {
    const [error, setError] = useState<string | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const appRef = useRef<PIXI.Application | null>(null);
    const modelRef = useRef<any>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);

    const onLoadedRef = useRef(onLoaded);
    useEffect(() => { onLoadedRef.current = onLoaded; }, [onLoaded]);
    const onAudioEndRef = useRef(onAudioEnd);
    useEffect(() => { onAudioEndRef.current = onAudioEnd; }, [onAudioEnd]);

    // ---- PIXI + Live2D init (once) ----
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        let isMounted = true;
        let app: PIXI.Application | null = null;

        const init = async () => {
            app = new PIXI.Application({
                view: document.createElement("canvas"),
                backgroundAlpha: 0,
                autoStart: true,
                powerPreference: "low-power",
                resizeTo: container,
            });

            const canvas = app.view as HTMLCanvasElement;
            canvas.style.position = "absolute";
            canvas.style.inset = "0";
            canvas.style.width = "100%";
            canvas.style.height = "100%";
            container.appendChild(canvas);
            appRef.current = app;

            try {
                const model = await Live2DModel.from(MODEL_URL, { autoInteract: true });
                if (!isMounted) { model.destroy(); return; }

                modelRef.current = model;
                app.stage.addChild(model as any);

                // ---- Scale to fill the panel ----
                const fit = () => {
                    const cw = container.clientWidth;
                    const ch = container.clientHeight;
                    if (!cw || !ch) return;

                    model.scale.set(1);
                    const nw = model.width;
                    const nh = model.height;

                    const scaleX = (cw * 0.9) / nw;
                    const scaleY = (ch * 0.9) / nh;
                    const scale = Math.min(scaleX, scaleY);

                    model.scale.set(scale);
                    model.x = (cw - nw * scale) / 2;
                    model.y = ch - nh * scale - 10;
                };
                fit();
                window.addEventListener("resize", fit);

                // ---- Interactive: Head → expression + flick_head, Body → tap_body ----
                model.on("hit", (areas: string[]) => {
                    if (areas.includes("head") || areas.includes("mouth")) {
                        model.expression();             // Random expression
                        model.motion("flick_head");     // Head-flick animation
                    }
                    if (areas.includes("body")) {
                        model.motion("tap_body");       // Body-tap gesture
                    }
                });

                // ---- Periodic random motions for liveliness ----
                // Every 8 seconds, trigger a random motion from any group
                let motionTimer: ReturnType<typeof setInterval> | null = null;
                motionTimer = setInterval(() => {
                    if (!modelRef.current) return;
                    const group = MOTION_GROUPS[Math.floor(Math.random() * MOTION_GROUPS.length)];
                    model.motion(group);
                }, 8000);

                // ---- LipSync ticker ----
                app.ticker.add(() => {
                    const m = modelRef.current;
                    const analyser = analyserRef.current;
                    const audio = audioRef.current;
                    if (!m || !analyser || !audio || audio.paused) return;

                    const buf = new Uint8Array(analyser.frequencyBinCount);
                    analyser.getByteFrequencyData(buf);
                    let sum = 0;
                    for (let i = 0; i < buf.length; i++) sum += buf[i];
                    const avg = sum / buf.length;
                    const mouth = Math.min(1.0, (avg / 50) * 1.5);

                    try {
                        const core = m.internalModel?.coreModel;
                        if (core?.setParamFloat) {
                            // Cubism 2 uses setParamFloat
                            core.setParamFloat("PARAM_MOUTH_OPEN_Y", mouth);
                        } else if (core?.setParameterValueById) {
                            // Cubism 3/4 uses setParameterValueById
                            core.setParameterValueById("ParamMouthOpenY", mouth);
                        }
                    } catch (_) { }
                });

                if (onLoadedRef.current) onLoadedRef.current();

                return () => {
                    window.removeEventListener("resize", fit);
                    if (motionTimer) clearInterval(motionTimer);
                };
            } catch (err: any) {
                console.error("[Live2D] Load failed:", err);
                if (isMounted) {
                    setError(String(err?.message || err));
                    if (onLoadedRef.current) onLoadedRef.current();
                }
            }
        };

        let extraCleanup: (() => void) | undefined;
        init().then(fn => { extraCleanup = fn; });

        return () => {
            isMounted = false;
            extraCleanup?.();
            if (app) { try { app.destroy(true, { children: true }); } catch (_) { } }
            appRef.current = null;
            modelRef.current = null;
        };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    // ---- Audio playback for TTS ----
    useEffect(() => {
        if (!audioUrl) return;

        if (!audioRef.current) {
            const audio = new Audio();
            audio.crossOrigin = "anonymous";
            audioRef.current = audio;

            const Ctx = window.AudioContext || (window as any).webkitAudioContext;
            const ctx = new Ctx();
            audioContextRef.current = ctx;
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 256;
            analyserRef.current = analyser;

            const src = ctx.createMediaElementSource(audio);
            src.connect(analyser);
            analyser.connect(ctx.destination);
        }

        audioRef.current.onended = () => { onAudioEndRef.current?.(); };

        if (audioContextRef.current?.state === "suspended") {
            audioContextRef.current.resume();
        }

        audioRef.current.src = audioUrl;
        audioRef.current.play().catch(e => console.warn("[TTS] autoplay blocked:", e));

        return () => {
            if (audioRef.current) { audioRef.current.pause(); audioRef.current.src = ""; }
        };
    }, [audioUrl]);

    return (
        <div
            ref={containerRef}
            className="absolute inset-0 w-full h-full overflow-hidden"
            style={{ zIndex: 5 }}
        >
            {error && (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-red-500/90 text-white px-6 py-4 rounded-xl shadow-2xl backdrop-blur-md max-w-sm text-center font-mono text-xs z-50">
                    <strong>Live2D Error:</strong><br />{error}
                </div>
            )}
        </div>
    );
}
