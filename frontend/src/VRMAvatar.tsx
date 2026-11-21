import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { AvatarEngine, type AvatarState, type Emotion, type Gesture } from "./avatar/AvatarEngine";
import { DebugPanel } from "./avatar/DebugPanel";
import type { VisemeCue } from "./avatar/lipsync";

export type { Viseme, VisemeCue } from "./avatar/lipsync";
export type { AvatarState, Emotion, Gesture } from "./avatar/AvatarEngine";

// See public/vrm/LICENSE.md for the model's licence
const MODEL_URL = "/vrm/avatar.vrm";
const DEBUG_ENABLED = import.meta.env.DEV || new URLSearchParams(location.search).has("debug");

export interface Speech {
    audioUrl: string;
    visemes: VisemeCue[];
    /** Emotion for this sentence; falls back to the `emotion` prop. */
    emotion?: Emotion;
}

export interface VRMAvatarHandle {
    /** One-shot gesture; ignored until the model has loaded. */
    gesture(name: Gesture): void;
}

interface VRMAvatarProps {
    onLoaded?: () => void;
    speech?: Speech | null;
    onAudioEnd?: () => void;
    state?: AvatarState;
    /** Base emotion when not speaking (and while speaking if the sentence has none). Default neutral. */
    emotion?: Emotion;
}

export const VRMAvatar = forwardRef<VRMAvatarHandle, VRMAvatarProps>(function VRMAvatar(
    { onLoaded, speech, onAudioEnd, state = "idle", emotion = "neutral" }, ref,
) {
    const [error, setError] = useState<string | null>(null);
    const [engine, setEngine] = useState<AvatarEngine | null>(null);
    const [showDebug, setShowDebug] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    const onLoadedRef = useRef(onLoaded);
    useEffect(() => { onLoadedRef.current = onLoaded; }, [onLoaded]);
    const onAudioEndRef = useRef(onAudioEnd);
    useEffect(() => { onAudioEndRef.current = onAudioEnd; }, [onAudioEnd]);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;
        let eng: AvatarEngine;
        try {
            eng = new AvatarEngine(container);
            const q = new URLSearchParams(location.search);
            if (q.get("quality") === "high" || q.get("bloom") === "1") eng.setQuality("high");
            if (DEBUG_ENABLED) (window as unknown as { __avatar?: AvatarEngine }).__avatar = eng;
        } catch (err: any) {
            console.error("[VRM] WebGL init failed:", err);
            setError(String(err?.message || err));
            onLoadedRef.current?.();
            return;
        }
        let disposed = false;
        eng.load(MODEL_URL).then(() => {
            if (disposed) return;
            setEngine(eng);
            onLoadedRef.current?.();
        }).catch(err => {
            console.error("[VRM] Load failed:", err);
            if (!disposed) { setError(String(err?.message || err)); onLoadedRef.current?.(); }
        });
        return () => { disposed = true; setEngine(null); eng.dispose(); };
    }, []);

    useEffect(() => { engine?.setState(state); }, [engine, state]);
    useEffect(() => { engine?.setEmotion(emotion); }, [engine, emotion]);
    useImperativeHandle(ref, () => ({ gesture: name => engine?.gesture(name) }), [engine]);

    useEffect(() => {
        if (!speech) return;
        if (engine) {
            engine.speak(speech.audioUrl, speech.visemes, () => onAudioEndRef.current?.(), speech.emotion);
            return () => engine.stopSpeech();
        }
        if (!error) return;
        // Avatar unavailable: still play the reply audio.
        const audio = new Audio(speech.audioUrl);
        audio.onended = () => onAudioEndRef.current?.();
        audio.play().catch(e => console.warn("[TTS] autoplay blocked:", e));
        return () => { audio.onended = null; audio.pause(); };
    }, [engine, error, speech]);

    return (
        <div ref={containerRef} className="absolute inset-0 w-full h-full overflow-hidden" style={{ zIndex: 5 }}>
            {error && (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-red-500/90 text-white px-6 py-4 rounded-xl shadow-2xl backdrop-blur-md max-w-sm text-center font-mono text-xs z-50">
                    <strong>Avatar Error:</strong><br />{error}
                </div>
            )}
            {DEBUG_ENABLED && engine && (
                showDebug
                    ? <DebugPanel engine={engine} />
                    : <button className="absolute top-10 left-3 z-40 px-1.5 py-0.5 rounded bg-black/60 border border-[#00e5ff]/40 text-[9px] font-mono text-cyan-200"
                        onClick={() => setShowDebug(true)} data-testid="avatar-debug-toggle">DBG</button>
            )}
        </div>
    );
});
