import { useEffect, useReducer, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { AVATAR_STATES, FRAMINGS, LIGHTING_PRESETS, QUALITIES, type AvatarEngine, type AvatarState, type Framing, type LightingPreset, type Quality } from "./AvatarEngine";
import { syntheticVisemes } from "./lipsync";

const API_BASE = "http://127.0.0.1:8000";
const TEST_SENTENCE = "你好，我是你的私人助理。今天想聊点什么呢？我会认真听你说的每一句话。";

/** Dev-only overlay for exercising the avatar without the LLM. */
export function DebugPanel({ engine, onClose }: { engine: AvatarEngine; onClose?: () => void }) {
    const [open, setOpen] = useState(true);
    const [, refresh] = useReducer(x => x + 1, 0);
    const [overrides, setOverrides] = useState<Record<string, number>>({});
    const [lighting, setLighting] = useState<LightingPreset>(engine.getLighting());
    const [quality, setQuality] = useState<Quality>(engine.getQuality());
    const [framing, setFraming] = useState<Framing>(engine.getFraming());
    const [ttsBusy, setTtsBusy] = useState(false);

    useEffect(() => {
        const id = setInterval(refresh, 500);
        return () => clearInterval(id);
    }, []);

    const setOverride = (name: string, value: number | null) => {
        engine.setExpressionOverride(name, value);
        setOverrides(prev => {
            const next = { ...prev };
            if (value === null) delete next[name]; else next[name] = value;
            return next;
        });
    };

    const ttsUrl = useRef<string | null>(null);
    const releaseTts = () => {
        if (ttsUrl.current) URL.revokeObjectURL(ttsUrl.current);
        ttsUrl.current = null;
    };
    useEffect(() => releaseTts, []);

    const stop = () => { engine.stopSpeech(); releaseTts(); engine.setState("idle"); };

    const playTts = async () => {
        setTtsBusy(true);
        try {
            const res = await fetch(`${API_BASE}/api/speech`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: TEST_SENTENCE }),
            });
            const { audio, mime, visemes } = await res.json();
            const bytes = Uint8Array.from(atob(audio), c => c.charCodeAt(0));
            releaseTts();
            const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
            ttsUrl.current = url;
            engine.setState("speaking");
            engine.speak(url, visemes, () => { releaseTts(); engine.setState("idle"); });
        } catch (e) {
            console.warn("[Debug] TTS failed:", e);
        } finally {
            setTtsBusy(false);
        }
    };

    const s = engine.stats;
    const btn = "avatar-debug-button";
    const active = "is-active";

    return createPortal(
        <div className="avatar-debug-panel" role="region" aria-label="角色调试面板"
            data-testid="avatar-debug-panel">
            <div className="avatar-debug-header">
                <button className="avatar-debug-heading" aria-expanded={open} onClick={() => setOpen(!open)}>
                    <span>AVATAR DEBUG</span>
                    <span>{s.fps}/{s.targetFps} FPS · {s.drawCalls} DC {open ? "▾" : "▸"}</span>
                </button>
                {onClose && <button className="avatar-debug-close" aria-label="关闭角色调试面板" onClick={onClose}>×</button>}
            </div>
            {open && (
                <div className="avatar-debug-body">
                    <div className="avatar-debug-muted">tris {s.triangles.toLocaleString()} · tex {s.textures} · geo {s.geometries}</div>

                    <section>
                        <div className="avatar-debug-label">STATE</div>
                        <div className="avatar-debug-actions">
                            {AVATAR_STATES.map((st: AvatarState) => (
                                <button key={st} className={`${btn} ${engine.getState() === st ? active : ""}`} onClick={() => engine.setState(st)}>{st}</button>
                            ))}
                            <button className={btn} onClick={engine.react}>react</button>
                        </div>
                    </section>

                    <section>
                        <div className="avatar-debug-label">LIP-SYNC</div>
                        <div className="avatar-debug-actions">
                            <button className={btn} onClick={() => {
                                const { cues, duration } = syntheticVisemes();
                                engine.setState("speaking");
                                engine.speakSilent(cues, duration, () => engine.setState("idle"));
                            }}>vowels (no TTS)</button>
                            <button className={btn} disabled={ttsBusy} onClick={playTts}>{ttsBusy ? "…" : "TTS sentence"}</button>
                            <button className={btn} onClick={stop}>stop</button>
                        </div>
                    </section>

                    <section>
                        <div className="avatar-debug-label">LOOK · QUALITY · FRAMING</div>
                        <div className="avatar-debug-selects">
                            <select className="avatar-debug-select" aria-label="调试灯光" value={lighting}
                                onChange={e => { const p = e.target.value as LightingPreset; setLighting(p); engine.setLighting(p); }}>
                                {LIGHTING_PRESETS.map(p => <option key={p} value={p}>{p}</option>)}
                            </select>
                            <select className="avatar-debug-select" aria-label="调试画质" value={quality}
                                onChange={e => { const q = e.target.value as Quality; setQuality(q); engine.setQuality(q); }}>
                                {QUALITIES.map(q => <option key={q} value={q}>{q}</option>)}
                            </select>
                            <select className="avatar-debug-select" aria-label="调试构图" value={framing}
                                onChange={e => { const f = e.target.value as Framing; setFraming(f); engine.setFraming(f); }}>
                                {FRAMINGS.map(f => <option key={f} value={f}>{f}</option>)}
                            </select>
                        </div>
                    </section>

                    <section>
                        <div className="avatar-debug-expression-heading">
                            <span>EXPRESSIONS</span>
                            <button className="avatar-debug-reset" onClick={() => Object.keys(overrides).forEach(n => setOverride(n, null))}>reset</button>
                        </div>
                        {engine.expressionNames.map(name => (
                            <label key={name} className="avatar-debug-expression">
                                <input type="checkbox" checked={name in overrides}
                                    onChange={e => setOverride(name, e.target.checked ? 1 : null)} />
                                <span className="avatar-debug-expression-name">{name}</span>
                                <input type="range" min={0} max={1} step={0.05} className="avatar-debug-slider"
                                    value={overrides[name] ?? 0} disabled={!(name in overrides)}
                                    onChange={e => setOverride(name, Number(e.target.value))} />
                            </label>
                        ))}
                    </section>
                </div>
            )}
        </div>,
        document.body,
    );
}
