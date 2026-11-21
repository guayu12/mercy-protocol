// Dev-only harness: /avatar-lab.html?model=/vrm/avatar.vrm&w=360&h=340&look=studio&quality=high&framing=auto
import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "../App.css";
import { AvatarEngine, type AvatarState, type Framing, type LightingPreset, type Quality } from "./AvatarEngine";
import { DebugPanel } from "./DebugPanel";

const q = new URLSearchParams(location.search);

function Lab() {
    const ref = useRef<HTMLDivElement>(null);
    const [engine, setEngine] = useState<AvatarEngine | null>(null);
    useEffect(() => {
        const eng = new AvatarEngine(ref.current!);
        (window as unknown as { __avatar?: AvatarEngine }).__avatar = eng;
        if (q.get("look")) eng.setLighting(q.get("look") as LightingPreset);
        if (q.get("quality")) eng.setQuality(q.get("quality") as Quality);
        if (q.get("framing")) eng.setFraming(q.get("framing") as Framing);
        if (q.get("state")) eng.setState(q.get("state") as AvatarState);
        eng.load(q.get("model") ?? "/vrm/avatar.vrm").then(() => {
            document.body.dataset.loaded = "1";
            setEngine(eng);
        });
        return () => eng.dispose();
    }, []);
    return (
        <div style={{ padding: 16 }}>
            <div ref={ref} data-testid="avatar-card"
                style={{ position: "relative", width: +(q.get("w") ?? 360), height: +(q.get("h") ?? 340), borderRadius: 16, overflow: "hidden" }} />
            {engine && !q.has("nopanel") && <DebugPanel engine={engine} />}
        </div>
    );
}

createRoot(document.getElementById("root")!).render(<Lab />);
