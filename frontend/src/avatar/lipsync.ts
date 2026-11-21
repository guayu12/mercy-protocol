export type Viseme = "aa" | "ih" | "ou" | "ee" | "oh";
export const VISEMES: Viseme[] = ["aa", "ih", "ou", "ee", "oh"];

export interface VisemeCue { start: number; end: number; v: Viseme }

const MOUTH_PEAK = 0.85;
const MOUTH_RAMP = 0.06; // seconds to open/close at the edges of each cue

export function visemeTargets(cues: VisemeCue[], t: number): Record<Viseme, number> {
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

/** A fixed vowel walk-through for testing mouth shapes without TTS. */
export function syntheticVisemes(): { cues: VisemeCue[]; duration: number } {
    const pattern: (Viseme | null)[] = [
        "aa", "aa", null, "ih", "ih", null, "ou", "ou", null, "ee", "ee", null, "oh", "oh", null,
        "aa", "ih", "ou", "ee", "oh", null, "ih", "aa",
    ];
    const step = 0.22;
    const cues: VisemeCue[] = [];
    pattern.forEach((v, i) => {
        if (v) cues.push({ start: 0.3 + i * step, end: 0.3 + i * step + step * 0.9, v });
    });
    return { cues, duration: 0.3 + pattern.length * step + 0.4 };
}
