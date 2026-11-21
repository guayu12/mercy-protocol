import { useState, useEffect, useRef } from "react";
import {
  Mic, Send, Settings, Camera, Image as ImageIcon,
  Loader2, X, Activity, Cpu, Database, Eye, Radio, AlertTriangle
} from "lucide-react";
import "./App.css";
import { VRMAvatar, type Speech } from "./VRMAvatar";

const API_BASE = "http://127.0.0.1:8000";

interface Message {
  role: "system" | "user" | "assistant";
  content: string;
  image?: string;
  id?: number;
}

// ── Real-time log entry from backend SSE ──────────────────────────────
interface LogEntry {
  ts: number;          // ms epoch
  level: string;       // "info" | "warn" | "tool" | "llm" | "flow" | "error"
  category: string;    // "LLM" | "TOOL" | "FLOW" | "ERROR" | ...
  message: string;
  data: Record<string, unknown>;
}

// ── Color map per level ───────────────────────────────────────────────
const LEVEL_COLOR: Record<string, string> = {
  flow: "#00e5ff",    // cyan
  llm: "#00ff88",    // green
  tool: "#ffab00",    // amber
  warn: "#ff9800",    // orange
  error: "#ff3d5a",    // red
  info: "#90a4ae",    // grey
};

// ── Live system metrics (fake, just for the hardware display) ────────
function useLiveMetrics() {
  const [cpu, setCpu] = useState(34);
  const [mem, setMem] = useState(61);
  const [net, setNet] = useState(12);
  useEffect(() => {
    const t = setInterval(() => {
      setCpu(c => Math.min(95, Math.max(15, c + (Math.random() - 0.5) * 10)));
      setMem(m => Math.min(85, Math.max(50, m + (Math.random() - 0.5) * 4)));
      setNet(n => Math.min(99, Math.max(2, n + (Math.random() - 0.5) * 20)));
    }, 1800);
    return () => clearInterval(t);
  }, []);
  return { cpu: Math.round(cpu), mem: Math.round(mem), net: Math.round(net) };
}

// ── Mini metric bar ───────────────────────────────────────────────────
function MetricBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="mb-2">
      <div className="flex justify-between text-[9px] font-mono mb-1" style={{ color }}>
        <span>{label}</span><span>{value}%</span>
      </div>
      <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${value}%`, background: color, boxShadow: `0 0 6px ${color}` }}
        />
      </div>
    </div>
  );
}

// ── Waveform ──────────────────────────────────────────────────────────
function Waveform({ active }: { active: boolean }) {
  const bars = [7, 12, 8];
  return (
    <div className="flex items-center gap-[2px] h-6 flex-none" style={{ width: '16px' }}>
      {bars.map((h, i) => (
        <div
          key={i}
          className="waveform-bar"
          style={{
            height: active ? `${h * 2}px` : '3px',
            animationDelay: `${i * 0.06}s`,
            animationDuration: active ? `${0.5 + (i % 5) * 0.12}s` : '0s',
            opacity: active ? 1 : 0.2,
            transition: 'height 0.3s ease',
          }}
        />
      ))}
    </div>
  );
}

// ── Format timestamp as HH:MM:SS.mmm ─────────────────────────────────
function fmtTs(ms: number): string {
  const d = new Date(ms);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
}

// ── Single log row ────────────────────────────────────────────────────
function LogRow({ entry }: { entry: LogEntry }) {
  const color = LEVEL_COLOR[entry.level] ?? "#90a4ae";
  return (
    <div className="flex gap-1.5 py-[3px] border-b border-white/[0.04] items-start group hover:bg-white/[0.02] transition-colors">
      <span className="flex-none text-[8px] font-mono text-white/20 pt-px">{fmtTs(entry.ts)}</span>
      <span className="flex-none text-[8px] font-mono font-bold px-1 rounded"
        style={{ color, textShadow: `0 0 6px ${color}40` }}>
        {entry.category}
      </span>
      <span className="text-[9px] font-mono leading-relaxed break-words min-w-0"
        style={{ color: entry.level === 'error' ? '#ff3d5a' : 'rgba(176,224,240,0.75)' }}>
        {entry.message}
      </span>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════
export function App() {
  const [query, setQuery] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isAvatarLoaded, setIsAvatarLoaded] = useState(false);
  const [speech, setSpeech] = useState<Speech | null>(null);
  const [currentTime, setCurrentTime] = useState("");
  const [currentDate, setCurrentDate] = useState("");
  // Session id doubles as the LangGraph thread_id; persisted so conversation memory survives reloads
  const [sessionId] = useState(() => {
    const saved = localStorage.getItem("mercy.threadId");
    if (saved) return saved;
    const id = Math.random().toString(36).substring(2, 10).toUpperCase();
    localStorage.setItem("mercy.threadId", id);
    return id;
  });
  const [msgCount, setMsgCount] = useState(0);
  const [messages, setMessages] = useState<Message[]>([
    { role: "system", content: "MERCY PROTOCOL v2.0 — AI JUDICIAL INTERFACE ONLINE. AWAITING SUBJECT INPUT." }
  ]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { cpu, mem, net } = useLiveMetrics();

  // Clock
  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setCurrentTime(now.toLocaleTimeString('en-US', { hour12: false }));
      setCurrentDate(now.toLocaleDateString('en-US', { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' }));
    };
    tick();
    const t = setInterval(tick, 1000);
    return () => clearInterval(t);
  }, []);

  // SSE — real-time backend log stream
  useEffect(() => {
    const es = new EventSource(`${API_BASE}/api/logs`);
    es.onmessage = (e) => {
      try {
        const entry: LogEntry = JSON.parse(e.data);
        setLogs(prev => {
          const next = [...prev, entry];
          return next.length > 300 ? next.slice(-300) : next; // cap at 300 entries
        });
      } catch { }
    };
    es.onerror = () => { };
    return () => es.close();
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, selectedImage]);

  // ASR
  useEffect(() => {
    if (!isListening) return;
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) { setIsListening(false); return; }
    const rec = new SR();
    rec.lang = "zh-CN";
    rec.interimResults = false;
    rec.onresult = (e: any) => {
      setQuery(prev => prev ? prev + " " + e.results[0][0].transcript : e.results[0][0].transcript);
      setIsListening(false);
    };
    rec.onerror = () => setIsListening(false);
    rec.onend = () => setIsListening(false);
    try { rec.start(); } catch { setIsListening(false); }
    return () => { try { rec.stop(); } catch { } };
  }, [isListening]);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      if (typeof ev.target?.result === 'string') setSelectedImage(ev.target.result);
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  const speak = async (text: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/speech`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const { audio, mime, visemes } = await res.json();
      const bytes = Uint8Array.from(atob(audio), c => c.charCodeAt(0));
      const audioUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
      setSpeech(prev => { if (prev) URL.revokeObjectURL(prev.audioUrl); return { audioUrl, visemes }; });
    } catch (e) {
      console.warn("[TTS] speech failed:", e);
    }
  };

  const handleSend = async () => {
    if ((!query.trim() && !selectedImage) || isLoading) return;
    const userMsg = query.trim();
    const img = selectedImage;
    setQuery(""); setSelectedImage(null);
    setMessages(prev => [...prev, { role: "user", content: userMsg, image: img || undefined }]);
    setMsgCount(c => c + 1);
    setIsLoading(true);

    const replyId = Date.now();
    const updateReply = (fn: (content: string) => string) => {
      setMessages(prev => {
        const idx = prev.findIndex(m => m.id === replyId);
        if (idx === -1) return [...prev, { role: "assistant", content: fn(""), id: replyId }];
        const next = [...prev];
        next[idx] = { ...next[idx], content: fn(next[idx].content) };
        return next;
      });
    };
    const appendToken = (token: string) => updateReply(c => c + token);
    const setFinalReply = (reply: string) => updateReply(() => reply);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMsg || "Please analyze this image.",
          image_base64: img || undefined,
          thread_id: sessionId,
        })
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalReply = "";
      for (; ;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sep;
        while ((sep = buffer.indexOf("\n\n")) !== -1) {
          const raw = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          const data = raw.split("\n").filter(l => l.startsWith("data:")).map(l => l.slice(5).trimStart()).join("\n");
          if (!data) continue;
          const evt = JSON.parse(data);
          if (evt.type === "token") appendToken(evt.content);
          else if (evt.type === "done") { finalReply = evt.reply; setFinalReply(evt.reply); }
          else if (evt.type === "error") setMessages(prev => [...prev, { role: "system", content: "COMM_FAIL >> " + evt.message }]);
        }
      }

      if (finalReply) {
        setMsgCount(c => c + 1);
        speak(finalReply);
      }
    } catch {
      setMessages(prev => [...prev, { role: "system", content: "NET_ERR >> BACKEND UNREACHABLE" }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full h-full tactical-grid scanlines vignette overflow-hidden relative flex flex-col"
      style={{ fontFamily: "'Rajdhani', sans-serif" }}>

      {/* ══════════ TOP COMMAND BAR ══════════ */}
      <div
        data-tauri-drag-region
        className="flex-none h-10 flex items-center justify-between px-5 z-50 cursor-move relative"
        style={{ background: 'rgba(0,5,15,0.95)', borderBottom: '1px solid rgba(0,229,255,0.15)' }}
      >
        <div className="flex items-center gap-3 pointer-events-none">
          <div className="w-2 h-2 rounded-full bg-[#00e5ff] pulse-cyan" />
          <span className="text-[10px] font-mono font-bold neon-cyan tracking-[0.25em] flicker">
            MERCY PROTOCOL // JUDICIAL AI v2.0
          </span>
        </div>
        <div className="flex items-center gap-6 pointer-events-none text-[9px] font-mono">
          <span className="text-white/30">SESSION</span>
          <span className="neon-amber tracking-widest">{sessionId}</span>
          <div className="w-px h-4 bg-white/10" />
          <span className="text-white/30">CASE</span>
          <span className="text-white/60 tracking-wider">2026-MERCY-{String(msgCount).padStart(3, '0')}</span>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right pointer-events-none">
            <div className="text-[11px] font-mono font-bold neon-cyan">{currentTime}</div>
            <div className="text-[8px] font-mono text-white/30 tracking-wider">{currentDate}</div>
          </div>
          <button className="btn-holo p-1.5 rounded text-[10px] pointer-events-auto"><Settings size={12} /></button>
        </div>
        <div className="top-bar-line absolute bottom-0 left-0 right-0" />
      </div>

      {/* ══════════ MAIN LAYOUT ══════════ */}
      <div className="flex-1 flex flex-row overflow-hidden p-3 gap-3 min-h-0">

        {/* ── LEFT SIDEBAR ─────────────────────────────────────── */}
        <div className="flex-none w-56 flex flex-col gap-3">

          {/* System Metrics */}
          <div className="holo-panel rounded-lg p-3 corner-cut flex-none">
            <div className="flex items-center gap-2 mb-3">
              <Cpu size={10} className="neon-cyan" />
              <span className="text-[9px] font-mono font-bold neon-cyan tracking-widest">SYS.RESOURCES</span>
            </div>
            <MetricBar label="CPU" value={cpu} color="#00e5ff" />
            <MetricBar label="MEM" value={mem} color="#ffab00" />
            <MetricBar label="NET" value={net} color="#00ff88" />
          </div>

          {/* Real-time Backend Log Panel */}
          <div className="holo-panel rounded-lg flex-1 flex flex-col overflow-hidden corner-cut" style={{ minHeight: 0 }}>
            <div className="flex-none px-3 pt-3 pb-2 flex items-center gap-2 border-b border-white/5">
              <Activity size={10} className="neon-amber" />
              <span className="text-[9px] font-mono font-bold neon-amber tracking-widest">BACKEND.LOG</span>
              <div className="flex-1" />
              <span className="text-[7px] font-mono text-white/20">{logs.length} events</span>
              <div className="w-1.5 h-1.5 rounded-full"
                style={{
                  background: logs.length > 0 ? '#00ff88' : '#ff3d5a',
                  boxShadow: logs.length > 0 ? '0 0 6px #00ff88' : '0 0 6px #ff3d5a'
                }} />
            </div>
            <div className="flex-1 overflow-y-auto mercy-scroll px-2 py-1 min-h-0">
              {logs.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-white/15 text-[8px] font-mono gap-2">
                  <Activity size={16} strokeWidth={1} />
                  <span>Waiting for backend events...</span>
                </div>
              ) : (
                logs.map((entry, i) => <LogRow key={i} entry={entry} />)
              )}
              <div ref={logsEndRef} />
            </div>
          </div>

          {/* Network Status */}
          <div className="holo-panel holo-panel-amber rounded-lg p-3 corner-cut flex-none">
            <div className="flex items-center gap-2 mb-2">
              <Radio size={10} className="neon-amber" />
              <span className="text-[9px] font-mono font-bold neon-amber tracking-widest">NETWORK</span>
            </div>
            <div className="space-y-1.5">
              {[
                { label: 'LANGGRAPH', status: 'LINKED', ok: true },
                { label: 'CHROMADB', status: 'READY', ok: true },
                { label: 'TTS API', status: 'STANDBY', ok: true },
                { label: 'ASR', status: isListening ? 'LISTENING' : 'IDLE', ok: isListening },
                { label: 'LOG STREAM', status: logs.length > 0 ? 'LIVE' : 'OFFLINE', ok: logs.length > 0 },
              ].map(item => (
                <div key={item.label} className="flex justify-between items-center">
                  <span className="text-[8px] font-mono text-white/40">{item.label}</span>
                  <span className={`text-[8px] font-mono font-bold ${item.ok ? 'neon-green' : 'text-white/30'}`}>
                    {item.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── CENTER: Chat / Evidence Panel ─────────────────────── */}
        <div className="flex-1 flex flex-col gap-3 min-w-0">

          {/* Visual Evidence Port */}
          <div className="holo-panel rounded-lg flex-none h-[30%] relative overflow-hidden corner-cut">
            <div className="scan-bar" style={{ animationDuration: '6s', animationDelay: '2s' }} />
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="flex flex-col items-center gap-3 opacity-35">
                <Eye size={36} strokeWidth={1} className="neon-cyan" />
                <span className="text-[9px] font-mono tracking-[0.3em] text-white/40">[ VISUAL EVIDENCE PORT ]</span>
                <span className="text-[8px] font-mono text-white/20 tracking-wider">READY FOR IMAGE / SCREEN CAPTURE</span>
              </div>
            </div>
            <div className="absolute top-2 left-3 text-[8px] font-mono text-white/20">REC.00</div>
            <div className="absolute top-2 right-3 text-[8px] font-mono text-white/20">●REC</div>
            <div className="absolute bottom-2 left-3 text-[8px] font-mono text-white/20">CHAN.A</div>
            <div className="absolute bottom-2 right-3 text-[8px] font-mono text-white/20">4K.HDR</div>
          </div>

          {/* ── Chat Transcript — more spacious ── */}
          <div className="holo-panel rounded-lg flex-1 overflow-y-auto mercy-scroll flex flex-col gap-4 corner-cut"
            style={{ minHeight: 0, padding: '16px 20px' }}>
            <div className="flex items-center gap-2 mb-1 flex-none">
              <Database size={10} className="neon-cyan" />
              <span className="text-[9px] font-mono font-bold neon-cyan tracking-widest">INTERROGATION.LOG</span>
              <div className="flex-1 h-px bg-white/10 mx-2" />
              <span className="text-[8px] font-mono text-white/30">{msgCount} ENTRIES</span>
            </div>

            {messages.map((msg, idx) => (
              <div key={idx} className={`flex flex-col gap-1.5 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                <div className="text-[9px] font-mono text-white/30 px-1">
                  {msg.role === 'user' ? '◎ SUBJECT' : msg.role === 'assistant' ? '◈ JUDICIAL.AI' : '⚠ SYS'}
                  <span className="ml-2 text-white/15">{new Date().toLocaleTimeString('en-US', { hour12: false })}</span>
                </div>
                <div className={`max-w-[88%] rounded-md corner-cut leading-relaxed
                  ${msg.role === 'user' ? 'msg-user' : msg.role === 'system' ? 'msg-system' : 'msg-assistant'}`}
                  style={{ padding: '10px 14px', fontSize: '12px', fontFamily: "'Share Tech Mono', monospace" }}>
                  {msg.role === 'system' && (
                    <span className="neon-amber mr-2 text-[9px] font-bold">⚠ SYS:</span>
                  )}
                  {msg.role === 'assistant' && (
                    <span className="neon-cyan mr-2 text-[9px] font-bold">◈ AI:</span>
                  )}
                  {msg.content}
                  {msg.image && (
                    <img src={msg.image} alt="evidence" className="mt-3 rounded max-h-40 object-contain border border-white/10" />
                  )}
                </div>
              </div>
            ))}

            {isLoading && messages[messages.length - 1]?.role !== "assistant" && (
              <div className="flex items-start gap-2">
                <div className="msg-assistant rounded-md corner-cut flex items-center gap-2"
                  style={{ padding: '10px 14px', fontSize: '12px', fontFamily: "'Share Tech Mono', monospace" }}>
                  <Loader2 size={12} className="animate-spin neon-cyan" />
                  <span className="neon-cyan/60">PROCESSING EVIDENCE...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Image Preview */}
          {selectedImage && (
            <div className="relative inline-flex items-start self-end holo-panel rounded p-2 corner-cut flex-none">
              <button onClick={() => setSelectedImage(null)} className="absolute -top-2 -right-2 btn-holo rounded-full p-0.5 z-10">
                <X size={10} />
              </button>
              <img src={selectedImage} alt="Evidence" className="h-16 rounded object-contain" />
              <span className="text-[7px] font-mono text-white/30 absolute bottom-0 left-0 right-0 text-center pb-0.5">EVIDENCE.QUEUED</span>
            </div>
          )}

          {/* ── Input Area — more spacious ── */}
          <div className={`flex-none holo-panel rounded-lg corner-cut flex items-end gap-2 ${isListening ? 'border-[#ff3d5a]/50' : ''}`}
            style={{ padding: '10px 12px', transition: 'border-color 0.3s ease' }}>

            <input type="file" accept="image/*" ref={fileInputRef} onChange={handleImageSelect} className="hidden" />

            <div className="flex items-center gap-1.5 flex-none pb-0.5">
              <button className="btn-holo p-2 rounded" title="Camera" onClick={() => { }}>
                <Camera size={14} />
              </button>
              <button className="btn-holo p-2 rounded" title="Upload" onClick={() => fileInputRef.current?.click()}>
                <ImageIcon size={14} />
              </button>
              <div className="w-px h-6 bg-white/10" />
              <button onClick={() => setIsListening(!isListening)}
                className={`p-2 rounded transition-all duration-300 ${isListening ? 'btn-mic-active btn-holo' : 'btn-holo'}`}>
                <Mic size={14} className={isListening ? "animate-pulse" : ""} />
              </button>
              <Waveform active={isListening} />
            </div>

            <textarea
              value={query}
              onChange={e => setQuery(e.target.value)}
              disabled={isLoading}
              rows={1}
              placeholder={
                isListening ? "LISTENING..."
                  : isLoading ? "DELIBERATING..."
                    : selectedImage ? "ANNOTATE..."
                      : "ENTER STATEMENT..."
              }
              className="flex-1 mercy-input tracking-wide px-3 py-2 resize-none min-h-[36px] max-h-[120px]"
              style={{ fontSize: '13px', lineHeight: '1.5', overflowY: 'auto' }}
              onInput={e => {
                const t = e.currentTarget;
                t.style.height = 'auto';
                t.style.height = Math.min(t.scrollHeight, 120) + 'px';
              }}
              onKeyDown={e => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
              }}
            />

            <button
              onClick={handleSend}
              disabled={isLoading || (!query.trim() && !selectedImage)}
              className="btn-send px-3 py-2 rounded flex items-center gap-1.5 font-mono font-bold tracking-wider flex-none"
              style={{ fontSize: '10px' }}>
              <Send size={13} />
              <span>SUBMIT</span>
            </button>
          </div>
        </div>

        {/* ── RIGHT: AI Avatar Terminal ─────────────────────────── */}
        <div className="flex-none w-[36%] flex flex-col gap-3">

          {/* Header row */}
          <div className="holo-panel rounded-lg p-2.5 flex items-center justify-between corner-cut flex-none">
            <div className="flex items-center gap-2">
              <AlertTriangle size={10} className="neon-red" />
              <span className="text-[9px] font-mono font-bold neon-red tracking-widest">JUDICIAL.AI // ACTIVE</span>
            </div>
            <div className="flex gap-2 items-center">
              <div className="w-1.5 h-1.5 rounded-full bg-[#00ff88] pulse-cyan" />
              <span className="text-[8px] font-mono text-white/30">ONLINE</span>
            </div>
          </div>

          {/* VRM Avatar Frame */}
          <div className="avatar-frame rounded-lg flex-1 relative overflow-hidden" style={{ minHeight: 0 }}>
            <div className="corner-tl" /><div className="corner-tr" />
            <div className="corner-bl" /><div className="corner-br" />
            <div className="scan-bar" style={{ animationDuration: '5s', opacity: 0.4 }} />
            <div className="absolute inset-0 bg-gradient-to-b from-[#000a1e] via-[#000510] to-[#000a1e] pointer-events-none" />
            <div className="absolute inset-0 pointer-events-none"
              style={{ background: 'radial-gradient(ellipse at 50% 80%, rgba(0,229,255,0.06) 0%, transparent 70%)' }} />
            <div className="absolute top-4 left-1/2 -translate-x-1/2 w-8 h-8 pointer-events-none z-20 opacity-20">
              <div className="absolute top-0 left-1/2 w-px h-3 bg-[#00e5ff]" style={{ transform: 'translateX(-50%)' }} />
              <div className="absolute bottom-0 left-1/2 w-px h-3 bg-[#00e5ff]" style={{ transform: 'translateX(-50%)' }} />
              <div className="absolute left-0 top-1/2 h-px w-3 bg-[#00e5ff]" style={{ transform: 'translateY(-50%)' }} />
              <div className="absolute right-0 top-1/2 h-px w-3 bg-[#00e5ff]" style={{ transform: 'translateY(-50%)' }} />
              <div className="absolute inset-2 rounded-full border border-[#00e5ff]/40" />
            </div>
            {!isAvatarLoaded && (
              <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-4">
                <div className="w-12 h-12 rounded-full border border-[#00e5ff]/30 flex items-center justify-center">
                  <div className="w-8 h-8 rounded-full border-t border-[#00e5ff] animate-spin" />
                </div>
                <span className="text-[9px] font-mono neon-cyan tracking-[0.3em]">LOADING NEURAL AVATAR...</span>
              </div>
            )}
            <VRMAvatar
              onLoaded={() => setIsAvatarLoaded(true)}
              speech={speech}
              state={speech ? "speaking" : isLoading ? "thinking" : isListening ? "listening" : "idle"}
              onAudioEnd={() => setSpeech(prev => { if (prev) URL.revokeObjectURL(prev.audioUrl); return null; })}
            />
            <div className="absolute bottom-0 left-0 right-0 p-3 z-10 pointer-events-none"
              style={{ background: 'linear-gradient(to top, rgba(0,5,15,0.9), transparent)' }}>
              <div className="flex justify-between items-end">
                <div>
                  <div className="text-[8px] font-mono text-white/25 tracking-wider">SUBJECT.ID</div>
                  <div className="text-[10px] font-mono neon-cyan">JUDICIAL.AI // MX-7</div>
                </div>
                <Waveform active={isLoading || isListening} />
                <div className="text-right">
                  <div className="text-[8px] font-mono text-white/25 tracking-wider">VERDICT</div>
                  <div className={`text-[10px] font-mono font-bold ${isLoading ? 'neon-amber' : 'neon-green'}`}>
                    {isLoading ? 'DELIBERATING' : 'AWAITING'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Stats panel */}
          <div className="holo-panel holo-panel-amber rounded-lg p-3 flex-none corner-cut">
            <div className="grid grid-cols-3 gap-3 text-center">
              <div>
                <div className="text-[9px] font-mono text-white/25 tracking-wider">QUERIES</div>
                <div className="text-[18px] font-bold neon-cyan font-mono">{msgCount}</div>
              </div>
              <div>
                <div className="text-[9px] font-mono text-white/25 tracking-wider">STATUS</div>
                <div className={`text-[11px] font-bold font-mono ${isAvatarLoaded ? 'neon-green' : 'neon-amber'}`}>
                  {isAvatarLoaded ? 'ACTIVE' : 'INIT'}
                </div>
              </div>
              <div>
                <div className="text-[9px] font-mono text-white/25 tracking-wider">THREAT</div>
                <div className="text-[11px] font-bold font-mono neon-green">LOW</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ══════════ BOTTOM STATUS BAR ══════════ */}
      <div className="flex-none h-7 flex items-center justify-between px-5 text-[8px] font-mono"
        style={{ background: 'rgba(0,5,15,0.95)', borderTop: '1px solid rgba(0,229,255,0.1)' }}>
        <div className="flex items-center gap-4 text-white/25">
          <span>MERCY.OS // BUILD 2026.02</span>
          <span>THREE.JS // VRM 1.0</span>
          <span>LANGGRAPH v0.3 // CHROMADB</span>
        </div>
        <div className="flex items-center gap-3 text-white/25">
          <span className="neon-green">● ALL SYSTEMS NOMINAL</span>
        </div>
      </div>
    </div>
  );
}
