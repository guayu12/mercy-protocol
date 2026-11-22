import {
  Activity,
  ArrowDown,
  FileText,
  Image as ImageIcon,
  Mic,
  PanelLeftOpen,
  Send,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ChangeEvent, type KeyboardEvent } from "react";
import ReactMarkdown, {
  type Components,
} from "react-markdown";
import remarkGfm from "remark-gfm";
import { AnimatePresence } from "framer-motion";
import { VRMAvatar, type Speech } from "./VRMAvatar";
import { Stage, type MediaStageData } from "./Stage";
import { API_BASE } from "./api";
import {
  markdownUrlTransform,
  ReportSources,
  reportImage,
  reportLink,
  reportTextLink,
} from "./reportMarkdown.mjs";
import "./App.css";

const ASSISTANT_NAME = "Aura";
const VOICE_LABEL = "晓伊";
type DialogueMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: number;
  turnId?: string;
  attachments?: Array<{
    media_id: string;
    kind: "image" | "video" | "document";
    filename: string;
  }>;
  interrupted?: boolean;
};

type UploadedAttachment = {
  media_id: string;
  kind: "image" | "video" | "document";
  mime: string;
  src: string;
  caption: string;
  filename: string;
  previewUrl: string | null;
};

type ReportSource = { number: number; title: string; url: string };

type Report = {
  report_id: string;
  title: string;
  query: string;
  markdown: string;
  created_at: number;
  complete?: boolean;
  link_policy?: "open" | "sources_only";
  sources?: ReportSource[];
};

type LogEntry = {
  ts: number;
  level: string;
  category: string;
  message: string;
};

type SpeechSentence = {
  turn_id: string;
  id: string;
  text: string;
};

type SpeechFetch = {
  controller: AbortController;
  promise: Promise<Speech>;
};

type ChatEvent =
  | { type: "turn"; turn_id: string }
  | { type: "status"; state: string; label?: string }
  | { type: "tool"; name: string; status: string }
  | ({ type: "speech.sentence" } & SpeechSentence)
  | {
      type: "report.open";
      report_id: string;
      title: string;
      query: string;
      link_policy?: "open" | "sources_only";
      sources?: ReportSource[];
    }
  | { type: "report.delta"; report_id: string; markdown: string }
  | { type: "report.done"; report_id: string }
  | { type: "report.highlight"; report_id: string; anchor: string }
  | ({ type: "stage.show" } & MediaStageData)
  | { type: "stage.clear" }
  | { type: "done"; turn_id: string; spoken_text: string }
  | { type: "error"; message: string };

type BackendState = "idle" | "thinking" | "working";

const markdownComponents: Components = {
  a: reportLink,
  img: reportImage,
  code: ({ children, className }) =>
    className ? (
      <code className={className}>{children}</code>
    ) : (
      <code className="inline-code">{children}</code>
    ),
  pre: ({ children }) => <pre className="md-code">{children}</pre>,
  table: ({ children }) => (
    <div className="table-wrap">
      <table className="md-table">{children}</table>
    </div>
  ),
};

const sourcesOnlyMarkdownComponents: Components = {
  a: reportTextLink,
  img: reportImage,
};

function makeId(prefix: string) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function timeLabel(timestamp?: number) {
  if (!timestamp) return "";
  const date = new Date(timestamp < 1e12 ? timestamp * 1000 : timestamp);
  return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

function stripMarkdown(value: string) {
  return value
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/`+/g, "")
    .replace(/\*\*|__|~~|\*|_/g, "")
    .replace(/^\s{0,3}#{1,6}\s*/, "")
    .replace(/^\s*>\s?/, "")
    .replace(/^\s*(?:[-+*]|\d+[.)])\s+/, "")
    .replace(/<[^>]*>/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function reportSummary(markdown: string, title: string) {
  const lines = markdown.split(/\r?\n/);
  const tableLines = new Set<number>();
  const separator = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/;
  for (let index = 0; index < lines.length; index += 1) {
    if (!separator.test(lines[index])) continue;
    tableLines.add(index);
    for (let row = index - 1; row >= 0 && lines[row].includes("|"); row -= 1) tableLines.add(row);
    for (let row = index + 1; row < lines.length && lines[row].includes("|"); row += 1) tableLines.add(row);
  }

  const normalizedTitle = stripMarkdown(title).toLocaleLowerCase();
  let fence: { character: string; length: number } | null = null;
  let hasCode = false;
  let hasTable = tableLines.size > 0;
  let prose = "";
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const fenceMatch = line.match(/^\s*(`{3,}|~{3,})/);
    if (fence) {
      if (
        fenceMatch &&
        fenceMatch[1][0] === fence.character &&
        fenceMatch[1].length >= fence.length
      ) {
        fence = null;
      }
      continue;
    }
    if (fenceMatch) {
      fence = { character: fenceMatch[1][0], length: fenceMatch[1].length };
      hasCode = true;
      continue;
    }
    if (tableLines.has(index)) continue;
    const cleanLine = stripMarkdown(line);
    if (!cleanLine) continue;
    const heading = line.match(/^\s{0,3}#{1,6}\s+/);
    if (heading && cleanLine.toLocaleLowerCase() === normalizedTitle) continue;
    prose = cleanLine;
    break;
  }

  if (prose) {
    const characters = Array.from(prose);
    return characters.length > 64 ? `${characters.slice(0, 63).join("")}…` : prose;
  }
  const fallback = [];
  if (hasTable) fallback.push("含表格");
  if (hasCode) fallback.push("含代码");
  return fallback.join(" · ");
}

function isAbortError(error: unknown) {
  return error instanceof Error && error.name === "AbortError";
}

function decodeAudio(audio: string, mime: string, visemes: Speech["visemes"]): Speech {
  const bytes = Uint8Array.from(atob(audio), (character) => character.charCodeAt(0));
  return {
    audioUrl: URL.createObjectURL(new Blob([bytes], { type: mime })),
    visemes,
  };
}

export function App() {
  const [sessionId] = useState(() => {
    const saved = localStorage.getItem("mercy.threadId");
    if (saved) return saved;
    const id = makeId("mp");
    localStorage.setItem("mercy.threadId", id);
    return id;
  });
  const [query, setQuery] = useState("");
  const [attachments, setAttachments] = useState<UploadedAttachment[]>([]);
  const [stage, setStage] = useState<MediaStageData | null>(null);
  const [stageCurrentMediaId, setStageCurrentMediaId] = useState<string | null>(null);
  const [isReportExpanded, setIsReportExpanded] = useState(false);
  const [isNarrow, setIsNarrow] = useState(window.innerWidth < 1100);
  const [isListening, setIsListening] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [hasQueuedSpeech, setHasQueuedSpeech] = useState(false);
  const [speech, setSpeech] = useState<Speech | null>(null);
  const [subtitle, setSubtitle] = useState("");
  const [backendState, setBackendState] = useState<BackendState>("idle");
  const [workingLabel, setWorkingLabel] = useState("");
  const [backendConnected, setBackendConnected] = useState(false);
  const [maxTierAvailable, setMaxTierAvailable] = useState(false);
  const [deepMode, setDeepMode] = useState(false);
  const [leftExpanded, setLeftExpanded] = useState(false);
  const [clock, setClock] = useState("");
  const [reports, setReports] = useState<Report[]>([]);
  const [pendingReportHighlight, setPendingReportHighlight] = useState<{
    reportId: string;
    anchor: string;
    nonce: number;
  } | null>(null);
  const [messages, setMessages] = useState<DialogueMessage[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [hasUnreadReports, setHasUnreadReports] = useState(false);
  const [isAvatarLoaded, setIsAvatarLoaded] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const reportListRef = useRef<HTMLDivElement>(null);
  const dialogueListRef = useRef<HTMLDivElement>(null);
  const logFeedRef = useRef<HTMLDivElement>(null);
  const subtitleTextRef = useRef<HTMLDivElement>(null);
  const reportAtBottomRef = useRef(true);
  const shouldFollowReportRef = useRef(false);
  const viewingReportIdRef = useRef<string | null>(null);
  const sessionRestoredRef = useRef(false);
  const requestControllerRef = useRef<AbortController | null>(null);
  const activeTurnIdRef = useRef<string | null>(null);
  const pendingInterruptedRef = useRef<{ turn_id: string; spoken_sentence_ids: string[] } | null>(null);
  const sentenceTextRef = useRef(new Map<string, string>());
  const spokenSentenceIdsRef = useRef(new Map<string, string[]>());
  const speechQueueRef = useRef<SpeechSentence[]>([]);
  const speechCacheRef = useRef(new Map<string, Speech>());
  const speechFetchesRef = useRef(new Map<string, SpeechFetch>());
  const speechFailuresRef = useRef(new Set<string>());
  const currentPlaybackRef = useRef<{ sentence: SpeechSentence; audio: Speech } | null>(null);
  const speechWatchdogRef = useRef<number | null>(null);
  const speechMetadataTimeoutRef = useRef<number | null>(null);
  const speechMetadataProbeRef = useRef<HTMLAudioElement | null>(null);
  const handleAudioEndRef = useRef<() => void>(() => undefined);
  const isPumpingRef = useRef(false);
  const audioEpochRef = useRef(0);
  const highlightTimersRef = useRef(
    new Map<string, { timer: number; target: HTMLElement }>(),
  );
  const reportHighlightNonceRef = useRef(0);

  const addSystemNotice = useCallback((content: string) => {
    setMessages((previous) => [
      ...previous,
      { id: makeId("notice"), role: "system", content, createdAt: Date.now() },
    ]);
  }, []);

  const getSpeechAudio = useCallback((sentenceId: string) => {
    const cached = speechCacheRef.current.get(sentenceId);
    if (cached) return Promise.resolve(cached);

    const activeFetch = speechFetchesRef.current.get(sentenceId);
    if (activeFetch) return activeFetch.promise;

    const controller = new AbortController();
    const promise = fetch(`${API_BASE}/api/speech/${encodeURIComponent(sentenceId)}`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Speech request failed: ${response.status}`);
        const payload = (await response.json()) as {
          audio: string;
          mime: string;
          visemes: Speech["visemes"];
        };
        if (controller.signal.aborted) throw new DOMException("Aborted", "AbortError");
        const audio = decodeAudio(payload.audio, payload.mime, payload.visemes);
        if (controller.signal.aborted) {
          URL.revokeObjectURL(audio.audioUrl);
          throw new DOMException("Aborted", "AbortError");
        }
        speechCacheRef.current.set(sentenceId, audio);
        return audio;
      })
      .catch((error: unknown) => {
        if (!isAbortError(error)) speechFailuresRef.current.add(sentenceId);
        throw error;
      })
      .finally(() => {
        if (speechFetchesRef.current.get(sentenceId)?.controller === controller) {
          speechFetchesRef.current.delete(sentenceId);
        }
      });

    speechFetchesRef.current.set(sentenceId, { controller, promise });
    return promise;
  }, []);

  const prefetchNextSentence = useCallback(() => {
    const next = speechQueueRef.current[0];
    if (!next || speechFailuresRef.current.has(next.id)) return;
    void getSpeechAudio(next.id).catch(() => undefined);
  }, [getSpeechAudio]);

  const clearSpeechMetadataProbe = useCallback(() => {
    if (speechMetadataTimeoutRef.current !== null) {
      window.clearTimeout(speechMetadataTimeoutRef.current);
      speechMetadataTimeoutRef.current = null;
    }
    const probe = speechMetadataProbeRef.current;
    if (probe) {
      probe.onloadedmetadata = null;
      probe.onerror = null;
      probe.pause();
      probe.removeAttribute("src");
      probe.load();
      speechMetadataProbeRef.current = null;
    }
  }, []);

  const pumpSpeechQueue = useCallback(async () => {
    if (isPumpingRef.current || currentPlaybackRef.current) return;
    const sentence = speechQueueRef.current.shift();
    if (!sentence) {
      setHasQueuedSpeech(false);
      if (!requestControllerRef.current) activeTurnIdRef.current = null;
      return;
    }

    const epoch = audioEpochRef.current;
    isPumpingRef.current = true;
    setSubtitle(sentence.text);
    try {
      if (!speechFailuresRef.current.has(sentence.id)) {
        const audio = await getSpeechAudio(sentence.id);
        if (epoch !== audioEpochRef.current) return;
        speechCacheRef.current.delete(sentence.id);
        const spoken = spokenSentenceIdsRef.current.get(sentence.turn_id) ?? [];
        if (!spoken.includes(sentence.id)) {
          spoken.push(sentence.id);
          spokenSentenceIdsRef.current.set(sentence.turn_id, spoken);
        }
        currentPlaybackRef.current = { sentence, audio };
        setSubtitle(sentence.text);
        setIsAudioPlaying(true);
        setHasQueuedSpeech(true);
        setSpeech(audio);
        const lastVisemeEnd = audio.visemes.reduce(
          (latestEnd, cue) => Math.max(latestEnd, cue.end),
          0,
        );
        const handedAt = performance.now();
        clearSpeechMetadataProbe();
        const probe = new Audio(audio.audioUrl);
        probe.preload = "metadata";
        speechMetadataProbeRef.current = probe;
        let metadataHandled = false;
        const scheduleWatchdog = (audioDuration: number) => {
          if (metadataHandled) return;
          metadataHandled = true;
          clearSpeechMetadataProbe();
          if (
            epoch !== audioEpochRef.current ||
            currentPlaybackRef.current?.sentence.id !== sentence.id
          ) {
            return;
          }
          if (speechWatchdogRef.current !== null) {
            window.clearTimeout(speechWatchdogRef.current);
          }
          const duration = Number.isFinite(audioDuration)
            ? audioDuration
            : lastVisemeEnd;
          const totalDuration = Math.max(lastVisemeEnd, duration, 0.5) + 3;
          const delay = Math.max(
            0,
            totalDuration * 1000 - (performance.now() - handedAt),
          );
          speechWatchdogRef.current = window.setTimeout(() => {
            speechWatchdogRef.current = null;
            handleAudioEndRef.current();
          }, delay);
        };
        probe.onloadedmetadata = () => scheduleWatchdog(probe.duration);
        probe.onerror = () => scheduleWatchdog(lastVisemeEnd);
        speechMetadataTimeoutRef.current = window.setTimeout(
          () => scheduleWatchdog(lastVisemeEnd),
          2000,
        );
      }
    } catch (error) {
      if (!isAbortError(error)) setSubtitle(sentence.text);
    } finally {
      if (epoch === audioEpochRef.current) isPumpingRef.current = false;
    }

    if (epoch !== audioEpochRef.current) return;
    if (currentPlaybackRef.current) {
      prefetchNextSentence();
      return;
    }
    if (speechQueueRef.current.length > 0) {
      setHasQueuedSpeech(true);
      requestAnimationFrame(() => void pumpSpeechQueue());
    } else {
      setHasQueuedSpeech(false);
      if (!requestControllerRef.current) activeTurnIdRef.current = null;
    }
  }, [clearSpeechMetadataProbe, getSpeechAudio, prefetchNextSentence]);

  const enqueueSentence = useCallback(
    (sentence: SpeechSentence) => {
      sentenceTextRef.current.set(sentence.id, sentence.text);
      speechQueueRef.current.push(sentence);
      setHasQueuedSpeech(true);
      if (currentPlaybackRef.current) prefetchNextSentence();
      else void pumpSpeechQueue();
    },
    [prefetchNextSentence, pumpSpeechQueue],
  );

  const handleAudioEnd = useCallback(() => {
    if (speechWatchdogRef.current !== null) {
      window.clearTimeout(speechWatchdogRef.current);
      speechWatchdogRef.current = null;
    }
    clearSpeechMetadataProbe();
    const playback = currentPlaybackRef.current;
    if (!playback) return;
    URL.revokeObjectURL(playback.audio.audioUrl);
    currentPlaybackRef.current = null;
    setSpeech(null);
    setIsAudioPlaying(false);
    if (speechQueueRef.current.length > 0) {
      setHasQueuedSpeech(true);
      requestAnimationFrame(() => void pumpSpeechQueue());
    } else {
      setHasQueuedSpeech(false);
      if (!requestControllerRef.current) activeTurnIdRef.current = null;
    }
  }, [clearSpeechMetadataProbe, pumpSpeechQueue]);

  useEffect(() => {
    handleAudioEndRef.current = handleAudioEnd;
  }, [handleAudioEnd]);

  const interruptActiveTurn = useCallback(() => {
    const turnId = activeTurnIdRef.current;
    const hasActiveWork =
      Boolean(turnId) ||
      Boolean(requestControllerRef.current) ||
      Boolean(currentPlaybackRef.current) ||
      speechQueueRef.current.length > 0 ||
      isPumpingRef.current;
    if (!hasActiveWork) return;

    if (turnId) {
      const spokenIds = [...(spokenSentenceIdsRef.current.get(turnId) ?? [])];
      pendingInterruptedRef.current = { turn_id: turnId, spoken_sentence_ids: spokenIds };
      const spokenText = spokenIds.map((id) => sentenceTextRef.current.get(id) ?? "").join("");
      setMessages((previous) =>
        previous.map((message) =>
          message.turnId === turnId
            ? { ...message, content: spokenText, interrupted: true }
            : message,
        ),
      );
    }

    requestControllerRef.current?.abort();
    requestControllerRef.current = null;
    for (const request of speechFetchesRef.current.values()) request.controller.abort();
    speechFetchesRef.current.clear();
    for (const audio of speechCacheRef.current.values()) URL.revokeObjectURL(audio.audioUrl);
    speechCacheRef.current.clear();
    speechFailuresRef.current.clear();
    speechQueueRef.current = [];
    clearSpeechMetadataProbe();
    if (speechWatchdogRef.current !== null) {
      window.clearTimeout(speechWatchdogRef.current);
      speechWatchdogRef.current = null;
    }
    const playback = currentPlaybackRef.current;
    if (playback) URL.revokeObjectURL(playback.audio.audioUrl);
    currentPlaybackRef.current = null;
    audioEpochRef.current += 1;
    isPumpingRef.current = false;
    activeTurnIdRef.current = null;
    setSpeech(null);
    setIsAudioPlaying(false);
    setHasQueuedSpeech(false);
    setIsLoading(false);
    setBackendState("idle");
  }, [clearSpeechMetadataProbe]);

  useEffect(() => {
    const updateClock = () =>
      setClock(new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false }));
    updateClock();
    const timer = window.setInterval(updateClock, 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const updateWidth = () => setIsNarrow(window.innerWidth < 1100);
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  useEffect(() => {
    const eventSource = new EventSource(`${API_BASE}/api/logs`);
    eventSource.onopen = () => setBackendConnected(true);
    eventSource.onerror = () => setBackendConnected(false);
    eventSource.onmessage = (event) => {
      try {
        const entry = JSON.parse(event.data) as LogEntry;
        setLogs((previous) => [...previous, entry].slice(-300));
      } catch {
        // Ignore malformed log entries and keep the live connection.
      }
    };
    return () => eventSource.close();
  }, []);

  useEffect(() => {
    if (!backendConnected) {
      setMaxTierAvailable(false);
      return;
    }
    const controller = new AbortController();
    fetch(`${API_BASE}/api/capabilities`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Capabilities request failed: ${response.status}`);
        return (await response.json()) as { max_tier?: boolean };
      })
      .then((capabilities) => {
        const available = capabilities.max_tier === true;
        setMaxTierAvailable(available);
        if (!available) setDeepMode(false);
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [backendConnected]);

  useEffect(() => {
    if (!backendConnected || sessionRestoredRef.current) return;
    const controller = new AbortController();
    fetch(`${API_BASE}/api/session/${encodeURIComponent(sessionId)}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          addSystemNotice("无法恢复本地会话记录。");
          return null;
        }
        return (await response.json()) as {
          reports?: Report[];
          stage?: MediaStageData | null;
          utterances?: Array<{
            id: number;
            turn_id: string;
            role: "user" | "assistant";
            text: string;
            attachments?: Array<{
              media_id: string;
              kind: "image" | "video" | "document";
              filename: string;
            }>;
            interrupted?: boolean;
            created_at: number;
          }>;
        };
      })
      .then((session) => {
        if (!session || controller.signal.aborted) return;
        sessionRestoredRef.current = true;
        const restoredReports = (session.reports ?? []).map((report) => ({
          ...report,
          complete: true,
          link_policy: report.link_policy ?? "open",
        }));
        const restoredMessages = (session.utterances ?? [])
          .filter((utterance) => utterance.text.trim() || utterance.attachments?.length)
          .map((utterance) => ({
            id: `utterance_${utterance.id}`,
            turnId: utterance.turn_id,
            role: utterance.role,
            content: utterance.text,
            attachments: utterance.attachments ?? [],
            interrupted: utterance.interrupted,
            createdAt: utterance.created_at,
          }));
        setReports((previous) => (previous.length ? previous : restoredReports));
        setMessages((previous) => (previous.length ? previous : restoredMessages));
        if (session.stage) {
          setStage(session.stage);
          setStageCurrentMediaId(session.stage.items[0]?.media_id ?? null);
        }
        shouldFollowReportRef.current = true;
      })
      .catch((error: unknown) => {
        if (isAbortError(error)) return;
      });
    return () => controller.abort();
  }, [addSystemNotice, backendConnected, sessionId]);

  useEffect(() => {
    const feed = logFeedRef.current;
    if (feed) feed.scrollTop = feed.scrollHeight;
  }, [logs]);

  useLayoutEffect(() => {
    const list = reportListRef.current;
    if (!list) return;
    if (shouldFollowReportRef.current) {
      list.scrollTop = list.scrollHeight;
      reportAtBottomRef.current = true;
      shouldFollowReportRef.current = false;
    } else if (shouldFollowReportRef.current || reportAtBottomRef.current) {
      list.scrollTop = list.scrollHeight;
      reportAtBottomRef.current = true;
    }
    if (shouldFollowReportRef.current) shouldFollowReportRef.current = false;
  }, [reports]);

  useEffect(() => {
    const list = reportListRef.current;
    if (!list) return;
    const cards = Array.from(list.querySelectorAll<HTMLElement>(".report-card"));
    const ratios = new Map<string, number>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const id = (entry.target as HTMLElement).dataset.reportId;
          if (id) ratios.set(id, entry.isIntersecting ? entry.intersectionRatio : 0);
        }
        const visible = [...ratios.entries()].sort((a, b) => b[1] - a[1])[0];
        if (visible && visible[1] > 0) viewingReportIdRef.current = visible[0];
      },
      { root: list, threshold: [0, 0.2, 0.5, 0.8, 1] },
    );
    cards.forEach((card) => observer.observe(card));
    return () => observer.disconnect();
  }, [reports]);

  useEffect(() => {
    const panel = dialogueListRef.current;
    if (panel) panel.scrollTop = panel.scrollHeight;
  }, [messages]);

  useEffect(() => {
    const element = subtitleTextRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [subtitle]);

  useEffect(() => {
    if (!isListening) return;
    const Recognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!Recognition) {
      setIsListening(false);
      return;
    }
    const recognition = new Recognition();
    recognition.lang = "zh-CN";
    recognition.interimResults = false;
    recognition.onresult = (event: any) => {
      const recognized = event.results?.[0]?.[0]?.transcript;
      if (recognized) setQuery((previous) => (previous ? `${previous} ${recognized}` : recognized));
      setIsListening(false);
    };
    recognition.onerror = () => setIsListening(false);
    recognition.onend = () => setIsListening(false);
    try {
      recognition.start();
    } catch {
      setIsListening(false);
    }
    return () => {
      try {
        recognition.stop();
      } catch {
        // Recognition may already have stopped.
      }
    };
  }, [isListening]);

  useEffect(
    () => () => {
      requestControllerRef.current?.abort();
      for (const request of speechFetchesRef.current.values()) request.controller.abort();
      for (const audio of speechCacheRef.current.values()) URL.revokeObjectURL(audio.audioUrl);
      const playback = currentPlaybackRef.current;
      if (playback) URL.revokeObjectURL(playback.audio.audioUrl);
      clearSpeechMetadataProbe();
      if (speechWatchdogRef.current !== null) {
        window.clearTimeout(speechWatchdogRef.current);
        speechWatchdogRef.current = null;
      }
      for (const { timer } of highlightTimersRef.current.values()) window.clearTimeout(timer);
    },
    [clearSpeechMetadataProbe],
  );

  const applyReportHighlight = useCallback((reportId: string, anchor: string) => {
    const list = reportListRef.current;
    if (!list) return false;
    const card = Array.from(list.querySelectorAll<HTMLElement>(".report-card")).find(
      (item) => item.dataset.reportId === reportId,
    );
    if (!card) return false;
    const target = Array.from(card.querySelectorAll<HTMLElement>("h1,h2,h3,h4,h5,h6,p,li,td,pre")).find(
      (item) => item.textContent?.toLocaleLowerCase().includes(anchor.toLocaleLowerCase()),
    );
    if (!target) return false;
    const previous = highlightTimersRef.current.get(reportId);
    if (previous) {
      window.clearTimeout(previous.timer);
      previous.target.classList.remove("report-highlight");
      highlightTimersRef.current.delete(reportId);
    }
    target.classList.add("report-highlight");
    list.scrollTo({
      top: list.scrollTop + target.getBoundingClientRect().top - list.getBoundingClientRect().top - 24,
      behavior: "smooth",
    });
    const timer = window.setTimeout(() => {
      if (highlightTimersRef.current.get(reportId)?.timer !== timer) return;
      target.classList.remove("report-highlight");
      highlightTimersRef.current.delete(reportId);
    }, 4500);
    highlightTimersRef.current.set(reportId, { timer, target });
    return true;
  }, []);

  useLayoutEffect(() => {
    if (!pendingReportHighlight || !isReportExpanded) return;
    const frame = window.requestAnimationFrame(() => {
      if (applyReportHighlight(pendingReportHighlight.reportId, pendingReportHighlight.anchor)) {
        setPendingReportHighlight((pending) =>
          pending?.nonce === pendingReportHighlight.nonce ? null : pending,
        );
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [applyReportHighlight, isReportExpanded, pendingReportHighlight, reports, stage]);

  const handleChatEvent = useCallback(
    (event: ChatEvent) => {
      switch (event.type) {
        case "turn":
          activeTurnIdRef.current = event.turn_id;
          spokenSentenceIdsRef.current.set(event.turn_id, []);
          setMessages((previous) => [
            ...previous,
            {
              id: event.turn_id,
              turnId: event.turn_id,
              role: "assistant",
              content: "",
              createdAt: Date.now(),
            },
          ]);
          break;
        case "status":
          if (event.state === "thinking") {
            setBackendState("thinking");
            setWorkingLabel("");
          } else if (event.state === "working") {
            setBackendState("working");
            setWorkingLabel(event.label ?? "正在处理");
          } else if (event.state === "idle") {
            setBackendState("idle");
            setWorkingLabel("");
          }
          break;
        case "speech.sentence":
          setMessages((previous) => {
            const index = previous.findIndex(
              (message) => message.role === "assistant" && message.turnId === event.turn_id,
            );
            if (index === -1) {
              return [
                ...previous,
                {
                  id: event.turn_id,
                  turnId: event.turn_id,
                  role: "assistant",
                  content: event.text,
                  createdAt: Date.now(),
                },
              ];
            }
            const next = [...previous];
            next[index] = { ...next[index], content: next[index].content + event.text };
            return next;
          });
          enqueueSentence({ turn_id: event.turn_id, id: event.id, text: event.text });
          break;
        case "report.open": {
          const follow = reportAtBottomRef.current;
          shouldFollowReportRef.current = follow;
          if (!follow) setHasUnreadReports(true);
          setReports((previous) => {
            if (previous.some((report) => report.report_id === event.report_id)) return previous;
            return [
              ...previous,
              {
                report_id: event.report_id,
                title: event.title,
                query: event.query,
                markdown: "",
                created_at: Date.now() / 1000,
                complete: false,
                link_policy: event.link_policy ?? "open",
                sources:
                  event.sources ??
                  (event.link_policy === "sources_only" ? [] : undefined),
              },
            ];
          });
          break;
        }
        case "report.delta":
          setReports((previous) =>
            previous.map((report) =>
              report.report_id === event.report_id
                ? { ...report, markdown: report.markdown + event.markdown }
                : report,
            ),
          );
          break;
        case "report.done":
          setReports((previous) =>
            previous.map((report) =>
              report.report_id === event.report_id ? { ...report, complete: true } : report,
            ),
          );
          break;
        case "report.highlight":
          setPendingReportHighlight({
            reportId: event.report_id,
            anchor: event.anchor,
            nonce: ++reportHighlightNonceRef.current,
          });
          setStage((previous) =>
            previous?.focus === "primary" ? { ...previous, focus: "normal" } : previous,
          );
          setIsReportExpanded(true);
          break;
        case "stage.show":
          setStage({ caption: event.caption, focus: event.focus, items: event.items });
          setStageCurrentMediaId((current) =>
            event.items.some((item) => item.media_id === current)
              ? current
              : event.items[0]?.media_id ?? null,
          );
          setIsReportExpanded(false);
          break;
        case "stage.clear":
          setStage(null);
          setStageCurrentMediaId(null);
          break;
        case "done":
          setMessages((previous) =>
            previous.map((message) =>
              message.role === "assistant" &&
              message.turnId === event.turn_id &&
              !message.content &&
              event.spoken_text
                ? { ...message, content: event.spoken_text }
                : message,
            ),
          );
          setBackendState("idle");
          setWorkingLabel("");
          break;
        case "error":
          addSystemNotice(event.message);
          setBackendState("idle");
          setWorkingLabel("");
          break;
        case "tool":
          break;
      }
    },
    [addSystemNotice, enqueueSentence],
  );

  const handleSend = useCallback(async () => {
    const message = query.trim();
    const pendingAttachments = attachments;
    if (!message && pendingAttachments.length === 0) return;
    if (
      activeTurnIdRef.current ||
      requestControllerRef.current ||
      currentPlaybackRef.current ||
      speechQueueRef.current.length > 0 ||
      isPumpingRef.current
    ) {
      interruptActiveTurn();
    }

    const interrupted = pendingInterruptedRef.current ?? undefined;
    pendingInterruptedRef.current = null;
    const text = message || "请帮我看看这份媒体。";
    setQuery("");
    setAttachments([]);
    pendingAttachments.forEach((attachment) => {
      if (attachment.previewUrl) URL.revokeObjectURL(attachment.previewUrl);
    });
    setMessages((previous) => [
      ...previous,
      {
        id: makeId("user"),
        role: "user",
        content: text,
        attachments: pendingAttachments.map(({ media_id, kind, filename }) => ({
          media_id,
          kind,
          filename,
        })),
        createdAt: Date.now(),
      },
    ]);

    const controller = new AbortController();
    requestControllerRef.current = controller;
    setIsLoading(true);
    setBackendState("thinking");
    setWorkingLabel("");
    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          message: text,
          thread_id: sessionId,
          attachments: pendingAttachments.map((attachment) => attachment.media_id),
          screen_state: {
            viewing_report_id: viewingReportIdRef.current,
            stage_media_ids: stage?.items.map((item) => item.media_id) ?? [],
            stage_current_media_id: stageCurrentMediaId,
          },
          model_tier: deepMode ? "max" : "normal",
          interrupted,
        }),
      });
      if (!response.ok) throw new Error(`Chat request failed: ${response.status}`);
      if (!response.body) throw new Error("Chat response did not include a stream.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const dispatch = (chunk: string) => {
        const data = chunk
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim())
          .join("\n");
        if (!data || controller.signal.aborted) return;
        try {
          handleChatEvent(JSON.parse(data) as ChatEvent);
        } catch {
          addSystemNotice("收到无法识别的后端事件。");
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let separator = buffer.indexOf("\n\n");
        while (separator !== -1) {
          dispatch(buffer.slice(0, separator));
          buffer = buffer.slice(separator + 2);
          separator = buffer.indexOf("\n\n");
        }
      }
      buffer += decoder.decode();
      if (buffer.trim()) dispatch(buffer);
    } catch (error) {
      if (!isAbortError(error)) addSystemNotice(error instanceof Error ? error.message : "对话请求失败。");
    } finally {
      if (requestControllerRef.current === controller) {
        requestControllerRef.current = null;
        setIsLoading(false);
      }
      if (
        !requestControllerRef.current &&
        !currentPlaybackRef.current &&
        speechQueueRef.current.length === 0 &&
        !isPumpingRef.current
      ) {
        activeTurnIdRef.current = null;
      }
    }
  }, [addSystemNotice, attachments, deepMode, handleChatEvent, interruptActiveTurn, query, sessionId, stage, stageCurrentMediaId]);

  const handleImageSelect = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    for (const file of files) {
      try {
        const form = new FormData();
        form.append("thread_id", sessionId);
        form.append("file", file);
        const response = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
        if (!response.ok) {
          const body = await response.json().catch(() => null);
          throw new Error(body?.detail ?? `上传失败：${response.status}`);
        }
        const uploaded = (await response.json()) as {
          media_id: string;
          kind: "image" | "video" | "document";
          mime: string;
        };
        setAttachments((previous) => [
          ...previous,
          {
            ...uploaded,
            src: `/api/media/${uploaded.media_id}`,
            caption: file.name,
            filename: file.name,
            previewUrl:
              uploaded.kind === "image" ? URL.createObjectURL(file) : null,
          },
        ]);
      } catch (error) {
        addSystemNotice(error instanceof Error ? error.message : "媒体上传失败。");
      }
    }
  };

  const handleRemoveAttachment = (mediaId: string) => {
    setAttachments((previous) => {
      const removed = previous.find((item) => item.media_id === mediaId);
      if (removed?.previewUrl) URL.revokeObjectURL(removed.previewUrl);
      return previous.filter((item) => item.media_id !== mediaId);
    });
  };

  const handleCloseStage = useCallback(async () => {
    setStage(null);
    setStageCurrentMediaId(null);
    try {
      const response = await fetch(
        `${API_BASE}/api/session/${encodeURIComponent(sessionId)}/stage`,
        { method: "DELETE" },
      );
      if (!response.ok) throw new Error(`关闭展示台失败：${response.status}`);
    } catch (error) {
      addSystemNotice(error instanceof Error ? error.message : "关闭展示台失败。");
    }
  }, [addSystemNotice, sessionId]);

  const handleComposerChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    const next = event.target.value;
    if (next && !query && (
      activeTurnIdRef.current ||
      requestControllerRef.current ||
      currentPlaybackRef.current ||
      speechQueueRef.current.length > 0 ||
      isPumpingRef.current
    )) {
      interruptActiveTurn();
    }
    setQuery(next);
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  };

  const handleMicToggle = () => {
    if (!isListening) interruptActiveTurn();
    setIsListening((previous) => !previous);
  };

  const handleReportScroll = () => {
    const list = reportListRef.current;
    if (!list) return;
    const atBottom = list.scrollTop + list.clientHeight >= list.scrollHeight - 36;
    reportAtBottomRef.current = atBottom;
    if (atBottom) setHasUnreadReports(false);
  };

  const scrollToLatestReport = () => {
    const list = reportListRef.current;
    if (!list) return;
    list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
    reportAtBottomRef.current = true;
    setHasUnreadReports(false);
  };

  const status =
    isListening || query.trim()
      ? { state: "listening", label: "倾听中" }
      : hasQueuedSpeech || isAudioPlaying
        ? { state: "speaking", label: "说话中" }
        : backendState === "thinking"
          ? { state: "thinking", label: "思考中" }
          : backendState === "working"
            ? { state: "thinking", label: workingLabel || "正在处理" }
            : { state: "idle", label: "空闲" };
  const latestReport = reports[reports.length - 1];
  const latestReportSummary = latestReport
    ? reportSummary(latestReport.markdown, latestReport.title)
    : "";
  const reportNeedsLayout = Boolean(
    latestReport &&
      (latestReport.markdown.length > 1500 ||
        /^\s*\|?.+\|\s*\n\s*\|?\s*:?-{3,}/m.test(latestReport.markdown)),
  );
  const layout = stage
    ? stage.focus === "primary" && !isNarrow
      ? "focus_evidence"
      : "evidence"
    : reportNeedsLayout
      ? "report"
      : "conversation";
  const layoutLabel = {
    conversation: "对话",
    report: "报告",
    evidence: "展示台",
    focus_evidence: "聚焦看图",
  }[layout];

  return (
    <div id="app" data-layout={layout} data-left-expanded={leftExpanded} data-state={status.state}>
      <header id="topbar" data-tauri-drag-region>
        <div className="topbar-left" data-tauri-drag-region>
          <div className="brand" data-tauri-drag-region>
            <div className="brand-mark">M</div>
            <span>Mercy</span>
          </div>
          <div className="topbar-divider" />
          <button className="sys-pill" type="button" onClick={() => setLeftExpanded((value) => !value)}>
            <span className={`dot-live ${backendConnected ? "" : "offline"}`} />
            <span>后端 {backendConnected ? "在线" : "离线"}</span>
          </button>
          <span className="session-pill">#{sessionId.slice(-8).toUpperCase()}</span>
        </div>
        <div className="topbar-right">
          {maxTierAvailable && (
            <button
              className={`deep-mode-toggle ${deepMode ? "active" : ""}`}
              type="button"
              aria-pressed={deepMode}
              onClick={() => setDeepMode((value) => !value)}
            >
              {deepMode && <span className="deep-mode-dot" />}
              <span>{deepMode ? "深度模式 · 开" : "深度模式"}</span>
            </button>
          )}
          <div className="layout-pill">
            布局：<strong>{layoutLabel}</strong>
          </div>
          <span className="clock-text">{clock}</span>
        </div>
      </header>

      <button
        id="left-drawer-backdrop"
        aria-label="关闭系统状态抽屉"
        type="button"
        onClick={() => setLeftExpanded(false)}
      />

      <main id="workspace">
        <aside id="left-rail" aria-label="系统状态与实时事件">
          <div className="left-full">
            <section className="rail-section">
              <div className="rail-head">
                <span className="rail-title">服务连接</span>
                <span className={`rail-badge ${backendConnected ? "ok" : "err"}`}>
                  {backendConnected ? "已连接" : "离线"}
                </span>
              </div>
              <div className="net-row">
                <span className="net-left">
                  {backendConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
                  后端事件流
                </span>
                <span className={`service-state ${backendConnected ? "online" : "offline"}`}>
                  {backendConnected ? "在线" : "离线"}
                </span>
              </div>
              <div className="net-row">
                <span className="net-left">
                  <Activity size={14} />
                  口语合成
                </span>
                <span className="service-state">{VOICE_LABEL} · 按需</span>
              </div>
            </section>

            <section className="rail-section grow">
              <div className="rail-head">
                <span className="rail-title">后端事件流</span>
                <span className="rail-badge">SSE</span>
              </div>
              <div className="log-feed" ref={logFeedRef}>
                {logs.length === 0 ? (
                  <div className="log-empty">等待后端事件…</div>
                ) : (
                  logs.map((entry, index) => (
                    <div className="log-item" key={`${entry.ts}-${index}`}>
                      <span className={`log-dot ${entry.level === "error" ? "err" : entry.level === "warning" ? "accent" : "ok"}`} />
                      <span className="log-time">{timeLabel(entry.ts)}</span>
                      <span className="log-content">
                        <strong>{entry.category}</strong>
                        <span>{entry.message}</span>
                      </span>
                    </div>
                  ))
                )}
              </div>
            </section>

            <section className="rail-section rail-foot">
              <div className="rail-head">
                <span className="rail-title">当前会话</span>
                <span className="rail-badge">{reports.length} 份报告</span>
              </div>
              <div className="rail-session-id">{sessionId}</div>
            </section>
          </div>
          <div className="left-slim">
            <button
              className="slim-btn"
              title="展开系统状态与日志"
              aria-label="展开系统状态与日志"
              type="button"
              onClick={() => setLeftExpanded(true)}
            >
              <PanelLeftOpen size={18} />
            </button>
            <span className={`slim-dot ${backendConnected ? "ok" : "err"}`} />
            <button
              className="slim-btn"
              title="查看报告"
              aria-label="查看报告"
              type="button"
              onClick={() => reportListRef.current?.scrollTo({ top: 0, behavior: "smooth" })}
            >
              <FileText size={17} />
            </button>
          </div>
        </aside>

        <section id="center-col">
          <AnimatePresence initial={false}>
            {stage && (
              <Stage
                stage={stage}
                currentMediaId={stageCurrentMediaId}
                onCurrentChange={setStageCurrentMediaId}
                onClose={() => void handleCloseStage()}
              />
            )}
          </AnimatePresence>
          {layout === "focus_evidence" && latestReport && !isReportExpanded ? (
            <div className="stage-report-summary">
              <strong>{latestReport.title}</strong>
              <span>{latestReportSummary}</span>
              <button type="button" onClick={() => setIsReportExpanded(true)}>展开</button>
            </div>
          ) : (
          <div id="log-panel" className="surface-card" aria-label="报告文档区">
            <div id="log-full-view">
              <div className="panel-header">
                <div className="panel-title-wrap">
                  <span className="panel-title">报告文档</span>
                  <span className="panel-subtitle">结构化资料在这里持续累积</span>
                </div>
                <span className="report-count-pill">{reports.length} 份报告</span>
              </div>

              <div id="report-list" ref={reportListRef} onScroll={handleReportScroll}>
                <div className="report-inner">
                  {reports.length === 0 ? (
                    <div className="report-empty">
                      <div className="empty-icon">
                        <FileText size={22} strokeWidth={1.4} />
                      </div>
                      <h2>报告会出现在这里</h2>
                      <p>日常对话会在右侧以口语呈现；需要表格、代码或详细说明时，Aura 会整理成报告。</p>
                    </div>
                  ) : (
                    reports.map((report) => (
                      <article className="report-card" data-report-id={report.report_id} key={report.report_id}>
                        <div className="report-meta">
                          <div className="report-meta-left">
                            <span className="report-id">#{report.report_id}</span>
                            <span>{timeLabel(report.created_at)}</span>
                          </div>
                          {!report.complete && <span className="report-progress">正在整理</span>}
                        </div>
                        {report.query && <div className="report-source-q">来源问题：{report.query}</div>}
                        {report.title && <h2 className="report-title">{report.title}</h2>}
                        {report.markdown ? (
                          <div className="report-markdown">
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              components={
                                report.link_policy === "sources_only"
                                  ? sourcesOnlyMarkdownComponents
                                  : markdownComponents
                              }
                              urlTransform={markdownUrlTransform}
                            >
                              {report.markdown}
                            </ReactMarkdown>
                          </div>
                        ) : (
                          <div className="report-loading">正在整理内容…</div>
                        )}
                        {report.link_policy === "sources_only" && (
                          <ReportSources sources={report.sources} />
                        )}
                      </article>
                    ))
                  )}
                </div>
                {hasUnreadReports && (
                  <button className="unread-pill" type="button" onClick={scrollToLatestReport}>
                    <ArrowDown size={15} />
                    有新报告到达，点击查看
                  </button>
                )}
              </div>
            </div>
          </div>
          )}

          {attachments.length > 0 && (
            <div className="attachment-list" aria-label="待发送附件">
              {attachments.map((attachment) => (
                <div className="image-preview-chip" key={attachment.media_id}>
                  {attachment.kind === "image" && attachment.previewUrl ? (
                    <img src={attachment.previewUrl} alt="" />
                  ) : attachment.kind === "document" ? (
                    <span className="document-chip-icon">
                      <FileText size={14} />
                      文档
                    </span>
                  ) : (
                    <span className="video-chip-icon">视频</span>
                  )}
                  <span>{attachment.filename}</span>
                  <button type="button" aria-label={`移除${attachment.filename}`} onClick={() => handleRemoveAttachment(attachment.media_id)}>
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div id="input-bar">
            <input
              type="file"
              accept=".jpg,.jpeg,.png,.webp,.gif,.mp4,.webm,.pdf,image/jpeg,image/png,image/webp,image/gif,video/mp4,video/webm,application/pdf"
              multiple
              ref={fileInputRef}
              onChange={handleImageSelect}
              hidden
            />
            <button
              className="composer-icon-btn"
              type="button"
              aria-label="上传图片、视频或文档"
              title="上传图片、视频或文档"
              onClick={() => fileInputRef.current?.click()}
            >
              <ImageIcon size={19} />
            </button>
            <textarea
              id="chat-input"
              value={query}
              rows={1}
              placeholder={isListening ? "正在倾听…" : `和 ${ASSISTANT_NAME} 说点什么…`}
              onChange={handleComposerChange}
              onKeyDown={handleComposerKeyDown}
              onInput={(event) => {
                const element = event.currentTarget;
                element.style.height = "auto";
                element.style.height = `${Math.min(element.scrollHeight, 120)}px`;
              }}
            />
            <button
              className={`composer-icon-btn mic-btn ${isListening ? "active" : ""}`}
              type="button"
              aria-label={isListening ? "停止语音输入" : "语音输入"}
              title={isListening ? "停止语音输入" : "语音输入"}
              onClick={handleMicToggle}
            >
              <Mic size={19} />
            </button>
            <button
              id="send-btn"
              type="button"
              disabled={!query.trim() && attachments.length === 0}
              onClick={() => void handleSend()}
            >
              <span>{isLoading ? "发送" : "发送"}</span>
              <Send size={15} />
            </button>
          </div>
        </section>

        <aside id="right-col" aria-label={`${ASSISTANT_NAME} 角色与口语对话`}>
          <div id="avatar-card" data-state={status.state} data-avatar-loaded={isAvatarLoaded}>
            <div className="avatar-scene">
              <VRMAvatar
                onLoaded={() => setIsAvatarLoaded(true)}
                speech={speech}
                onAudioEnd={handleAudioEnd}
              />
            </div>
            <div className="avatar-vignette" />
            <div className="avatar-top-bar">
              <div className="avatar-name-pill">
                <span>{ASSISTANT_NAME}</span>
                <span className="avatar-voice-tag">· {VOICE_LABEL}</span>
              </div>
              <div className="status-pill">
                <span className="status-dot" />
                <span>{status.label}</span>
              </div>
            </div>
            {!isAvatarLoaded && <span className="avatar-loading">角色加载中</span>}
            <div id="subtitle-bar" aria-live="polite">
              <div className="subtitle-top">
                <span>实时字幕</span>
                <span className={`voice-bars ${hasQueuedSpeech ? "active" : ""}`} aria-hidden="true">
                  <i />
                  <i />
                  <i />
                </span>
              </div>
              <div id="subtitle-text" ref={subtitleTextRef}>
                {subtitle || "等待她的第一句话…"}
              </div>
            </div>
          </div>

          <div id="dialogue-panel" className="surface-card">
            <div className="panel-header">
              <div className="panel-title-wrap">
                <span className="panel-title">口语对话</span>
                <span className="panel-subtitle">字幕与对话记录</span>
              </div>
              <span className="dialogue-live"><span />实时</span>
            </div>
            <div id="dialogue-list" ref={dialogueListRef}>
              {messages.length === 0 ? (
                <div className="dialogue-empty">对话内容会显示在这里。</div>
              ) : (
                messages.map((message) => (
                  <div
                    className={`dialogue-message ${message.role} ${message.interrupted ? "interrupted" : ""}`}
                    key={message.id}
                  >
                    <div className="dialogue-meta">
                      <span>
                        {message.role === "user"
                          ? "你"
                          : message.role === "assistant"
                            ? ASSISTANT_NAME
                            : "系统"}
                      </span>
                      <time>{timeLabel(message.createdAt)}</time>
                      {message.interrupted ? (
                        <span className="interrupted-tag">已打断</span>
                      ) : null}
                    </div>
                    {(message.content || message.attachments?.length) ? (
                      <div className="dialogue-bubble">
                        {message.content && <p>{message.content}</p>}
                        {message.attachments?.map((attachment) =>
                          attachment.kind === "image" ? (
                            <img
                              className="dialogue-image"
                              key={attachment.media_id}
                              src={`${API_BASE}/api/media/${attachment.media_id}`}
                              alt={attachment.filename}
                            />
                          ) : attachment.kind === "video" ? (
                            <div className="dialogue-video" key={attachment.media_id}>
                              视频 · {attachment.filename}
                            </div>
                          ) : (
                            <div className="dialogue-document" key={attachment.media_id}>
                              <FileText size={14} />
                              <span>{attachment.filename}</span>
                            </div>
                          ),
                        )}
                      </div>
                    ) : null}
                    {message.role === "assistant" && !message.content && isLoading && (
                      <div className="dialogue-bubble thinking-bubble">正在思考…</div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
