import { AnimatePresence, motion } from "framer-motion";
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";

export type StageItem = {
  media_id: string;
  kind: "image" | "video";
  mime: string;
  src: string;
  caption: string;
};

export type MediaStageData = {
  caption: string;
  focus: "normal" | "primary";
  items: StageItem[];
};

type Props = {
  stage: MediaStageData | null;
  currentMediaId: string | null;
  onCurrentChange: (mediaId: string) => void;
  onClose: () => void;
};

export function Stage({ stage, currentMediaId, onCurrentChange, onClose }: Props) {
  const [index, setIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [enlarged, setEnlarged] = useState(false);
  const items = stage?.items ?? [];
  const item = items[index];

  useEffect(() => {
    const currentIndex = items.findIndex((media) => media.media_id === currentMediaId);
    setIndex(currentIndex >= 0 ? currentIndex : 0);
    setLoading(true);
    setFailed(false);
  }, [currentMediaId, items]);

  const navigate = (nextIndex: number) => {
    const normalized = (nextIndex + items.length) % items.length;
    setIndex(normalized);
    setLoading(true);
    setFailed(false);
    setEnlarged(false);
    onCurrentChange(items[normalized].media_id);
  };

  useEffect(() => {
    if (!stage) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setEnlarged(false);
      else if (items.length > 1 && event.key === "ArrowLeft") navigate(index - 1);
      else if (items.length > 1 && event.key === "ArrowRight") navigate(index + 1);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [index, items, stage]);

  const mediaUrl = useMemo(
    () => (item ? `${item.src}${item.src.includes("?") ? "&" : "?"}v=${item.media_id}` : ""),
    [item],
  );
  if (!stage || !item) return null;

  const media = (
    <>
      {loading && !failed && <div className="stage-loading">正在加载媒体…</div>}
      {failed ? (
        <div className="stage-error">
          <strong>无法加载这份媒体</strong>
          <span>{item.caption || stage.caption}</span>
        </div>
      ) : item.kind === "video" ? (
        <video
          key={item.media_id}
          src={mediaUrl}
          controls
          preload="metadata"
          playsInline
          onLoadedData={() => setLoading(false)}
          onError={() => {
            setLoading(false);
            setFailed(true);
          }}
        />
      ) : (
        <img
          key={item.media_id}
          src={mediaUrl}
          alt={item.caption}
          onLoad={() => setLoading(false)}
          onError={() => {
            setLoading(false);
            setFailed(true);
          }}
          onClick={() => setEnlarged((value) => !value)}
        />
      )}
    </>
  );

  return (
    <motion.section
      className={`evidence-stage ${stage.focus === "primary" ? "is-primary" : ""}`}
      layout
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.22 }}
      aria-label="媒体展示台"
    >
      <div className="stage-head">
        <span className="stage-caption">{stage.caption || item.caption || "展示台"}</span>
        <button type="button" className="stage-close" onClick={onClose} aria-label="关闭展示台">
          <X size={16} />
        </button>
      </div>
      <div className="stage-media-wrap">
        <AnimatePresence mode="wait">
          <motion.div
            key={item.media_id}
            className="stage-media"
            initial={{ opacity: 0.6 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0.6 }}
            transition={{ duration: 0.18 }}
          >
            {media}
          </motion.div>
        </AnimatePresence>
        {items.length > 1 && (
          <>
            <button className="stage-arrow previous" type="button" onClick={() => navigate(index - 1)} aria-label="上一项">
              <ChevronLeft size={18} />
            </button>
            <button className="stage-arrow next" type="button" onClick={() => navigate(index + 1)} aria-label="下一项">
              <ChevronRight size={18} />
            </button>
            <div className="stage-dots">
              {items.map((mediaItem, itemIndex) => (
                <button
                  key={mediaItem.media_id}
                  type="button"
                  className={itemIndex === index ? "active" : ""}
                  aria-label={`第 ${itemIndex + 1} 项`}
                  onClick={() => navigate(itemIndex)}
                />
              ))}
            </div>
          </>
        )}
      </div>
      {enlarged &&
        item.kind === "image" &&
        !failed &&
        createPortal(
          <div className="stage-enlarged" role="dialog" aria-label={item.caption} onClick={() => setEnlarged(false)}>
            <img src={mediaUrl} alt={item.caption} />
          </div>,
          document.getElementById("center-col") ?? document.body,
        )}
    </motion.section>
  );
}
