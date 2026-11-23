export const API_BASE = "http://127.0.0.1:8000";

const MEDIA_REF = /^media:m_[A-Za-z0-9_-]+$/;

export function isMediaRef(value) {
  return MEDIA_REF.test(value);
}

export function mediaUrl(mediaId) {
  return `${API_BASE}/api/media/${encodeURIComponent(mediaId)}`;
}
