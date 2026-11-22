import { createElement } from "react";
import { defaultUrlTransform } from "react-markdown";
import { isMediaRef, mediaUrl } from "./mediaHelpers.js";

export const markdownUrlTransform = (url) =>
  isMediaRef(url) ? url : defaultUrlTransform(url);

export function reportLink({ children, href }) {
  return createElement(
    "a",
    {
      className: "report-link",
      href,
      title: href,
      target: "_blank",
      rel: "noreferrer",
    },
    children,
  );
}

export function reportImage({ alt, src }) {
  if (!isMediaRef(src ?? "")) {
    return createElement(
      "span",
      { className: "report-image-placeholder" },
      `［图片：${alt || "未命名"}］`,
    );
  }
  return createElement("img", {
    className: "report-media-image",
    src: mediaUrl(src.slice("media:".length)),
    alt: alt || "",
    loading: "lazy",
  });
}
