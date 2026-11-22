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

export function reportTextLink({ children }) {
  return children;
}

export function ReportSources({ sources }) {
  if (!Array.isArray(sources) || sources.length === 0) {
    return createElement(
      "div",
      { className: "report-sources-empty" },
      "未找到来源，内容未经核实",
    );
  }

  const items = sources.map((source, index) => {
    const number = source?.number ?? index + 1;
    const title = String(source?.title ?? "");
    const url = typeof source?.url === "string" ? source.url : "";
    let safeUrl = false;
    try {
      const protocol = new URL(url).protocol;
      safeUrl = protocol === "http:" || protocol === "https:";
    } catch {
      safeUrl = false;
    }
    const sourceTitle = safeUrl
      ? createElement(
          "a",
          {
            href: url,
            title: url,
            target: "_blank",
            rel: "noopener noreferrer",
          },
          title,
        )
      : title;
    return createElement(
      "li",
      { key: `${number}-${index}` },
      `[${number}] `,
      sourceTitle,
    );
  });

  return createElement(
    "section",
    { className: "report-sources" },
    createElement("h3", null, "来源"),
    createElement("ol", null, ...items),
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
