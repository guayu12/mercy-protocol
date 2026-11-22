import assert from "node:assert/strict";
import React from "react";
import ReactMarkdown from "react-markdown";
import { renderToStaticMarkup } from "react-dom/server";
import { API_BASE, mediaUrl } from "../src/mediaHelpers.js";
import {
  markdownUrlTransform,
  reportImage,
  reportLink,
} from "../src/reportMarkdown.mjs";

function renderMarkdown(markdown) {
  return renderToStaticMarkup(
    React.createElement(
      ReactMarkdown,
      {
        components: { a: reportLink, img: reportImage },
        urlTransform: markdownUrlTransform,
      },
      markdown,
    ),
  );
}

const validMarkup = renderMarkdown("![图](media:m_abc_123)");
assert.ok(
  validMarkup.includes(`src="${API_BASE}/api/media/m_abc_123"`),
  "registered media refs should render from the backend",
);

const blockedMarkup = renderMarkdown(
  "![恶意](media:m_x/../a) ![外部](https://example.com/image.png)",
);
assert.ok(!blockedMarkup.includes("<img"), "untrusted images must not render");
assert.ok(blockedMarkup.includes("［图片：恶意］"));
assert.ok(blockedMarkup.includes("［图片：外部］"));
assert.equal(mediaUrl("m_abc_123"), `${API_BASE}/api/media/m_abc_123`);

const linkMarkup = renderMarkdown("[报告来源](https://example.com/report)");
assert.ok(
  linkMarkup.includes('title="https://example.com/report"'),
  "report links should expose their target URL as the title",
);

console.log("media rendering checks passed");
