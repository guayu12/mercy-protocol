import assert from "node:assert/strict";
import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { renderToStaticMarkup } from "react-dom/server";
import { API_BASE, mediaUrl } from "../src/mediaHelpers.js";
import {
  ReportSources,
  markdownUrlTransform,
  reportImage,
  reportLink,
  reportTextLink,
} from "../src/reportMarkdown.mjs";

function renderMarkdown(markdown, { linkPolicy = "open", sources } = {}) {
  const components =
    linkPolicy === "sources_only"
      ? { a: reportTextLink, img: reportImage }
      : { a: reportLink, img: reportImage };
  const children = [
    React.createElement(
      ReactMarkdown,
      {
        remarkPlugins: [remarkGfm],
        components,
        urlTransform: markdownUrlTransform,
      },
      markdown,
    ),
  ];
  if (linkPolicy === "sources_only") {
    children.push(React.createElement(ReportSources, { sources }));
  }
  return renderToStaticMarkup(
    React.createElement(React.Fragment, null, ...children),
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
assert.equal((linkMarkup.match(/<a\b/g) ?? []).length, 1);

const sourcesOnlyBodies = [
  "[x]( //h)",
  "[x](\n//h)",
  "[x](<//h>)",
  String.raw`[x](\/\/h)`,
  "[x](&#47;&#47;h)",
  "[x](http&#58;//h)",
  "[x](https&colon;//h)",
  String.raw`[x](https\://h)`,
  "[x][1]\n\n[1]:\n//h",
  "[x][1]\n\n[1]: http&#58;//h",
  "www.x.test/a",
  "https://x.test/a",
];
for (const body of sourcesOnlyBodies) {
  const markup = renderMarkdown(body, {
    linkPolicy: "sources_only",
    sources: [],
  });
  assert.equal(
    (markup.match(/<a\b/g) ?? []).length,
    0,
    `sources-only body rendered a link: ${JSON.stringify(body)}`,
  );
}

const sourceMarkup = renderMarkdown("", {
  linkPolicy: "sources_only",
  sources: [
    {
      number: 1,
      title: "C](//evil.test) 结尾",
      url: "https://c.test",
    },
  ],
});
assert.equal((sourceMarkup.match(/<a\b/g) ?? []).length, 1);
assert.ok(sourceMarkup.includes('href="https://c.test"'));
assert.ok(sourceMarkup.includes('title="https://c.test"'));
assert.ok(sourceMarkup.includes('target="_blank"'));
assert.ok(sourceMarkup.includes('rel="noopener noreferrer"'));
assert.ok(sourceMarkup.includes("C](//evil.test) 结尾"));
assert.ok(!sourceMarkup.includes('href="//evil.test"'));

const unsafeSourceMarkup = renderMarkdown("", {
  linkPolicy: "sources_only",
  sources: [{ number: 1, title: "不安全链接", url: "javascript:alert(1)" }],
});
assert.equal((unsafeSourceMarkup.match(/<a\b/g) ?? []).length, 0);

const emptySourcesMarkup = renderMarkdown("", {
  linkPolicy: "sources_only",
  sources: [],
});
assert.ok(emptySourcesMarkup.includes("未找到来源，内容未经核实"));

console.log("media rendering checks passed");
