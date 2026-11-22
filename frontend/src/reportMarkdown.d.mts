import type { Components, UrlTransform } from "react-markdown";
import type { ReactNode } from "react";

export const markdownUrlTransform: UrlTransform;
export const reportLink: Components["a"];
export const reportTextLink: Components["a"];
export const reportImage: Components["img"];
export function ReportSources(props: {
  sources?: Array<{ number: number; title: string; url: string }> | null;
}): ReactNode;
