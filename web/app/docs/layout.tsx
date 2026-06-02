import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "API Reference",
  description:
    "PixelRAG HTTP API — search a hosted index of 8.28M Wikipedia screenshot tiles by text or image, and fetch the matching screenshots.",
}

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
