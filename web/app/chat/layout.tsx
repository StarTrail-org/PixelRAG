import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Ask",
  description:
    "Ask anything about Wikipedia — PixelRAG searches screenshot tiles, reads them, and answers with citations.",
}

export default function ChatLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
