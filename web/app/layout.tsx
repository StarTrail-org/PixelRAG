import type { Metadata } from "next"
import { Inter, Crimson_Pro, JetBrains_Mono } from "next/font/google"

import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { cn } from "@/lib/utils"
import { NavLinks } from "@/components/NavLinks"

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" })
const crimsonPro = Crimson_Pro({ subsets: ["latin"], variable: "--font-display" })
const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
})

export const metadata: Metadata = {
  metadataBase: new URL("https://pixelrag.ai"),
  title: {
    default: "PixelRAG — visual Wikipedia search",
    template: "%s · PixelRAG",
  },
  description:
    "Search Wikipedia by how it looks, not just the text it contains. PixelRAG renders pages as screenshots and retrieves over the images.",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn(
        "antialiased",
        inter.variable,
        crimsonPro.variable,
        jetbrainsMono.variable,
        "font-sans"
      )}
    >
      <body>
        <ThemeProvider>
          <nav className="sticky top-0 z-50 bg-background/70 backdrop-blur-xl">
            <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
              {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
              <a href="/" className="group flex items-center gap-2.5">
                {/* Logo mark — PixelRAG cat (inverts to white in dark mode) */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/logo.png"
                  alt=""
                  className="h-9 w-auto transition-transform duration-300 group-hover:scale-105 dark:invert"
                />
                <span className="flex items-baseline gap-0.5">
                  <span className="font-display text-xl font-semibold tracking-tight">
                    Pixel
                  </span>
                  <span className="font-display text-xl font-semibold tracking-tight text-primary">
                    RAG
                  </span>
                </span>
              </a>
              <div className="flex items-center gap-4">
                <NavLinks />
                <a
                  href="https://github.com/StarTrail-org/PixelRAG"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="GitHub repository"
                  className="text-muted-foreground transition-colors hover:text-foreground"
                >
                  <svg
                    viewBox="0 0 16 16"
                    width="20"
                    height="20"
                    fill="currentColor"
                    aria-hidden="true"
                  >
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
                  </svg>
                </a>
              </div>
            </div>
            {/* Gradient fade bottom border */}
            <div className="h-px bg-gradient-to-r from-transparent via-border to-transparent" />
          </nav>
          <main>{children}</main>
        </ThemeProvider>
      </body>
    </html>
  )
}
