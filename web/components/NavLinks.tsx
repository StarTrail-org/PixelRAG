"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { ExternalLink } from "lucide-react"

const links = [
  { href: "/", label: "Search" },
  { href: "/chat", label: "Agent" },
  { href: "/docs", label: "API Docs" },
]

export function NavLinks() {
  const pathname = usePathname()

  return (
    <div className="flex items-center gap-3 sm:gap-6">
      {links.map(({ href, label }) => {
        const isActive =
          href === "/" ? pathname === "/" : pathname.startsWith(href)
        return (
          <Link
            key={href}
            href={href}
            className={`whitespace-nowrap text-sm transition-colors hover:text-foreground ${
              isActive
                ? "nav-link-active text-foreground"
                : "text-muted-foreground"
            }`}
          >
            {label}
          </Link>
        )
      })}
      {/* Status page is hosted independently (GitHub Pages) so it survives
          outages of this app — industry best practice (cf. githubstatus.com) */}
      <a
        href="https://status.pixelrag.ai"
        target="_blank"
        rel="noopener noreferrer"
        className="hidden items-center gap-1 whitespace-nowrap text-sm text-muted-foreground transition-colors hover:text-foreground sm:flex"
      >
        Status
        <ExternalLink className="h-3 w-3" />
      </a>
    </div>
  )
}
