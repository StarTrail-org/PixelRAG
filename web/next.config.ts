import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      {
        // The in-app status page was removed; status now lives on
        // independent infrastructure. Forward the old URL there.
        source: "/status",
        destination: "https://status.pixelrag.ai",
        permanent: true,
      },
    ];
  },
  async rewrites() {
    // Proxy search-backend endpoints (/api/search, /tile, /status, /health,
    // /reconstruct) to the public search API. Route handlers like /api/chat
    // take precedence over this rewrite. Uses the public host so it works on
    // Vercel (localhost only exists in local dev).
    const searchBackend =
      process.env.PIXELRAG_SEARCH_PROXY || "https://api.pixelrag.ai";
    return [
      {
        source: "/api/:path*",
        destination: `${searchBackend}/:path*`,
      },
    ];
  },
};

export default nextConfig;
