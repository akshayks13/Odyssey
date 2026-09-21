const BACKEND = process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  // The browser only ever talks to this server; Next forwards /api/* to the backend. That keeps every call
  // same-origin, so it does not matter whether the app is opened as localhost, 127.0.0.1 or a LAN address
  // (a direct browser call to :8000 is blocked by CORS on any origin the backend does not list).
  // The live event stream is app/api/stream/route.ts, a real route, which takes precedence over this rewrite.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${BACKEND}/api/:path*` }];
  },
};

export default nextConfig;
