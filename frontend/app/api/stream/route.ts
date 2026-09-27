import { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND = process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

function jsonError(status: number, detail: string) {
  return new Response(JSON.stringify({ detail }), { status, headers: { "Content-Type": "application/json" } });
}

/** Proxy FastAPI SSE; abort upstream when the tab closes. */
export async function GET(req: NextRequest) {
  const threadId = req.nextUrl.searchParams.get("threadId");
  if (!threadId) {
    return jsonError(400, "threadId required");
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND}/api/plan/${encodeURIComponent(threadId)}/stream`, {
      headers: { Accept: "text/event-stream" },
      cache: "no-store",
      signal: req.signal,
    });
  } catch {
    // The backend is down or unreachable. An unhandled rejection here used to become a bare
    // Next.js 500 with an empty body; the client's fetch would still resolve with `!res.ok`
    // and no useful message, so give it a real status and a message worth showing.
    return jsonError(502, "Could not reach the planning server. It may be restarting — try again in a moment.");
  }

  if (!upstream.ok || !upstream.body) {
    // Read the upstream's own error detail rather than forwarding its headers verbatim — a
    // stale content-length/content-encoding on a re-wrapped response body can corrupt it.
    let detail = `Upstream returned ${upstream.status}`;
    try {
      const text = await upstream.text();
      const parsed = text && JSON.parse(text);
      if (parsed?.detail) detail = String(parsed.detail);
    } catch {
      /* keep the generic message */
    }
    return jsonError(upstream.status || 502, detail);
  }

  const { readable, writable } = new TransformStream();
  upstream.body.pipeTo(writable).catch(() => {
    /* client disconnect / abort */
  });

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
