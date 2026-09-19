import { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND = process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

/**
 * Node.js TransformStream proxy for FastAPI SSE.
 * Forwards `req.signal` so a tab close aborts the upstream graph run.
 */
export async function GET(req: NextRequest) {
  const threadId = req.nextUrl.searchParams.get("threadId");
  if (!threadId) {
    return new Response(JSON.stringify({ error: "threadId required" }), { status: 400 });
  }

  const upstream = await fetch(`${BACKEND}/api/plan/${encodeURIComponent(threadId)}/stream`, {
    headers: { Accept: "text/event-stream" },
    cache: "no-store",
    signal: req.signal,
  });

  if (!upstream.ok || !upstream.body) {
    return new Response(upstream.body, { status: upstream.status, headers: upstream.headers });
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
