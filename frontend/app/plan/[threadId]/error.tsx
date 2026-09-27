"use client";

import { useEffect } from "react";
import Link from "next/link";

/** Next.js error boundary for this route segment. Without this, an uncaught render-time throw
 * anywhere on the plan page (a malformed field from the API, a chart choking on unexpected
 * data) took down the whole screen with the framework's own error page — no way back except
 * a manual URL edit. This keeps the header/footer chrome and offers a real way out. */
export default function PlanError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error("Plan page crashed:", error);
  }, [error]);

  return (
    <div className="mx-auto max-w-xl px-4 py-24 text-center">
      <h1 className="font-display text-2xl font-semibold text-ink">Something went wrong showing this plan</h1>
      <p className="mx-auto mt-3 max-w-md text-sm text-muted">
        The page hit an unexpected error while rendering your itinerary. Your plan is safe — this is just a display
        problem.
      </p>
      <div className="mt-6 flex items-center justify-center gap-3">
        <button
          type="button"
          onClick={reset}
          className="rounded-full bg-brand-accent px-5 py-2.5 text-sm font-medium text-white hover:bg-[#9A4B2E]"
        >
          Try again
        </button>
        <Link href="/" className="rounded-full border border-line px-5 py-2.5 text-sm font-medium text-ink hover:bg-white">
          Start a new search
        </Link>
      </div>
    </div>
  );
}
