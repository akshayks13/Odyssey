import Link from "next/link";
import { Wordmark } from "@/components/Wordmark";

export function Footer() {
  return (
    <footer className="border-t border-line bg-wash">
      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
          <div className="md:col-span-2">
            <Wordmark size="sm" />
            <p className="mt-3 max-w-md text-sm leading-relaxed text-muted">
              Plan trips that fit your time, budget, and pace. Destinations, routes, stays, and a
              day-by-day schedule — in one place.
            </p>
          </div>
          <div>
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-ink">Product</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/#plan" className="text-muted hover:text-ink">
                  Plan a trip
                </Link>
              </li>
              <li>
                <Link href="/#how-it-works" className="text-muted hover:text-ink">
                  How it works
                </Link>
              </li>
              <li>
                <Link href="/#features" className="text-muted hover:text-ink">
                  Features
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-ink">Resources</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <span className="text-muted">Privacy</span>
              </li>
              <li>
                <span className="text-muted">Terms</span>
              </li>
              <li>
                <span className="text-muted">Support</span>
              </li>
            </ul>
          </div>
        </div>
        <div className="mt-8 flex flex-col items-center justify-between gap-2 border-t border-line pt-6 text-sm text-muted md:flex-row">
          <p>© {new Date().getFullYear()} Odyssey. All rights reserved.</p>
          <p>Built for travellers who like a clear plan.</p>
        </div>
      </div>
    </footer>
  );
}
