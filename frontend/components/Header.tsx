"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Compass, Map, Menu, Sparkles, X } from "lucide-react";
import { useState } from "react";
import { Wordmark } from "@/components/Wordmark";
import { cn } from "@/lib/cn";

const links = [
  { name: "Home", href: "/", icon: Compass },
  { name: "How it works", href: "/#how-it-works", icon: Map },
  { name: "Features", href: "/#features", icon: Sparkles },
];

export function Header() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-line bg-paper/90 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center" onClick={() => setOpen(false)}>
          <Wordmark size="sm" />
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {links.map((item) => {
            const active = item.href === "/" ? pathname === "/" : false;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "flex items-center gap-2 rounded-full px-3.5 py-2 text-sm font-medium transition-colors",
                  active
                    ? "bg-selected text-brand-primary"
                    : "text-muted hover:bg-wash hover:text-ink"
                )}
              >
                <item.icon className="h-4 w-4" />
                {item.name}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-2">
          <Link
            href="/#plan"
            className="hidden rounded-full bg-brand-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[#9A4B2E] md:inline-flex"
          >
            Plan a trip
          </Link>
          <button
            type="button"
            className="rounded-full p-2 text-muted hover:bg-wash md:hidden"
            aria-label={open ? "Close menu" : "Open menu"}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {open && (
        <div className="border-t border-line px-4 py-3 md:hidden">
          <nav className="flex flex-col gap-1">
            {links.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                onClick={() => setOpen(false)}
                className="flex items-center gap-2 rounded-full px-3 py-2 text-sm font-medium text-muted hover:bg-wash hover:text-ink"
              >
                <item.icon className="h-4 w-4" />
                {item.name}
              </Link>
            ))}
            <Link
              href="/#plan"
              onClick={() => setOpen(false)}
              className="mt-2 rounded-full bg-brand-accent px-4 py-2 text-center text-sm font-medium text-white"
            >
              Plan a trip
            </Link>
          </nav>
        </div>
      )}
    </header>
  );
}
