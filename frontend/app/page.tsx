"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Calendar,
  Compass,
  MapPin,
  Search,
  Sparkles,
  Wallet,
} from "lucide-react";
import { Wordmark } from "@/components/Wordmark";
import { openSample } from "@/lib/api";

const EXAMPLES = [
  "5 days in Kerala with 3 friends, around ₹40,000, nature and adventure at a relaxed pace",
  "From Chennai to Delhi for 4 days, budget ₹50,000, food and culture",
  "4-day Kerala trip for 2, by train, budget ₹35,000, beach and slow mornings",
];

const features = [
  {
    icon: MapPin,
    title: "Places that match you",
    description: "We rank cities and activities against what you actually like — hills, food, beaches, culture.",
  },
  {
    icon: Compass,
    title: "A sensible route",
    description: "We pick road, train, or flight between stops. Say “by train” if you want to choose.",
  },
  {
    icon: Wallet,
    title: "Budget you can trust",
    description: "Stays, meals, activities, and transport are itemized against your ceiling — no surprise totals.",
  },
  {
    icon: Calendar,
    title: "A day-by-day plan",
    description: "Dated days with opening hours, travel time, meals, and the way there and back — a schedule you can follow.",
  },
];

const steps = [
  {
    step: "1",
    title: "Describe the trip",
    description: "Tell us where, how long, who’s going, and what you care about — in plain language.",
  },
  {
    step: "2",
    title: "Review the plan",
    description: "See destinations, a route, hotels, costs, and a timed itinerary in one view.",
  },
  {
    step: "3",
    title: "Change anything, just ask",
    description: "Say “make day 2 lighter” or “add another city”. We rebuild only what needs to change — and you can undo it.",
  },
];

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0 },
};

export default function Home() {
  const router = useRouter();
  const [message, setMessage] = useState("");
  const [sampleError, setSampleError] = useState(false);

  function handleSubmit(text: string) {
    if (!text.trim()) return;
    const threadId = crypto.randomUUID();
    sessionStorage.setItem(`odyssey:${threadId}:message`, text);
    router.push(`/plan/${threadId}`);
  }

  async function showSample() {
    try {
      router.push(`/plan/${await openSample()}`);
    } catch {
      setSampleError(true);
    }
  }

  function onSearch(event: FormEvent) {
    event.preventDefault();
    handleSubmit(message);
  }

  return (
    <div className="overflow-hidden bg-paper">
      <section className="relative pb-20 pt-16 sm:pb-28 sm:pt-24" id="plan">
        <div
          aria-hidden
          className="absolute inset-0 -z-10"
          style={{
            backgroundImage: "radial-gradient(rgba(28,25,23,0.08) 1px, transparent 1px)",
            backgroundSize: "22px 22px",
            WebkitMaskImage: "radial-gradient(ellipse 70% 60% at 50% 35%, #000 40%, transparent 100%)",
            maskImage: "radial-gradient(ellipse 70% 60% at 50% 35%, #000 40%, transparent 100%)",
          }}
        />
        <motion.div
          aria-hidden
          className="absolute -left-10 top-10 -z-10 h-72 w-72 rounded-full blur-3xl"
          style={{ background: "radial-gradient(circle, rgba(30,92,85,0.28), transparent 70%)" }}
          animate={{ y: [0, 30, 0], x: [0, 20, 0] }}
          transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          aria-hidden
          className="absolute right-0 top-24 -z-10 h-72 w-72 rounded-full blur-3xl"
          style={{ background: "radial-gradient(circle, rgba(184,92,56,0.18), transparent 70%)" }}
          animate={{ y: [0, -25, 0], x: [0, -15, 0] }}
          transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
        />

        <div className="mx-auto max-w-3xl px-4 text-center sm:px-6">
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 rounded-full border border-line bg-paper/80 px-4 py-1.5 text-sm font-medium text-muted shadow-sm backdrop-blur"
          >
            <Sparkles className="h-4 w-4 text-brand-primary" />
            Travel planning, made simple
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-10"
          >
            <Wordmark size="lg" />
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08 }}
            className="mx-auto mt-5 max-w-xl text-base leading-relaxed text-muted md:text-lg"
          >
            Destinations, a route, stays, and a schedule — fitted to your time and budget.
          </motion.p>

          <motion.form
            onSubmit={onSearch}
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mt-10 rounded-search border border-line bg-white px-5 py-3 text-left shadow-search transition-shadow focus-within:border-brand-primary/40 focus-within:shadow-search-hover hover:shadow-search-hover"
          >
            <div className="flex items-start gap-3">
              <Search className="mt-2 h-5 w-5 shrink-0 text-muted" />
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(message);
                  }
                }}
                rows={2}
                placeholder="Where to? Days, people, budget, and what you enjoy…"
                className="min-h-[3.25rem] w-full resize-none bg-transparent text-base text-ink placeholder:text-grey-500 focus:outline-none"
              />
            </div>
            <div className="mt-2 flex justify-end">
              <button
                type="submit"
                disabled={!message.trim()}
                className="inline-flex items-center gap-2 rounded-full bg-brand-accent px-5 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-[#9A4B2E] disabled:opacity-40"
              >
                Search
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
          </motion.form>

          <div className="mt-6 flex flex-wrap justify-center gap-2">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => handleSubmit(example)}
                className="max-w-full rounded-full border border-line bg-white px-3.5 py-1.5 text-left text-xs text-muted transition-colors hover:border-grey-400 hover:bg-grey-50 md:text-sm"
              >
                {example}
              </button>
            ))}
          </div>
          <button type="button" onClick={showSample} className="mt-4 text-sm font-medium text-brand-primary hover:underline">
            {sampleError ? "Sample trip unavailable" : "Or see a sample trip"}
          </button>
        </div>
      </section>

      <section className="border-y border-line bg-wash" id="features">
        <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
          <div className="mb-12 text-center">
            <h2 className="font-display text-3xl font-semibold tracking-tight text-ink md:text-4xl">What you get</h2>
            <p className="mx-auto mt-3 max-w-xl text-muted">
              A complete itinerary — not a list of links. Built around your preferences and your
              budget.
            </p>
          </div>
          <motion.div
            variants={{ show: { transition: { staggerChildren: 0.1 } } }}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4"
          >
            {features.map((feature) => (
              <motion.div
                key={feature.title}
                variants={fadeUp}
                whileHover={{ y: -6 }}
                className="group rounded-3xl border border-line bg-paper p-7 shadow-sm transition-shadow hover:shadow-card"
              >
                <div className="inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-selected text-brand-primary">
                  <feature.icon className="h-7 w-7" />
                </div>
                <h3 className="mt-5 text-lg font-semibold text-ink">{feature.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{feature.description}</p>
                <div className="mt-5 h-1 w-0 rounded-full bg-brand-primary transition-all duration-300 group-hover:w-12" />
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      <section className="bg-paper py-20" id="how-it-works">
        <div className="mx-auto max-w-6xl px-4 sm:px-6">
          <div className="mb-14 text-center">
            <h2 className="font-display text-3xl font-semibold tracking-tight text-ink md:text-4xl">How it works</h2>
            <p className="mx-auto mt-3 max-w-xl text-muted">Three steps from a sentence to a plan you can travel on.</p>
          </div>
          <div className="relative grid grid-cols-1 gap-10 md:grid-cols-3">
            <div className="absolute left-[16%] right-[16%] top-8 hidden h-px bg-line md:block" />
            {steps.map((step, index) => (
              <motion.div
                key={step.step}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.12 }}
                className="relative text-center"
              >
                <div className="relative z-10 mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-brand-primary text-lg font-semibold text-white">
                  {step.step}
                </div>
                <h3 className="mt-6 text-xl font-semibold text-ink">{step.title}</h3>
                <p className="mx-auto mt-2 max-w-xs text-muted">{step.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section id="pricing" className="relative overflow-hidden bg-brand-primary py-20">
        <div className="relative mx-auto max-w-3xl px-4 text-center sm:px-6">
          <h2 className="font-display text-3xl font-semibold tracking-tight text-white md:text-4xl">Ready when you are</h2>
          <p className="mx-auto mt-3 max-w-xl text-lg text-white/85">
            Free to try. Describe a trip and get a full itinerary in a couple of minutes.
          </p>
          <a
            href="#plan"
            className="mt-8 inline-flex items-center gap-2 rounded-full bg-white px-8 py-3.5 text-base font-medium text-brand-primary transition-transform hover:scale-[1.03]"
          >
            Start planning
            <ArrowRight className="h-5 w-5" />
          </a>
        </div>
      </section>
    </div>
  );
}
