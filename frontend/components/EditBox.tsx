"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { MessageSquare, Send } from "lucide-react";
import { cn } from "@/lib/cn";

export interface ChatTurn {
  role: "user" | "assistant";
  text: string;
}

const SUGGESTIONS = ["Make day 2 lighter", "Cheaper hotels", "Add another city", "Why these cities?"];

export function EditBox({
  turns,
  disabled,
  onSend,
}: {
  turns: ChatTurn[];
  disabled: boolean;
  onSend: (message: string) => void;
}) {
  const [text, setText] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [turns.length]);

  function submit(event?: FormEvent) {
    event?.preventDefault();
    const value = text.trim();
    if (!value || disabled) return;
    onSend(value);
    setText("");
  }

  return (
    <div className="rounded-3xl border border-line bg-white p-5">
      <h3 className="flex items-center gap-2 font-semibold text-ink">
        <MessageSquare className="h-4 w-4 text-brand-primary" />
        Change anything
      </h3>
      <p className="mt-1 text-sm text-muted">Say what you want. Only the parts that need to change are rebuilt.</p>

      {turns.length > 0 && (
        <div className="mt-4 max-h-64 space-y-3 overflow-y-auto pr-1">
          {turns.map((turn, i) => (
            <div key={i} className={cn("flex", turn.role === "user" ? "justify-end" : "justify-start")}>
              <p
                className={cn(
                  "max-w-[90%] rounded-2xl px-4 py-2.5 text-sm",
                  turn.role === "user" ? "bg-brand-primary text-white" : "bg-wash text-ink"
                )}
              >
                {turn.text}
              </p>
            </div>
          ))}
          <div ref={endRef} />
        </div>
      )}

      <form onSubmit={submit} className="mt-4 flex items-end gap-2">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={2}
          disabled={disabled}
          placeholder={disabled ? "Working on it…" : "e.g. day 3 is too packed, or swap one city for another"}
          className="min-h-[3rem] flex-1 resize-none rounded-2xl border border-line bg-white px-4 py-2.5 text-sm text-ink placeholder:text-grey-500 focus:border-brand-primary focus:outline-none disabled:bg-wash"
        />
        <button
          type="submit"
          disabled={disabled || !text.trim()}
          className="inline-flex h-11 items-center gap-2 rounded-full bg-brand-accent px-5 text-sm font-medium text-white transition-colors hover:bg-[#9A4B2E] disabled:opacity-40"
        >
          <Send className="h-4 w-4" />
          Send
        </button>
      </form>

      <div className="mt-3 flex flex-wrap gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            disabled={disabled}
            onClick={() => setText(s)}
            className="rounded-full border border-line bg-white px-3 py-1 text-xs text-muted hover:border-grey-400 hover:bg-grey-50 disabled:opacity-50"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
