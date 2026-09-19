"use client";

import { useState } from "react";
import { Itinerary } from "@/lib/types";

function formatHour(h: number): string {
  const hours = Math.floor(h);
  const minutes = Math.round((h - hours) * 60);
  const period = hours >= 12 ? "PM" : "AM";
  const displayHour = hours % 12 === 0 ? 12 : hours % 12;
  return `${displayHour}:${minutes.toString().padStart(2, "0")} ${period}`;
}

export function ItineraryView({ itinerary }: { itinerary: Itinerary }) {
  const [openDay, setOpenDay] = useState<number | null>(itinerary.days[0]?.day_number ?? null);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between rounded-3xl bg-white px-5 py-4 shadow-card">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-muted">Fit score</p>
          <p className="text-2xl font-semibold text-ink">{(itinerary.optimization_score * 100).toFixed(0)}%</p>
        </div>
        <div className="text-right">
          <p className="text-xs font-medium uppercase tracking-wide text-muted">Estimated total</p>
          <p className="text-2xl font-semibold text-brand-primary">
            ₹{itinerary.total_cost_inr.toLocaleString("en-IN")}
          </p>
        </div>
      </div>

      {itinerary.days.map((day) => {
        const isOpen = openDay === day.day_number;
        return (
          <div key={day.day_number} className="overflow-hidden rounded-3xl border border-line bg-white">
            <button
              onClick={() => setOpenDay(isOpen ? null : day.day_number)}
              className="flex w-full items-center justify-between bg-white px-5 py-4 text-left hover:bg-grey-50"
            >
              <div>
                <span className="font-semibold text-ink">Day {day.day_number}</span>
                <span className="ml-2 text-muted">— {day.destination}</span>
                {day.travel_leg && (
                  <span className="ml-2 text-xs text-grey-500">
                    ({day.travel_leg.mode}, {day.travel_leg.duration_hours.toFixed(1)}h, ₹
                    {day.travel_leg.cost_inr.toLocaleString("en-IN")})
                  </span>
                )}
              </div>
              <span className="text-muted">{isOpen ? "−" : "+"}</span>
            </button>
            {isOpen && (
              <div className="divide-y divide-grey-100 border-t border-line">
                {day.items.length === 0 && (
                  <p className="px-5 py-3 text-sm text-muted">Free day / rest day.</p>
                )}
                {day.items.map((item) => (
                  <div key={item.activity_id} className="flex items-center gap-4 px-5 py-3">
                    <span className="w-32 shrink-0 text-sm font-medium text-muted">
                      {formatHour(item.start_hour)} – {formatHour(item.end_hour)}
                    </span>
                    <span className="flex-1 text-ink">{item.activity_name}</span>
                    <span className="rounded-full bg-selected px-2 py-0.5 text-xs text-brand-primary">
                      {item.category}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
