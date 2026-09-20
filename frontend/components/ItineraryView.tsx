"use client";

import { useState } from "react";
import { Hotel, Itinerary } from "@/lib/types";

function formatHour(h: number): string {
  const hours = Math.floor(h);
  const minutes = Math.round((h - hours) * 60);
  const period = hours >= 12 ? "PM" : "AM";
  const displayHour = hours % 12 === 0 ? 12 : hours % 12;
  return `${displayHour}:${minutes.toString().padStart(2, "0")} ${period}`;
}

export function ItineraryView({
  itinerary,
  hotels = [],
}: {
  itinerary: Itinerary;
  hotels?: Hotel[];
}) {
  const [openDay, setOpenDay] = useState<number | null>(itinerary.days[0]?.day_number ?? null);
  const lastDay = itinerary.days[itinerary.days.length - 1]?.day_number;

  function overnightFor(dayNumber: number, destination: string, attached?: Hotel | null): Hotel | null {
    if (attached) return attached;
    if (dayNumber === lastDay) return null;
    return hotels.find((hotel) => hotel.destination === destination) || hotels[0] || null;
  }

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
        const overnight = overnightFor(day.day_number, day.destination, day.overnight_hotel);
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
                    ({day.travel_leg.mode}
                    {day.travel_leg.origin_iata ? ` ${day.travel_leg.origin_iata}→${day.travel_leg.destination_iata}` : ""}
                    , {day.travel_leg.duration_hours.toFixed(1)}h, ₹
                    {day.travel_leg.cost_inr.toLocaleString("en-IN")})
                  </span>
                )}
              </div>
              <span className="text-muted">{isOpen ? "−" : "+"}</span>
            </button>
            {isOpen && (
              <div className="divide-y divide-grey-100 border-t border-line">
                {day.travel_leg && (
                  <div className="bg-wash px-5 py-3">
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted">
                      {day.day_number === 1 ? "Arrival" : "Travel"}
                    </p>
                    <p className="mt-1 text-sm font-medium text-ink">
                      {day.travel_leg.summary ||
                        `${day.travel_leg.origin} → ${day.travel_leg.destination} · ${day.travel_leg.mode}`}
                    </p>
                    <p className="mt-0.5 text-xs text-muted">
                      {day.travel_leg.mode}
                      {day.travel_leg.origin_iata && day.travel_leg.destination_iata
                        ? ` · ${day.travel_leg.origin_iata}–${day.travel_leg.destination_iata}`
                        : ""}
                      {day.travel_leg.airline ? ` · ${day.travel_leg.airline}` : ""}
                      {` · ${day.travel_leg.duration_hours.toFixed(1)}h · ₹${day.travel_leg.cost_inr.toLocaleString("en-IN")}`}
                    </p>
                  </div>
                )}
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
                {overnight && (
                  <div className="bg-wash px-5 py-3">
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted">Overnight stay</p>
                    <p className="mt-1 text-sm font-medium text-ink">{overnight.name}</p>
                    <p className="mt-0.5 text-xs text-muted">
                      {overnight.destination}
                      {` · ${overnight.rating.toFixed(1)}★ · ₹${overnight.price_per_night_inr.toLocaleString("en-IN")}/night`}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
