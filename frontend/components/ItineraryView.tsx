"use client";

import { useState } from "react";
import { Bed, Bus, Car, Coffee, CloudRain, Sun, TrainFront, Utensils } from "lucide-react";
import { Hotel, Itinerary, ItineraryDay, RouteLeg } from "@/lib/types";
import { cn } from "@/lib/cn";

function formatHour(h: number): string {
  const hours = Math.floor(h);
  const minutes = Math.round((h - hours) * 60);
  const period = hours >= 12 ? "PM" : "AM";
  const displayHour = hours % 12 === 0 ? 12 : hours % 12;
  return `${displayHour}:${minutes.toString().padStart(2, "0")} ${period}`;
}

function formatDate(iso?: string | null): string | null {
  if (!iso) return null;
  const d = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString("en-IN", { weekday: "short", day: "numeric", month: "short" });
}

const KIND_LABEL: Record<string, string> = { travel: "Travel day", leisure: "Free time" };

function LegIcon({ leg }: { leg: RouteLeg }) {
  const className = "h-3.5 w-3.5";
  if (leg.mode === "rail") return <TrainFront className={className} />;
  return leg.vehicle === "bus" ? <Bus className={className} /> : <Car className={className} />;
}

function LegBlock({ leg, label }: { leg: RouteLeg; label: string }) {
  return (
    <div className="bg-wash px-5 py-3">
      <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted">
        <LegIcon leg={leg} />
        {label}
      </p>
      <p className="mt-1 text-sm font-medium text-ink">
        {leg.summary || `${leg.origin} → ${leg.destination} · ${leg.mode}`}
      </p>
      <p className="mt-0.5 text-xs text-muted">
        {leg.mode}
        {` · ${leg.duration_hours.toFixed(1)}h · ₹${leg.cost_inr.toLocaleString("en-IN")}`}
      </p>
    </div>
  );
}

export function ItineraryView({ itinerary, hotels = [] }: { itinerary: Itinerary; hotels?: Hotel[] }) {
  const [openDay, setOpenDay] = useState<number | null>(itinerary.days[0]?.day_number ?? null);
  const lastDay = itinerary.days[itinerary.days.length - 1]?.day_number;

  function overnightFor(day: ItineraryDay): Hotel | null {
    if (day.overnight_hotel) return day.overnight_hotel;
    if (day.day_number === lastDay) return null;
    return hotels.find((hotel) => hotel.destination === day.destination) || null;
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
        const overnight = overnightFor(day);
        const date = formatDate(day.date);
        const kindLabel = day.kind ? KIND_LABEL[day.kind] : undefined;
        return (
          <div key={day.day_number} className="overflow-hidden rounded-3xl border border-line bg-white">
            <button
              onClick={() => setOpenDay(isOpen ? null : day.day_number)}
              className="flex w-full items-center justify-between gap-3 bg-white px-5 py-4 text-left hover:bg-grey-50"
            >
              <div className="min-w-0">
                <span className="font-semibold text-ink">Day {day.day_number}</span>
                {date && <span className="ml-2 text-sm text-muted">{date}</span>}
                <span className="ml-2 text-muted">— {day.destination}</span>
                {day.weather && (
                  <span
                    title="Typical weather for that month"
                    className={cn(
                      "ml-2 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium",
                      day.weather.rainy ? "bg-[#e6eef2] text-[#2f5d75]" : "bg-wash text-muted"
                    )}
                  >
                    {day.weather.rainy ? <CloudRain className="h-3 w-3" /> : <Sun className="h-3 w-3" />}
                    {day.weather.rainy && day.weather.rain_chance != null ? `Rain ${day.weather.rain_chance}%` : day.weather.summary}
                  </span>
                )}
                {kindLabel && (
                  <span className="ml-2 rounded-full bg-wash px-2 py-0.5 text-xs font-medium text-muted">{kindLabel}</span>
                )}
              </div>
              <span className="text-muted">{isOpen ? "−" : "+"}</span>
            </button>
            {isOpen && (
              <div className="divide-y divide-grey-100 border-t border-line">
                {day.travel_leg && <LegBlock leg={day.travel_leg} label={day.day_number === 1 ? "Arrival" : "Travel"} />}
                {day.note && !day.travel_leg && (
                  <p className="bg-wash px-5 py-3 text-sm text-muted">{day.note}</p>
                )}
                {day.items.length === 0 && (
                  <p className="px-5 py-3 text-sm text-muted">
                    {day.kind === "travel" ? "Most of today is spent getting there." : "Free day — no set plans."}
                  </p>
                )}
                {day.items.map((item, i) => {
                  const isMeal = item.kind === "meal";
                  return (
                    <div key={`${item.activity_id}-${i}`} className={cn("flex items-center gap-4 px-5 py-3", isMeal && "bg-paper/60")}>
                      <span className="w-32 shrink-0 text-sm font-medium text-muted">
                        {formatHour(item.start_hour)} – {formatHour(item.end_hour)}
                      </span>
                      <span className={cn("flex flex-1 items-center gap-2", isMeal ? "text-muted" : "text-ink")}>
                        {isMeal &&
                          (item.activity_name === "Dinner" ? (
                            <Utensils className="h-4 w-4" />
                          ) : (
                            <Coffee className="h-4 w-4" />
                          ))}
                        {item.activity_name}
                      </span>
                      {!isMeal && (
                        <>
                          {!!item.cost_inr && (
                            <span className="text-xs text-muted">₹{item.cost_inr.toLocaleString("en-IN")} pp</span>
                          )}
                          <span className="rounded-full bg-selected px-2 py-0.5 text-xs text-brand-primary">
                            {item.category}
                          </span>
                        </>
                      )}
                    </div>
                  );
                })}
                {overnight && (
                  <div className="bg-wash px-5 py-3">
                    <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted">
                      <Bed className="h-3.5 w-3.5" />
                      Overnight stay
                    </p>
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
