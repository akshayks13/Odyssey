"use client";

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BudgetBreakdown } from "@/lib/types";
import { palette } from "@/lib/palette";

const COLORS: Record<string, string> = {
  hotels: palette.sea,
  food: palette.copper,
  activities: palette.sage,
  transport: palette.ink,
};

export function BudgetChart({ budget }: { budget: BudgetBreakdown }) {
  const data = budget.line_items.map((item) => ({
    name: item.category ? item.category[0].toUpperCase() + item.category.slice(1) : "Other",
    amount: item.amount_inr,
    key: item.category || "other",
  }));

  const overBudget = budget.total_inr > budget.ceiling_inr;

  return (
    <div className="rounded-3xl border border-line bg-white p-5 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-semibold text-ink">Budget</h3>
        <span className={`text-sm font-medium ${overBudget ? "text-brand-clay" : "text-brand-sage"}`}>
          ₹{budget.total_inr.toLocaleString("en-IN")} / ₹{budget.ceiling_inr.toLocaleString("en-IN")}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eeeeee" />
          <XAxis dataKey="name" tick={{ fontSize: 12, fill: "#5f6368" }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fontSize: 12, fill: "#5f6368" }} axisLine={false} tickLine={false} />
          <Tooltip formatter={((v: number) => [`₹${Number(v).toLocaleString("en-IN")}`, "Amount"]) as never} />
          <Bar dataKey="amount" radius={[8, 8, 0, 0]}>
            {data.map((entry) => (
              <Cell key={entry.key} fill={COLORS[entry.key] ?? palette.muted} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {budget.selected_hotels.length > 0 && (
        <div className="mt-4 border-t border-line pt-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted">Place to stay</p>
          <ul className="mt-2 space-y-2">
            {budget.selected_hotels.map((hotel) => (
              <li key={`${hotel.destination}-${hotel.name}`} className="flex items-baseline justify-between gap-3 text-sm">
                <span className="text-ink">
                  {hotel.name}
                  <span className="ml-2 text-xs text-muted">{hotel.destination}</span>
                </span>
                <span className="shrink-0 text-muted">
                  {hotel.rating.toFixed(1)}★ · ₹{hotel.price_per_night_inr.toLocaleString("en-IN")}/night
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {budget.tradeoff_suggestions.length > 0 && (
        <div className="mt-3 rounded-2xl bg-wash p-3 text-sm text-ink">
          <p className="font-medium">Suggestions</p>
          <ul className="ml-4 list-disc">
            {budget.tradeoff_suggestions.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
