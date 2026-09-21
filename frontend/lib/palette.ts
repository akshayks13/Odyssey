/** Odyssey brand — sea, sand, and copper. */
export const palette = {
  ink: "#1C1917",
  muted: "#6B6258",
  line: "#E6DCCF",
  wash: "#F3EDE3",
  paper: "#F8F4ED",
  sea: "#1E5C55",
  seaHover: "#164740",
  copper: "#B85C38",
  copperHover: "#9A4B2E",
  sage: "#4F6F56",
  clay: "#A34532",
} as const;

export const stepAccent: Record<string, string> = {
  trip_analyst: palette.sea,
  destination_agent: palette.sea,
  mobility_agent: palette.seaHover,
  budget_agent: palette.copper,
  itinerary_architect: palette.sea,
  critic_replanner: palette.sage,
  edit_router: palette.copper,
};
