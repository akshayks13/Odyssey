"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import "maplibre-gl/dist/maplibre-gl.css";
import { Coordinates, Destination, Route, RouteLeg } from "@/lib/types";

// OpenStreetMap tiles: no account or key.
const OSM_STYLE = {
  version: 8 as const,
  sources: {
    osm: {
      type: "raster" as const,
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [{ id: "osm", type: "raster" as const, source: "osm" }],
};

interface Stop {
  name: string;
  coordinates: Coordinates;
  kind: "origin" | "stop";
}

function journeyStops(destinations: Destination[], route: Route | null): { names: string[]; legs: RouteLeg[] } {
  const legs = route?.legs || [];
  if (legs.length) {
    const names = [legs[0].origin, ...legs.map((leg) => leg.destination)];
    return { names, legs };
  }
  if (route?.ordered_destinations?.length) {
    return { names: route.ordered_destinations, legs: [] };
  }
  return { names: destinations.map((d) => d.name), legs: [] };
}

function coordLookup(destinations: Destination[]): Record<string, Coordinates> {
  const out: Record<string, Coordinates> = {};
  for (const d of destinations) {
    out[d.name.toLowerCase()] = d.coordinates;
  }
  return out;
}

/** Look a stop up by name, narrowed by the trip's own region so "Springfield" lands in the right country. */
async function geocodeName(name: string, region: string): Promise<Coordinates | null> {
  const query = region && !name.toLowerCase().includes(region.toLowerCase()) ? `${name}, ${region}` : name;
  try {
    const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(query)}`);
    if (!res.ok) return null;
    const hit = (await res.json())?.[0];
    return hit ? { lat: parseFloat(hit.lat), lng: parseFloat(hit.lon) } : null;
  } catch {
    return null;
  }
}

function hopLabel(leg: RouteLeg | undefined): string {
  if (!leg) return "→";
  const hours = leg.duration_hours ? `${leg.duration_hours.toFixed(1)}h` : "";
  return [leg.mode, hours].filter(Boolean).join(" · ") + " →";
}

export function MapView({ destinations, route }: { destinations: Destination[]; route: Route | null }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<import("maplibre-gl").Map | null>(null);
  const [mapError, setMapError] = useState(false);
  const [stops, setStops] = useState<Stop[]>([]);

  const { names, legs } = useMemo(() => journeyStops(destinations, route), [destinations, route]);

  useEffect(() => {
    let cancelled = false;
    const known = coordLookup(destinations);
    // The trip's own region ("Tuscany, Italy"), so a stop we have no coordinates for is not searched worldwide.
    const region = destinations.find((d) => d.region)?.region ?? "";

    (async () => {
      const resolved: Stop[] = [];
      for (let i = 0; i < names.length; i++) {
        const name = names[i];
        const coords = known[name.toLowerCase()] || (await geocodeName(name, region));
        if (!coords) continue;
        const isOrigin =
          i === 0 && !destinations.some((d) => d.name.toLowerCase() === name.toLowerCase());
        resolved.push({
          name,
          coordinates: coords,
          kind: isOrigin ? "origin" : "stop",
        });
      }
      if (!cancelled) setStops(resolved);
    })();

    return () => {
      cancelled = true;
    };
  }, [names, destinations, legs.length]);

  useEffect(() => {
    if (!containerRef.current || stops.length === 0) return;

    let cancelled = false;

    import("maplibre-gl")
      .then((maplibregl) => {
        if (cancelled || !containerRef.current) return;
        const map = new maplibregl.Map({
          container: containerRef.current,
          style: OSM_STYLE,
          center: [stops[0].coordinates.lng, stops[0].coordinates.lat],
          zoom: stops.length === 1 ? 8 : 6,
        });
        mapRef.current = map;

        map.on("load", () => {
          let stopNumber = 0;
          stops.forEach((stop) => {
            const el = document.createElement("div");
            const isOrigin = stop.kind === "origin";
            if (!isOrigin) stopNumber += 1;
            el.className = isOrigin
              ? "flex h-7 w-7 items-center justify-center rounded-full border-2 border-[#1E5C55] bg-white text-[10px] font-bold text-[#1E5C55]"
              : "flex h-7 w-7 items-center justify-center rounded-full bg-[#1E5C55] text-xs font-bold text-white";
            el.textContent = isOrigin ? "A" : String(stopNumber);
            new maplibregl.Marker({ element: el })
              .setLngLat([stop.coordinates.lng, stop.coordinates.lat])
              .setPopup(
                new maplibregl.Popup({ offset: 16 }).setText(isOrigin ? `From ${stop.name}` : stop.name)
              )
              .addTo(map);
          });

          if (stops.length > 1) {
            const lineCoords = stops.map((s) => [s.coordinates.lng, s.coordinates.lat]);
            map.addSource("route", {
              type: "geojson",
              data: { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: lineCoords } },
            });
            map.addLayer({
              id: "route-line",
              type: "line",
              source: "route",
              paint: { "line-color": "#1E5C55", "line-width": 3, "line-dasharray": [1, 1.5] },
            });

            const bounds = new maplibregl.LngLatBounds(lineCoords[0] as [number, number], lineCoords[0] as [number, number]);
            lineCoords.forEach((c) => bounds.extend(c as [number, number]));
            map.fitBounds(bounds, { padding: 56, maxZoom: 9 });
          }
        });

        // Fall back to the list only if the map never came up. Once it has loaded, a tile that fails to
        // arrive (a rate-limited or slow tile server) is a blank square, not a reason to throw the map away.
        let ready = false;
        map.on("load", () => {
          ready = true;
        });
        map.on("error", () => {
          if (!ready) setMapError(true);
        });
      })
      .catch(() => setMapError(true));

    return () => {
      cancelled = true;
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, [stops]);

  const caption = names.length ? names.join(" → ") : "";

  const strip = (
    <div className="flex flex-wrap items-center gap-2">
      {names.map((name, i) => (
        <div key={`${name}-${i}`} className="flex items-center gap-2">
          <div className="rounded-full border border-line bg-white px-3 py-1.5 text-sm font-medium text-ink">
            {i === 0 && !destinations.some((d) => d.name.toLowerCase() === name.toLowerCase())
              ? `From ${name}`
              : name}
          </div>
          {i < names.length - 1 && (
            <span className="text-xs uppercase text-grey-400">{hopLabel(legs[i])}</span>
          )}
        </div>
      ))}
    </div>
  );

  if (mapError) {
    return (
      <div className="rounded-3xl border border-line bg-white p-4">
        {strip}
        <p className="mt-2 text-xs text-muted">The map could not load right now, so the route is shown as a list.</p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-3xl border border-line bg-white">
      {caption && (
        <div className="border-b border-line px-4 py-3">
          {strip}
        </div>
      )}
      <div ref={containerRef} className="h-72 w-full" />
    </div>
  );
}
