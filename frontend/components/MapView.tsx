"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import "mapbox-gl/dist/mapbox-gl.css";
import { Coordinates, Destination, Route, RouteLeg } from "@/lib/types";

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

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

async function geocodeName(name: string): Promise<Coordinates | null> {
  if (!MAPBOX_TOKEN) return null;
  const query = encodeURIComponent(`${name}, India`);
  const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${query}.json?access_token=${MAPBOX_TOKEN}&limit=1&country=IN`;
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    const center = data?.features?.[0]?.center;
    if (!Array.isArray(center) || center.length < 2) return null;
    return { lng: center[0], lat: center[1] };
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
  const mapRef = useRef<import("mapbox-gl").Map | null>(null);
  const [mapError, setMapError] = useState(false);
  const [stops, setStops] = useState<Stop[]>([]);

  const { names, legs } = useMemo(() => journeyStops(destinations, route), [destinations, route]);

  useEffect(() => {
    let cancelled = false;
    const known = coordLookup(destinations);

    (async () => {
      const resolved: Stop[] = [];
      for (let i = 0; i < names.length; i++) {
        const name = names[i];
        const coords = known[name.toLowerCase()] || (await geocodeName(name));
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
    if (!MAPBOX_TOKEN || !containerRef.current || stops.length === 0) return;

    let cancelled = false;

    import("mapbox-gl")
      .then((mapboxgl) => {
        if (cancelled || !containerRef.current) return;
        mapboxgl.default.accessToken = MAPBOX_TOKEN;

        const map = new mapboxgl.default.Map({
          container: containerRef.current,
          style: "mapbox://styles/mapbox/light-v11",
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
            new mapboxgl.default.Marker({ element: el })
              .setLngLat([stop.coordinates.lng, stop.coordinates.lat])
              .setPopup(
                new mapboxgl.default.Popup({ offset: 16 }).setText(isOrigin ? `From ${stop.name}` : stop.name)
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

            const bounds = new mapboxgl.default.LngLatBounds(lineCoords[0] as [number, number], lineCoords[0] as [number, number]);
            lineCoords.forEach((c) => bounds.extend(c as [number, number]));
            map.fitBounds(bounds, { padding: 56, maxZoom: 9 });
          }
        });

        map.on("error", () => setMapError(true));
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

  if (!MAPBOX_TOKEN || mapError) {
    return (
      <div className="rounded-3xl border border-line bg-white p-4">
        {strip}
        <p className="mt-2 text-xs text-muted">Interactive map available when a map token is configured.</p>
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
