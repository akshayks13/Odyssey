"use client";

import { useEffect, useRef, useState } from "react";
import "mapbox-gl/dist/mapbox-gl.css";
import { Destination, Route } from "@/lib/types";

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

export function MapView({ destinations, route }: { destinations: Destination[]; route: Route | null }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<import("mapbox-gl").Map | null>(null);
  const [mapError, setMapError] = useState(false);

  useEffect(() => {
    if (!MAPBOX_TOKEN || !containerRef.current || destinations.length === 0) return;

    let cancelled = false;

    import("mapbox-gl")
      .then((mapboxgl) => {
        if (cancelled || !containerRef.current) return;
        mapboxgl.default.accessToken = MAPBOX_TOKEN;

        const map = new mapboxgl.default.Map({
          container: containerRef.current,
          style: "mapbox://styles/mapbox/light-v11",
          center: [destinations[0].coordinates.lng, destinations[0].coordinates.lat],
          zoom: 6.5,
        });
        mapRef.current = map;

        map.on("load", () => {
          destinations.forEach((d, i) => {
            const el = document.createElement("div");
            el.className = "flex h-7 w-7 items-center justify-center rounded-full bg-[#1E5C55] text-xs font-bold text-white shadow-md";
            el.textContent = String(i + 1);
            new mapboxgl.default.Marker({ element: el })
              .setLngLat([d.coordinates.lng, d.coordinates.lat])
              .setPopup(new mapboxgl.default.Popup({ offset: 16 }).setText(d.name))
              .addTo(map);
          });

          if (route && route.ordered_destinations.length > 1) {
            const coordsByName = Object.fromEntries(destinations.map((d) => [d.name, d.coordinates]));
            const lineCoords = route.ordered_destinations
              .map((name) => coordsByName[name])
              .filter(Boolean)
              .map((c) => [c.lng, c.lat]);

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
  }, [destinations, route]);

  if (!MAPBOX_TOKEN || mapError) {
    // Graceful fallback — no NEXT_PUBLIC_MAPBOX_TOKEN configured, or the map
    // failed to load. Show a simple ordered-route strip instead of a blank box.
    return (
      <div className="flex flex-wrap items-center gap-2 rounded-3xl border border-line bg-white p-4">
        {destinations.map((d, i) => (
          <div key={d.name} className="flex items-center gap-2">
            <div className="flex items-center gap-2 rounded-full border border-line bg-white px-3 py-1.5">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-brand-primary text-[10px] font-bold text-white">
                {i + 1}
              </span>
              <span className="text-sm font-medium text-ink">{d.name}</span>
            </div>
            {i < destinations.length - 1 && <span className="text-grey-400">→</span>}
          </div>
        ))}
        <p className="mt-2 w-full text-xs text-muted">Interactive map available when a map token is configured.</p>
      </div>
    );
  }

  return <div ref={containerRef} className="h-72 w-full overflow-hidden rounded-3xl border border-line" />;
}
