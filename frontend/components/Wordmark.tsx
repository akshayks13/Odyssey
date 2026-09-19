import { cn } from "@/lib/cn";

export function Wordmark({
  className,
  size = "md",
}: {
  className?: string;
  size?: "sm" | "md" | "lg";
}) {
  const sizes = {
    sm: "text-[1.35rem]",
    md: "text-2xl",
    lg: "text-6xl md:text-8xl",
  };
  return (
    <span
      className={cn("font-display font-semibold tracking-tight text-ink", sizes[size], className)}
      aria-label="Odyssey"
    >
      Odyssey
    </span>
  );
}
