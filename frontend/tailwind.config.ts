import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        brand: {
          primary: "#1E5C55",
          hover: "#164740",
          accent: "#B85C38",
          sage: "#4F6F56",
          clay: "#A34532",
        },
        ink: "#1C1917",
        muted: "#6B6258",
        line: "#E6DCCF",
        wash: "#F3EDE3",
        paper: "#F8F4ED",
        selected: "#E4EFE9",
        grey: {
          50: "#F8F4ED",
          100: "#F3EDE3",
          200: "#E6DCCF",
          300: "#D4C7B6",
          400: "#B5A48F",
          500: "#8A7D6E",
          600: "#6B6258",
          700: "#4A433C",
          800: "#2E2925",
          900: "#1C1917",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "Georgia", "serif"],
      },
      boxShadow: {
        search: "0 1px 8px rgba(28, 25, 23, 0.12)",
        "search-hover": "0 4px 16px rgba(28, 25, 23, 0.14)",
        card: "0 1px 2px rgba(28, 25, 23, 0.06), 0 8px 24px rgba(28, 25, 23, 0.04)",
      },
      borderRadius: {
        search: "24px",
      },
    },
  },
  plugins: [],
};
export default config;
