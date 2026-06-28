/** Sentiment Lab design system — "comparison spectrometer", light lab theme. */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#F6F8FC", // app background — cool lab paper
        surface: "#FFFFFF",
        line: "#E6EAF2", // hairline borders
        ink: { DEFAULT: "#101725", soft: "#51607A", faint: "#8A97AD" },
        brand: { DEFAULT: "#2C5BFF", ink: "#1E3FBF", wash: "#EBF0FF" },
        pos: { DEFAULT: "#0E9F6E", wash: "#E6F6F0" },
        neg: { DEFAULT: "#E5484D", wash: "#FDECEC" },
        neu: { DEFAULT: "#64748B", wash: "#EEF1F6" },
      },
      fontFamily: {
        display: ['"Space Grotesk"', "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ['"IBM Plex Sans"', "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      borderRadius: { xl: "0.875rem", "2xl": "1.125rem" },
      boxShadow: {
        card: "0 1px 2px rgba(16,23,37,.04), 0 8px 24px -12px rgba(16,23,37,.10)",
        lift: "0 2px 6px rgba(16,23,37,.06), 0 18px 48px -16px rgba(16,23,37,.20)",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "scale-in": {
          "0%": { opacity: "0", transform: "scale(.96)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
      },
      animation: {
        "fade-up": "fade-up .45s cubic-bezier(.2,.7,.3,1) both",
        "scale-in": "scale-in .2s cubic-bezier(.2,.7,.3,1) both",
      },
    },
  },
  plugins: [],
};
