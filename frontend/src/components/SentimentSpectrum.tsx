import { useEffect, useState } from "react";
import type { AnalysisResult } from "../types";

/** Map any model's result to one signed position in [-1, 1]. */
export function signedScore(r: AnalysisResult): number {
  const s = r.scores || {};
  if (typeof s.compound === "number") return clamp(s.compound); // VADER
  const pos = s.positive ?? s.pos ?? 0;
  const neg = s.negative ?? s.neg ?? 0;
  if (pos || neg) return clamp(pos - neg); // proba/logit spread
  const dir = r.label === "positive" ? 1 : r.label === "negative" ? -1 : 0;
  return clamp(dir * (r.confidence || 0));
}
const clamp = (n: number) => Math.max(-1, Math.min(1, n));
const toneOf = (n: number) =>
  n > 0.05 ? "pos" : n < -0.05 ? "neg" : "neu";
const DOT: Record<string, string> = {
  pos: "bg-pos",
  neg: "bg-neg",
  neu: "bg-neu",
};

export default function SentimentSpectrum({
  results,
  nameOf,
}: {
  results: AnalysisResult[];
  nameOf: (id: string) => string;
}) {
  const items = results
    .filter((r) => r.available && !r.error)
    .map((r) => ({ r, score: signedScore(r) }));

  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const t = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(t);
  }, []);

  if (items.length === 0) return null;

  const scores = items.map((i) => i.score);
  const spread = Math.max(...scores) - Math.min(...scores);
  const consensus =
    spread < 0.35
      ? { text: "The instruments agree", tone: "text-pos" }
      : spread < 0.9
      ? { text: "Partial disagreement", tone: "text-neu" }
      : { text: "The instruments disagree — open a card to see why", tone: "text-neg" };

  return (
    <section className="card animate-fade-up p-5" aria-label="Sentiment spectrum">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-soft">
          Consensus spectrum
        </h2>
        <span className={`text-xs font-medium ${consensus.tone}`}>
          {consensus.text}
        </span>
      </div>

      {/* axis labels */}
      <div className="mb-2 flex justify-between text-xs font-medium">
        <span className="text-neg">◄ Negative</span>
        <span className="text-ink-faint">Neutral</span>
        <span className="text-pos">Positive ►</span>
      </div>

      {/* the track + needles */}
      <div className="relative h-16">
        <div
          className="absolute inset-x-0 top-7 h-2.5 rounded-full"
          style={{
            background:
              "linear-gradient(90deg,#E5484D 0%,#F2C5A0 28%,#E6EAF2 50%,#A8E0CC 72%,#0E9F6E 100%)",
          }}
        />
        {/* neutral centre tick */}
        <div className="absolute left-1/2 top-5 h-6 w-px -translate-x-1/2 bg-ink-faint/50" />

        {items.map(({ r, score }, i) => {
          const left = ((score + 1) / 2) * 100;
          const tone = toneOf(score);
          return (
            <div
              key={r.model_id}
              className="absolute top-7 -translate-x-1/2 transition-[left] duration-700 ease-out"
              style={{ left: `${mounted ? left : 50}%` }}
              title={`${nameOf(r.model_id)}: ${r.label} (${score.toFixed(2)})`}
            >
              {/* needle dot on the track */}
              <span
                className={`block h-4 w-4 -translate-y-0.5 rounded-full border-2 border-surface shadow-card ${DOT[tone]}`}
              />
              {/* alternating label above/below to reduce overlap */}
              <span
                className={`absolute left-1/2 -translate-x-1/2 whitespace-nowrap font-mono text-[10px] text-ink-soft ${
                  i % 2 ? "top-5" : "-top-5"
                }`}
              >
                {nameOf(r.model_id)}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
