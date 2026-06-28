import type { AnalysisResult } from "../types";

const TONE: Record<string, { pill: string; bar: string }> = {
  positive: { pill: "pill-pos", bar: "bg-pos" },
  negative: { pill: "pill-neg", bar: "bg-neg" },
  neutral: { pill: "pill-neu", bar: "bg-neu" },
};
const FACE: Record<string, string> = {
  positive: "▲",
  negative: "▼",
  neutral: "●",
};

export default function ModelCard({
  name,
  family,
  result,
  index = 0,
  onClick,
}: {
  name: string;
  family: string;
  result: AnalysisResult;
  index?: number;
  onClick: () => void;
}) {
  if (result.error && !result.available)
    return (
      <div className="card animate-fade-up p-4 opacity-70" style={delay(index)}>
        <div className="flex items-center justify-between">
          <b className="font-display">{name}</b>
          <span className="pill bg-neu-wash text-neu">unavailable</span>
        </div>
        <p className="mt-2 text-sm text-ink-faint">{result.error}</p>
      </div>
    );

  const tone = TONE[result.label] ?? TONE.neutral;
  const pct = Math.round(result.confidence * 100);

  return (
    <button
      onClick={onClick}
      style={delay(index)}
      className="card group animate-fade-up p-4 text-left transition hover:-translate-y-0.5 hover:shadow-lift"
    >
      <div className="flex items-start justify-between">
        <div>
          <b className="font-display text-base">{name}</b>
          <p className="text-xs text-ink-faint">{family}</p>
        </div>
        <span className={`pill ${tone.pill}`}>
          <span aria-hidden>{FACE[result.label] ?? FACE.neutral}</span>
          {result.label}
        </span>
      </div>

      <div className="mt-4 flex items-center justify-between text-xs text-ink-faint">
        <span>confidence</span>
        <span className="tnum text-ink-soft">{pct}%</span>
      </div>
      <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-line">
        <div
          className={`h-2 rounded-full ${tone.bar} transition-[width] duration-700 ease-out`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {result.aspects && result.aspects.length > 0 && (
        <ul className="mt-3 flex flex-wrap gap-1.5">
          {result.aspects.map((a, i) => (
            <li
              key={i}
              className={`pill ${TONE[a.polarity]?.pill ?? "pill-neu"} text-[11px]`}
            >
              {a.term}
            </li>
          ))}
        </ul>
      )}

      <p className="mt-3 text-xs font-medium text-brand-ink opacity-0 transition group-hover:opacity-100">
        How did it decide? →
      </p>
    </button>
  );
}

const delay = (i: number) => ({ animationDelay: `${Math.min(i, 6) * 60}ms` });
