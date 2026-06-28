import type { AnalysisResult } from "../types";

const COLOR: Record<string, string> = {
  positive: "bg-emerald-50 text-emerald-700 border-emerald-200",
  negative: "bg-rose-50 text-rose-700 border-rose-200",
  neutral: "bg-slate-50 text-slate-600 border-slate-200",
};

export default function ModelCard({
  name,
  family,
  result,
  onClick,
}: {
  name: string;
  family: string;
  result: AnalysisResult;
  onClick: () => void;
}) {
  if (result.error && !result.available)
    return (
      <div className="rounded-xl border p-4 opacity-60">
        <b>{name}</b>
        <p className="text-sm text-slate-500">{result.error}</p>
      </div>
    );
  return (
    <button
      onClick={onClick}
      className="rounded-xl border bg-white p-4 text-left shadow-sm transition hover:shadow-md"
    >
      <div className="flex items-center justify-between">
        <b>{name}</b>
        <span className="text-xs text-slate-400">{family}</span>
      </div>
      <span
        className={`mt-2 inline-block rounded-full border px-2 py-0.5 text-sm ${
          COLOR[result.label] || COLOR.neutral
        }`}
      >
        {result.label}
      </span>
      <div className="mt-3 h-2 w-full rounded bg-slate-100">
        <div
          className="h-2 rounded bg-indigo-500"
          style={{ width: `${Math.round(result.confidence * 100)}%` }}
        />
      </div>
      <p className="mt-1 text-xs text-slate-400">
        {Math.round(result.confidence * 100)}% confidence · click for why
      </p>
      {result.aspects && result.aspects.length > 0 && (
        <ul className="mt-2 text-xs text-slate-600">
          {result.aspects.map((a, i) => (
            <li key={i}>
              {a.term}: {a.polarity}
            </li>
          ))}
        </ul>
      )}
    </button>
  );
}
