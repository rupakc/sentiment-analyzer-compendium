import { useEffect, useState } from "react";
import { explain } from "../api";
import type { AnalysisResult, Explanation } from "../types";

const TONE: Record<string, string> = {
  positive: "pill-pos",
  negative: "pill-neg",
  neutral: "pill-neu",
};

export default function ExplanationModal({
  text,
  result,
  name,
  family,
  onClose,
}: {
  text: string;
  result: AnalysisResult;
  name: string;
  family?: string;
  onClose: () => void;
}) {
  const [exp, setExp] = useState<Explanation | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    explain(text, result.model_id)
      .then(setExp)
      .catch(() => setErr("Could not load explanation."));
  }, [text, result.model_id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const pct = Math.round(result.confidence * 100);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4 backdrop-blur-sm"
      onClick={onClose}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`Why ${name} predicted ${result.label}`}
        className="max-h-[86vh] w-full max-w-2xl animate-scale-in overflow-auto rounded-2xl bg-surface shadow-lift"
        onClick={(e) => e.stopPropagation()}
      >
        {/* header */}
        <div className="sticky top-0 z-10 flex items-start justify-between gap-4 border-b border-line bg-surface/95 px-6 py-4 backdrop-blur">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-display text-lg">{name}</h2>
              {family && <span className="text-xs text-ink-faint">{family}</span>}
            </div>
            <div className="mt-1.5 flex items-center gap-2">
              <span className={`pill ${TONE[result.label] ?? "pill-neu"}`}>
                {result.label}
              </span>
              <span className="tnum text-xs text-ink-soft">{pct}% confidence</span>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg p-1 text-ink-faint transition hover:bg-paper hover:text-ink"
          >
            ✕
          </button>
        </div>

        <div className="space-y-6 px-6 py-5">
          {!exp && !err && <p className="text-ink-soft">Loading explanation…</p>}
          {err && <p className="text-neg">{err}</p>}

          {exp && (
            <>
              <span
                className={`pill ${
                  exp.explanation_type === "native"
                    ? "bg-brand-wash text-brand-ink"
                    : "bg-neu-wash text-neu"
                }`}
              >
                {exp.explanation_type === "native"
                  ? "model-native evidence"
                  : "AI-narrated"}
              </span>

              {exp.method && (
                <Field label="Method">
                  <p className="text-ink-soft">{exp.method}</p>
                </Field>
              )}

              {exp.summary && (
                <Field label="In plain language">
                  <p className="leading-relaxed text-ink">{exp.summary}</p>
                </Field>
              )}

              {exp.steps && exp.steps.length > 0 && (
                <Field label="How it reached this answer">
                  <ol className="space-y-2">
                    {exp.steps.map((s, i) => (
                      <li key={i} className="flex gap-3">
                        <span className="tnum mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-brand-wash text-[11px] font-semibold text-brand-ink">
                          {i + 1}
                        </span>
                        <span className="text-sm text-ink-soft">{s}</span>
                      </li>
                    ))}
                  </ol>
                </Field>
              )}

              {exp.evidence && exp.evidence.length > 0 && (
                <Field label="Evidence">
                  <Evidence items={exp.evidence} />
                </Field>
              )}

              {exp.biases && exp.biases.length > 0 && (
                <Field label="Biases & caveats">
                  <ul className="space-y-1.5 rounded-xl border border-neg/20 bg-neg-wash/50 p-3">
                    {exp.biases.map((b, i) => (
                      <li key={i} className="flex gap-2 text-sm text-ink-soft">
                        <span aria-hidden className="text-neg">
                          ⚠
                        </span>
                        {b}
                      </li>
                    ))}
                  </ul>
                </Field>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-faint">
        {label}
      </h3>
      {children}
    </div>
  );
}

/** Diverging bars: weight>0 leans positive (teal, right), weight<0 negative (coral, left). */
function Evidence({
  items,
}: {
  items: { label: string; detail?: string; weight?: number }[];
}) {
  const weighted = items.filter((i) => typeof i.weight === "number");
  if (weighted.length === 0) {
    // no numeric weights — render as labelled chips (e.g. CRF aspects)
    return (
      <ul className="flex flex-wrap gap-2">
        {items.map((it, i) => (
          <li key={i} className="pill bg-neu-wash text-neu">
            {it.label}
            {it.detail ? ` · ${it.detail}` : ""}
          </li>
        ))}
      </ul>
    );
  }
  const maxAbs = Math.max(...weighted.map((i) => Math.abs(i.weight!))) || 1;

  return (
    <ul className="space-y-1.5">
      {items.map((it, i) => {
        const w = it.weight ?? 0;
        const frac = (Math.abs(w) / maxAbs) * 50; // half-width, diverging from centre
        const pos = w >= 0;
        return (
          <li key={i} className="flex items-center gap-3 text-sm">
            <span className="w-28 shrink-0 truncate font-mono text-xs text-ink">
              {it.label}
            </span>
            <span className="relative h-4 flex-1 rounded bg-paper">
              <span className="absolute left-1/2 top-0 h-4 w-px bg-line" />
              <span
                className={`absolute top-0.5 h-3 rounded ${pos ? "bg-pos" : "bg-neg"}`}
                style={
                  pos
                    ? { left: "50%", width: `${frac}%` }
                    : { right: "50%", width: `${frac}%` }
                }
              />
            </span>
            <span className="tnum w-14 shrink-0 text-right text-xs text-ink-faint">
              {w >= 0 ? "+" : ""}
              {w.toFixed(2)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
