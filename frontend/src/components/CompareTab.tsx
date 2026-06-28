import { useEffect, useState } from "react";
import { listModels, analyze } from "../api";
import type { ModelInfo, AnalysisResult } from "../types";
import MultiSelect from "./MultiSelect";
import ModelCard from "./ModelCard";
import ExplanationModal from "./ExplanationModal";
import SentimentSpectrum from "./SentimentSpectrum";

const SAMPLES = [
  "This is not very good, honestly.",
  "The battery is amazing but the screen is terrible.",
  "Oh great, another meeting that could have been an email.",
  "I absolutely love how fast and reliable this is!",
];

export default function CompareTab({
  onRun,
}: {
  onRun: (r: { text: string; results: AnalysisResult[] }) => void;
}) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [text, setText] = useState("");
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [active, setActive] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
        // preselect the available models so first run is one click
        setSelected(new Set(m.filter((x) => x.available).map((x) => x.id)));
      })
      .catch(() => setModels([]));
  }, []);

  const nameOf = (id: string) => models.find((m) => m.id === id)?.name ?? id;
  const familyOf = (id: string) => models.find((m) => m.id === id)?.family ?? "";

  const toggle = (id: string) =>
    setSelected((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });

  async function run() {
    if (!text.trim() || selected.size === 0) return;
    setLoading(true);
    try {
      const r = await analyze(text, [...selected]);
      setResults(r);
      onRun({ text, results: r });
    } finally {
      setLoading(false);
    }
  }

  const canRun = !!text.trim() && selected.size > 0 && !loading;

  return (
    <div className="space-y-6">
      {/* sample chamber */}
      <section className="card p-5">
        <label htmlFor="sample" className="text-sm font-medium text-ink-soft">
          Text to analyze
        </label>
        <textarea
          id="sample"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
          maxLength={5000}
          placeholder="Type or paste a sentence — try a negation, a mixed review, or some sarcasm…"
          className="mt-2 w-full resize-y rounded-xl border border-line bg-paper/60 p-3.5 text-ink
            placeholder:text-ink-faint focus:border-brand focus:bg-surface focus:outline-none"
        />
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <span className="mr-1 text-xs text-ink-faint">Try:</span>
          {SAMPLES.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setText(s)}
              className="rounded-full border border-line bg-surface px-2.5 py-1 text-xs text-ink-soft transition hover:border-brand hover:text-brand-ink"
            >
              {s.length > 38 ? s.slice(0, 36) + "…" : s}
            </button>
          ))}
          <span className="tnum ml-auto text-xs text-ink-faint">{text.length}/5000</span>
        </div>

        <div className="mt-5">
          <MultiSelect
            models={models}
            selected={selected}
            onToggle={toggle}
            onBulk={(ids) => setSelected(new Set(ids))}
          />
        </div>

        <div className="mt-5 flex items-center gap-3">
          <button onClick={run} disabled={!canRun} className="btn-primary">
            {loading ? (
              <>
                <Spinner /> Analyzing…
              </>
            ) : (
              <>Analyze ▸</>
            )}
          </button>
          {selected.size === 0 && (
            <span className="text-xs text-ink-faint">Pick at least one model.</span>
          )}
        </div>
      </section>

      {results.length > 0 ? (
        <>
          <SentimentSpectrum results={results} nameOf={nameOf} />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {results.map((r, i) => (
              <ModelCard
                key={r.model_id}
                name={nameOf(r.model_id)}
                family={familyOf(r.model_id)}
                result={r}
                index={i}
                onClick={() => setActive(r)}
              />
            ))}
          </div>
        </>
      ) : (
        !loading && <EmptyState />
      )}

      {active && (
        <ExplanationModal
          text={text}
          result={active}
          name={nameOf(active.model_id)}
          family={familyOf(active.model_id)}
          onClose={() => setActive(null)}
        />
      )}
    </div>
  );
}

function Spinner() {
  return (
    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
  );
}

function EmptyState() {
  return (
    <div className="card flex flex-col items-center gap-2 p-10 text-center">
      <span
        aria-hidden
        className="h-8 w-28 rounded-full"
        style={{
          background:
            "linear-gradient(90deg,#E5484D,#E6EAF2 50%,#0E9F6E)",
        }}
      />
      <p className="font-display text-lg">No readings yet</p>
      <p className="max-w-sm text-sm text-ink-soft">
        Enter some text and hit Analyze. Each model drops a needle on the shared
        spectrum, then click any card to see exactly how it decided.
      </p>
    </div>
  );
}
