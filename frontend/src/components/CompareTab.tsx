import { useEffect, useState } from "react";
import { listModels, analyze } from "../api";
import type { ModelInfo, AnalysisResult } from "../types";
import MultiSelect from "./MultiSelect";
import ModelCard from "./ModelCard";
import ExplanationModal from "./ExplanationModal";

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
    listModels().then(setModels).catch(() => setModels([]));
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

  return (
    <div className="space-y-6">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={4}
        placeholder="Enter text to analyze…"
        className="w-full rounded-lg border p-3 focus:border-indigo-500 focus:outline-none"
      />
      <MultiSelect models={models} selected={selected} onToggle={toggle} />
      <button
        onClick={run}
        disabled={loading || !text.trim() || selected.size === 0}
        className="rounded-lg bg-indigo-600 px-5 py-2 font-medium text-white disabled:opacity-40"
      >
        {loading ? "Analyzing…" : "Analyze"}
      </button>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {results.map((r) => (
          <ModelCard
            key={r.model_id}
            name={nameOf(r.model_id)}
            family={familyOf(r.model_id)}
            result={r}
            onClick={() => setActive(r)}
          />
        ))}
      </div>
      {active && (
        <ExplanationModal
          text={text}
          result={active}
          name={nameOf(active.model_id)}
          onClose={() => setActive(null)}
        />
      )}
    </div>
  );
}
