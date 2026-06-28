import type { ModelInfo } from "../types";

export default function MultiSelect({
  models,
  selected,
  onToggle,
}: {
  models: ModelInfo[];
  selected: Set<string>;
  onToggle: (id: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {models.map((m) => (
        <button
          key={m.id}
          type="button"
          disabled={!m.available}
          aria-pressed={selected.has(m.id)}
          onClick={() => onToggle(m.id)}
          title={m.available ? m.description : "Unavailable (API key not set)"}
          className={`rounded-full border px-3 py-1 text-sm transition ${
            selected.has(m.id)
              ? "border-indigo-600 bg-indigo-50 text-indigo-700"
              : "border-slate-300 text-slate-600 hover:border-slate-400"
          } ${!m.available ? "cursor-not-allowed opacity-40" : ""}`}
        >
          {m.name}
        </button>
      ))}
    </div>
  );
}
