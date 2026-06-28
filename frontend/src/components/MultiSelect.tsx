import type { ModelInfo } from "../types";

// One accent dot per family — a quiet legend that ties chips to the lineage.
const FAMILY_DOT: Record<string, string> = {
  "Lexicon/rules": "#E5484D",
  "Classical ML": "#E8A23D",
  Sequence: "#0E9F6E",
  "Deep learning": "#2C5BFF",
  LLM: "#7C5CFC",
};

export default function MultiSelect({
  models,
  selected,
  onToggle,
  onBulk,
}: {
  models: ModelInfo[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  onBulk?: (ids: string[]) => void;
}) {
  const available = models.filter((m) => m.available).map((m) => m.id);
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <label className="text-sm font-medium text-ink-soft">
          Models <span className="text-ink-faint">({selected.size} selected)</span>
        </label>
        {onBulk && (
          <div className="flex gap-3 text-xs font-medium text-brand-ink">
            <button type="button" onClick={() => onBulk(available)} className="hover:underline">
              Select all
            </button>
            <button type="button" onClick={() => onBulk([])} className="hover:underline">
              Clear
            </button>
          </div>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        {models.map((m) => {
          const on = selected.has(m.id);
          return (
            <button
              key={m.id}
              type="button"
              disabled={!m.available}
              aria-pressed={on}
              onClick={() => onToggle(m.id)}
              title={m.available ? m.description : "Unavailable — API key not set"}
              className={`group inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm transition ${
                on
                  ? "border-brand bg-brand-wash text-brand-ink shadow-card"
                  : "border-line bg-surface text-ink-soft hover:border-ink-faint"
              } ${!m.available ? "cursor-not-allowed opacity-45" : ""}`}
            >
              <span
                aria-hidden
                className="h-2 w-2 rounded-full"
                style={{ background: FAMILY_DOT[m.family] ?? "#8A97AD" }}
              />
              {m.name}
              {!m.available && <span className="text-[10px] text-ink-faint">· no key</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}
