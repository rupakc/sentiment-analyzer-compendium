import type { Run } from "../hooks/useHistory";
import { signedScore } from "./SentimentSpectrum";

const TONE: Record<string, string> = {
  positive: "pill-pos",
  negative: "pill-neg",
  neutral: "pill-neu",
};

export default function HistoryTab({
  history,
  clear,
}: {
  history: Run[];
  clear: () => void;
}) {
  if (history.length === 0)
    return (
      <div className="card flex flex-col items-center gap-2 p-10 text-center">
        <p className="font-display text-lg">No history yet</p>
        <p className="max-w-sm text-sm text-ink-soft">
          Every analysis you run is saved here, in this browser only. Nothing
          leaves your machine.
        </p>
      </div>
    );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-ink-soft">
          <span className="tnum font-semibold text-ink">{history.length}</span> run
          {history.length === 1 ? "" : "s"} · stored locally in this browser
        </p>
        <button
          onClick={clear}
          className="rounded-lg border border-line px-3 py-1.5 text-sm font-medium text-neg transition hover:bg-neg-wash"
        >
          Clear history
        </button>
      </div>

      <ul className="space-y-3">
        {history.map((run) => {
          const ok = run.results.filter((r) => r.available && !r.error);
          return (
            <li key={run.id} className="card p-4">
              <div className="flex items-center justify-between text-xs text-ink-faint">
                <span className="tnum">{new Date(run.ts).toLocaleString()}</span>
                <span>
                  {run.results.length} model{run.results.length === 1 ? "" : "s"}
                </span>
              </div>
              <p className="mt-1.5 line-clamp-2 font-medium text-ink">“{run.text}”</p>

              {/* mini consensus strip */}
              {ok.length > 0 && (
                <div className="relative mt-3 h-1.5 rounded-full"
                  style={{
                    background:
                      "linear-gradient(90deg,#FBE3E3,#EEF1F6 50%,#DCF2E9)",
                  }}
                >
                  {ok.map((r) => {
                    const left = ((signedScore(r) + 1) / 2) * 100;
                    const tone =
                      r.label === "positive"
                        ? "bg-pos"
                        : r.label === "negative"
                        ? "bg-neg"
                        : "bg-neu";
                    return (
                      <span
                        key={r.model_id}
                        title={`${r.model_id}: ${r.label}`}
                        className={`absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-surface ${tone}`}
                        style={{ left: `${left}%` }}
                      />
                    );
                  })}
                </div>
              )}

              <div className="mt-3 flex flex-wrap gap-1.5">
                {run.results.map((r) => (
                  <span
                    key={r.model_id}
                    className={`pill ${
                      r.available && !r.error ? TONE[r.label] ?? "pill-neu" : "bg-neu-wash text-neu"
                    }`}
                  >
                    {r.model_id}: {r.available && !r.error ? r.label : "n/a"}
                  </span>
                ))}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
