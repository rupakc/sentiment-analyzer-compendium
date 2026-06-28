import type { Run } from "../hooks/useHistory";

export default function HistoryTab({
  history,
  clear,
}: {
  history: Run[];
  clear: () => void;
}) {
  if (history.length === 0) return <p className="text-slate-500">No analyses yet.</p>;
  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <button onClick={clear} className="text-sm text-rose-600 hover:underline">
          Clear history
        </button>
      </div>
      <ul className="space-y-3">
        {history.map((run) => (
          <li key={run.id} className="rounded-lg border bg-white p-4">
            <div className="flex justify-between text-xs text-slate-400">
              <span>{new Date(run.ts).toLocaleString()}</span>
              <span>{run.results.length} model(s)</span>
            </div>
            <p className="mt-1 truncate font-medium">{run.text}</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {run.results.map((r) => (
                <span
                  key={r.model_id}
                  className="rounded-full bg-slate-100 px-2 py-0.5 text-xs"
                >
                  {r.model_id}: {r.label}
                </span>
              ))}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
