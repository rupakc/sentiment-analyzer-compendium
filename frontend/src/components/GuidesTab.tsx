import { GUIDES } from "../content/guides";

export default function GuidesTab() {
  return (
    <div className="space-y-3">
      <p className="text-slate-600">
        How each model family decides sentiment — the 20-year arc.
      </p>
      {GUIDES.map((g) => (
        <details key={g.id} className="rounded-lg border bg-white p-4 open:shadow-sm">
          <summary className="cursor-pointer font-medium">
            {g.title} <span className="ml-2 text-xs text-slate-400">{g.era}</span>
          </summary>
          <p className="mt-3 text-slate-700">{g.body}</p>
        </details>
      ))}
    </div>
  );
}
