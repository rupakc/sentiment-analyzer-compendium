import { useEffect, useState } from "react";
import { explain } from "../api";
import type { AnalysisResult, Explanation } from "../types";

export default function ExplanationModal({
  text,
  result,
  name,
  onClose,
}: {
  text: string;
  result: AnalysisResult;
  name: string;
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

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose}
      role="presentation"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`Why ${name} predicted ${result.label}`}
        className="max-h-[80vh] w-full max-w-lg overflow-auto rounded-2xl bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between">
          <h2 className="text-lg font-semibold">
            {name} — {result.label}
          </h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="text-slate-400 hover:text-slate-700"
          >
            ✕
          </button>
        </div>
        {!exp && !err && <p className="mt-4 text-slate-500">Loading explanation…</p>}
        {err && <p className="mt-4 text-rose-600">{err}</p>}
        {exp && (
          <>
            <span className="mt-3 inline-block rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
              {exp.explanation_type === "native" ? "model-native evidence" : "AI-narrated"}
            </span>
            <p className="mt-3 text-slate-700">{exp.summary}</p>
            {exp.evidence.length > 0 && (
              <table className="mt-4 w-full text-sm">
                <tbody>
                  {exp.evidence.map((row, i) => (
                    <tr key={i} className="border-t">
                      {Object.values(row).map((v, j) => (
                        <td key={j} className="py-1 pr-4">
                          {String(v)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
      </div>
    </div>
  );
}
