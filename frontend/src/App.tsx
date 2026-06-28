import { useState } from "react";
import { useHistory } from "./hooks/useHistory";
import CompareTab from "./components/CompareTab";
import GuidesTab from "./components/GuidesTab";
import HistoryTab from "./components/HistoryTab";

type Tab = "compare" | "guides" | "history";

const TABS: [Tab, string, string][] = [
  ["compare", "Compare", "Run the instruments"],
  ["guides", "Guides", "How each model thinks"],
  ["history", "History", "Your past readings"],
];

export default function App() {
  const [tab, setTab] = useState<Tab>("compare");
  const hist = useHistory();

  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-surface/80 backdrop-blur">
        <div className="mx-auto max-w-6xl px-5 pt-7">
          <div className="flex items-center gap-3">
            {/* spectrum wordmark — negative→neutral→positive, the app's thesis */}
            <span
              aria-hidden
              className="h-6 w-6 rounded-md"
              style={{
                background:
                  "linear-gradient(135deg,#E5484D 0%,#64748B 50%,#0E9F6E 100%)",
              }}
            />
            <h1 className="text-2xl font-bold">Sentiment&nbsp;Lab</h1>
            <span className="pill bg-brand-wash text-brand-ink">
              {hist.history.length > 0 ? `${hist.history.length} runs` : "five models"}
            </span>
          </div>
          <p className="mt-1.5 max-w-xl text-sm text-ink-soft">
            One sentence, five minds. Watch a 1997 dictionary, a classical
            classifier, a sequence tagger, a transformer, and a frontier LLM
            judge the same text — on one shared spectrum.
          </p>

          <nav className="mt-5 flex gap-1" role="tablist" aria-label="Sections">
            {TABS.map(([id, label, hint]) => {
              const on = tab === id;
              return (
                <button
                  key={id}
                  role="tab"
                  aria-selected={on}
                  title={hint}
                  onClick={() => setTab(id)}
                  className={`group relative rounded-t-lg px-4 py-2.5 text-sm font-medium transition ${
                    on ? "text-brand-ink" : "text-ink-faint hover:text-ink-soft"
                  }`}
                >
                  {label}
                  <span
                    className={`absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-brand transition-transform ${
                      on ? "scale-x-100" : "scale-x-0 group-hover:scale-x-50"
                    }`}
                  />
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-5 py-8">
        {tab === "compare" && <CompareTab onRun={hist.addRun} />}
        {tab === "guides" && <GuidesTab />}
        {tab === "history" && (
          <HistoryTab history={hist.history} clear={hist.clear} />
        )}
      </main>

      <footer className="mx-auto max-w-6xl px-5 pb-10 pt-2 text-xs text-ink-faint">
        Lexicon → Classical ML → Sequence → Deep learning → LLM · explanations
        are model-native where possible, AI-narrated where noted.
      </footer>
    </div>
  );
}
