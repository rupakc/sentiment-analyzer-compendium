import { useState } from "react";
import { useHistory } from "./hooks/useHistory";
import CompareTab from "./components/CompareTab";
import GuidesTab from "./components/GuidesTab";
import HistoryTab from "./components/HistoryTab";

type Tab = "compare" | "guides" | "history";

export default function App() {
  const [tab, setTab] = useState<Tab>("compare");
  const hist = useHistory();
  const tabs: [Tab, string][] = [
    ["compare", "Compare"],
    ["guides", "Guides"],
    ["history", "History"],
  ];
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b bg-white">
        <div className="mx-auto max-w-5xl px-4 pt-6">
          <h1 className="text-xl font-bold tracking-tight">Sentiment Lab</h1>
          <p className="text-sm text-slate-500">
            Compare 20 years of sentiment-analysis models, side by side.
          </p>
        </div>
        <nav className="mx-auto mt-4 flex max-w-5xl gap-1 px-4" role="tablist">
          {tabs.map(([id, label]) => (
            <button
              key={id}
              role="tab"
              aria-selected={tab === id}
              onClick={() => setTab(id)}
              className={`px-4 py-3 text-sm font-medium ${
                tab === id
                  ? "border-b-2 border-indigo-600 text-indigo-700"
                  : "text-slate-500 hover:text-slate-700"
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-8">
        {tab === "compare" && <CompareTab onRun={hist.addRun} />}
        {tab === "guides" && <GuidesTab />}
        {tab === "history" && <HistoryTab history={hist.history} clear={hist.clear} />}
      </main>
    </div>
  );
}
