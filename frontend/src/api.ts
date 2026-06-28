import type { ModelInfo, AnalysisResult, Explanation } from "./types";

async function post<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

export async function listModels(): Promise<ModelInfo[]> {
  const res = await fetch("/api/models");
  if (!res.ok) throw new Error("failed to load models");
  return res.json();
}
export async function analyze(text: string, ids: string[]): Promise<AnalysisResult[]> {
  const data = await post<{ results: AnalysisResult[] }>("/api/analyze", {
    text,
    model_ids: ids,
  });
  return data.results;
}
export async function explain(text: string, id: string): Promise<Explanation> {
  return post<Explanation>("/api/explain", { text, model_id: id });
}
