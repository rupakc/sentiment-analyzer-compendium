import { useState, useCallback } from "react";
import type { AnalysisResult } from "../types";

export interface Run {
  id: string;
  text: string;
  results: AnalysisResult[];
  ts: number;
}
const KEY = "sentiment-history";

function load(): Run[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]");
  } catch {
    return [];
  }
}

export function useHistory() {
  const [history, setHistory] = useState<Run[]>(load);
  const addRun = useCallback((r: { text: string; results: AnalysisResult[] }) => {
    setHistory((prev) => {
      const next = [
        { ...r, id: crypto.randomUUID(), ts: Date.now() },
        ...prev,
      ].slice(0, 100);
      try {
        localStorage.setItem(KEY, JSON.stringify(next));
      } catch {
        /* quota: skip persistence */
      }
      return next;
    });
  }, []);
  const clear = useCallback(() => {
    localStorage.removeItem(KEY);
    setHistory([]);
  }, []);
  return { history, addRun, clear };
}
