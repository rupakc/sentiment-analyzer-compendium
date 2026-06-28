import { it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useHistory } from "../hooks/useHistory";

it("persists a run to localStorage", () => {
  localStorage.clear();
  const { result } = renderHook(() => useHistory());
  act(() => result.current.addRun({ text: "hi", results: [] }));
  expect(result.current.history).toHaveLength(1);
  expect(JSON.parse(localStorage.getItem("sentiment-history")!)).toHaveLength(1);
});
