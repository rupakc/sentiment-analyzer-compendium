import { render, screen, waitFor } from "@testing-library/react";
import { it, expect, vi } from "vitest";
import ExplanationModal from "../components/ExplanationModal";
import * as api from "../api";

it("loads and shows explanation summary", async () => {
  vi.spyOn(api, "explain").mockResolvedValue({
    model_id: "vader",
    explanation_type: "native",
    summary: "because great is positive",
    evidence: [],
  });
  render(
    <ExplanationModal
      text="great"
      name="VADER"
      result={{
        model_id: "vader",
        label: "positive",
        confidence: 0.8,
        scores: {},
        aspects: null,
        available: true,
        error: null,
      }}
      onClose={() => {}}
    />,
  );
  await waitFor(() =>
    expect(screen.getByText(/because great is positive/)).toBeInTheDocument(),
  );
  expect(screen.getByText(/native/i)).toBeInTheDocument();
});
