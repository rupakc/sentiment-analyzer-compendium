import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { it, expect, vi } from "vitest";
import CompareTab from "../components/CompareTab";
import * as api from "../api";

it("analyzes and renders a card", async () => {
  vi.spyOn(api, "listModels").mockResolvedValue([
    { id: "vader", name: "VADER", family: "Lexicon", description: "", available: true },
  ]);
  vi.spyOn(api, "analyze").mockResolvedValue([
    {
      model_id: "vader",
      label: "positive",
      confidence: 0.8,
      scores: {},
      aspects: null,
      available: true,
      error: null,
    },
  ]);
  render(<CompareTab onRun={() => {}} />);
  await screen.findByText("VADER"); // available models are preselected on load
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "I love it" } });
  fireEvent.click(screen.getByRole("button", { name: /analyze/i }));
  // a result card renders (its hover affordance is unique to cards)
  await waitFor(() =>
    expect(screen.getByText(/how did it decide/i)).toBeInTheDocument(),
  );
  expect(screen.getAllByText(/positive/i).length).toBeGreaterThan(0);
});
