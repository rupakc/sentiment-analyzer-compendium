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
  await screen.findByText("VADER");
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "I love it" } });
  fireEvent.click(screen.getByText("VADER")); // select model
  fireEvent.click(screen.getByRole("button", { name: /analyze/i }));
  await waitFor(() => expect(screen.getByText(/positive/i)).toBeInTheDocument());
});
