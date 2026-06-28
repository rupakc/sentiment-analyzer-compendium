import { render, screen, fireEvent } from "@testing-library/react";
import { it, expect, vi } from "vitest";
import HistoryTab from "../components/HistoryTab";

it("renders runs and clears", () => {
  const clear = vi.fn();
  render(
    <HistoryTab
      clear={clear}
      history={[
        {
          id: "1",
          text: "I love it",
          ts: Date.now(),
          results: [
            {
              model_id: "vader",
              label: "positive",
              confidence: 0.8,
              scores: {},
              aspects: null,
              available: true,
              error: null,
            },
          ],
        },
      ]}
    />,
  );
  expect(screen.getByText(/I love it/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /clear/i }));
  expect(clear).toHaveBeenCalled();
});
