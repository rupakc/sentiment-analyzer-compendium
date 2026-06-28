import { render, screen } from "@testing-library/react";
import { it, expect } from "vitest";
import GuidesTab from "../components/GuidesTab";

it("renders a section per family", () => {
  render(<GuidesTab />);
  expect(screen.getByText(/Lexicon/i)).toBeInTheDocument();
  expect(screen.getByText(/Deep learning/i)).toBeInTheDocument();
  expect(screen.getByText(/LLMs \(Claude\)/i)).toBeInTheDocument();
});
