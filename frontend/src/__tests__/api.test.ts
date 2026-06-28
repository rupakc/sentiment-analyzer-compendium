import { describe, it, expect, vi } from "vitest";
import { analyze } from "../api";

describe("api.analyze", () => {
  it("posts text and ids, returns results", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ results: [{ model_id: "vader", label: "positive" }] }),
    }) as unknown as typeof fetch;
    const r = await analyze("hi", ["vader"]);
    expect(r[0].model_id).toBe("vader");
  });
});
