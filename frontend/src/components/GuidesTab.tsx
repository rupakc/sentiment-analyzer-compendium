import { GUIDES, type Guide } from "../content/guides";
import Math from "./Math";

export default function GuidesTab() {
  return (
    <div className="animate-fade-up space-y-8">
      {/* Hero */}
      <header className="max-w-2xl space-y-3">
        <span className="pill bg-brand-wash text-brand-ink font-mono">
          field guide
        </span>
        <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">
          From dictionaries to LLMs
        </h1>
        <p className="text-ink-soft leading-relaxed">
          Twenty-five years of sentiment analysis in five model families. Each
          advance is the same move — take something previously{" "}
          <em>hand-specified</em> and let the model <em>learn it from data</em>,
          paying in compute and interpretability for generality. The math below
          is real and hand-checkable.
        </p>
      </header>

      {/* Era rail + sections */}
      <ol className="relative space-y-6 border-l border-line pl-6 sm:pl-8">
        {GUIDES.map((g, i) => (
          <FamilySection key={g.id} guide={g} index={i} />
        ))}
      </ol>
    </div>
  );
}

function FamilySection({ guide: g, index }: { guide: Guide; index: number }) {
  return (
    <li className="relative">
      {/* timeline node */}
      <span
        aria-hidden
        className="absolute -left-[1.6rem] top-2 flex h-7 w-7 items-center justify-center rounded-full border border-line bg-surface font-mono text-xs text-brand-ink shadow-card sm:-left-[2.1rem]"
      >
        {index + 1}
      </span>

      <details className="card group overflow-hidden open:shadow-lift" {...(index === 0 ? { open: true } : {})}>
        <summary className="flex cursor-pointer list-none items-start justify-between gap-4 p-5 [&::-webkit-details-marker]:hidden">
          <div className="space-y-1">
            <span className="font-mono text-xs text-ink-faint">{g.era}</span>
            <h2 className="font-display text-xl font-semibold text-ink">
              {g.title}
            </h2>
            <p className="text-sm text-ink-soft">{g.tagline}</p>
          </div>
          <span className="mt-1 select-none font-mono text-ink-faint transition-transform group-open:rotate-90">
            ›
          </span>
        </summary>

        <div className="space-y-7 border-t border-line px-5 pb-6 pt-5">
          <Section title="Intuition">
            <p className="text-ink-soft leading-relaxed">{g.intuition}</p>
          </Section>

          <Section title="How it works">
            <ol className="space-y-2">
              {g.howItWorks.map((step, i) => (
                <li key={i} className="flex gap-3 text-ink-soft leading-relaxed">
                  <span className="mt-0.5 font-mono text-xs text-brand-ink">
                    {(i + 1).toString().padStart(2, "0")}
                  </span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
          </Section>

          <Section title="Key equations">
            <div className="space-y-4">
              {g.math.map((m, i) => (
                <div
                  key={i}
                  className="rounded-xl border border-line bg-paper/60 p-4"
                >
                  <div className="font-mono text-xs uppercase tracking-wide text-ink-faint">
                    {m.label}
                  </div>
                  <div className="my-2 overflow-x-auto">
                    <Math display tex={m.tex} />
                  </div>
                  {m.note && (
                    <p className="text-sm text-ink-soft">{m.note}</p>
                  )}
                </div>
              ))}
            </div>
          </Section>

          {g.proof && (
            <Section title="Derivation">
              <div className="rounded-xl border border-brand/20 bg-brand-wash/50 p-4">
                <p className="mb-3 text-sm font-medium text-brand-ink">
                  Claim. {g.proof.claim}
                </p>
                <ol className="space-y-3">
                  {g.proof.steps.map((s, i) => (
                    <li key={i} className="space-y-1.5">
                      <p className="text-sm text-ink-soft leading-relaxed">
                        {s.text}
                      </p>
                      {s.tex && (
                        <div className="overflow-x-auto">
                          <Math display tex={s.tex} />
                        </div>
                      )}
                    </li>
                  ))}
                </ol>
              </div>
            </Section>
          )}

          <Section title="Worked example">
            <pre className="tnum overflow-x-auto rounded-xl border border-line bg-ink/[0.03] p-4 text-xs leading-relaxed text-ink">
              {g.workedExample}
            </pre>
          </Section>

          {/* Pros / Cons */}
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-xl border border-pos/20 bg-pos-wash p-4">
              <h3 className="mb-2 font-display text-sm font-semibold text-pos">
                Strengths
              </h3>
              <ul className="space-y-1.5">
                {g.pros.map((p, i) => (
                  <li key={i} className="flex gap-2 text-sm text-ink-soft">
                    <span aria-hidden className="text-pos">
                      +
                    </span>
                    <span>{p}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="rounded-xl border border-neg/20 bg-neg-wash p-4">
              <h3 className="mb-2 font-display text-sm font-semibold text-neg">
                Weaknesses
              </h3>
              <ul className="space-y-1.5">
                {g.cons.map((c, i) => (
                  <li key={i} className="flex gap-2 text-sm text-ink-soft">
                    <span aria-hidden className="text-neg">
                      −
                    </span>
                    <span>{c}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* When to use */}
          <div className="rounded-xl border border-line bg-surface p-4">
            <h3 className="mb-1 font-mono text-xs uppercase tracking-wide text-brand-ink">
              When to use
            </h3>
            <p className="text-sm text-ink-soft leading-relaxed">
              {g.whenToUse}
            </p>
          </div>
        </div>
      </details>
    </li>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-2.5">
      <h3 className="font-display text-sm font-semibold uppercase tracking-wide text-ink-faint">
        {title}
      </h3>
      {children}
    </section>
  );
}
