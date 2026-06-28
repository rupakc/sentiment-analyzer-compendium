import katex from "katex";

/** Renders a LaTeX string with KaTeX. `display` → centered block, else inline. */
export default function Math({
  tex,
  display = false,
  className = "",
}: {
  tex: string;
  display?: boolean;
  className?: string;
}) {
  const html = katex.renderToString(tex, {
    displayMode: display,
    throwOnError: false,
  });
  const Tag = display ? "div" : "span";
  return (
    <Tag
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
