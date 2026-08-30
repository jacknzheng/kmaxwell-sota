import fs from "node:fs";
import katex from "../nature-of-momentum-2026-08-29/node_modules/katex/dist/katex.mjs";

const formulas = {
  factorization: String.raw`\lambda_i=\frac{R_i}{\lVert W_i\rVert^2},\qquad
    R_i\equiv \lambda_i\lVert W_i\rVert^2`,
  gauge_chain: String.raw`W\mapsto cW
    \quad\Longrightarrow\quad
    \lambda(cW)\approx\frac{\lambda(W)}{c^2}
    \quad\Longrightarrow\quad
    \lVert W\rVert^2\propto\eta
    \;\Rightarrow\;
    \lambda\propto\eta^{-1}`,
};

let html = fs.readFileSync(new URL("index.template.html", import.meta.url), "utf8");
html = html.replace(/\{\{(inline|display):([A-Za-z0-9_]+)\}\}/g, (_, mode, name) => {
  if (!(name in formulas)) throw new Error(`Unknown formula: ${name}`);
  return katex.renderToString(formulas[name], {
    displayMode: mode === "display",
    output: "htmlAndMathml",
    throwOnError: true,
    strict: "error",
  });
});
if (html.includes("{{inline:") || html.includes("{{display:")) {
  throw new Error("Unrendered math placeholder remains");
}
fs.writeFileSync(new URL("index.html", import.meta.url), html);
