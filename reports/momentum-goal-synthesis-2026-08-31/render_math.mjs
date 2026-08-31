import fs from "node:fs";
import katex from "../nature-of-momentum-2026-08-29/node_modules/katex/dist/katex.mjs";

const formulas = {
  filtered_spectrum: String.raw`D_w(\omega)=|H_w(\omega)|^2S_g(\omega)`,
  residual: String.raw`R_i=\lambda_i\lVert W_i\rVert^2`,
  quadratic_boundary: String.raw`D_t=\frac{\Delta_t^\top H_t\Delta_t}{\lVert\Delta_t\rVert^2},\qquad
    A_t=\frac{-2\langle g_t,\Delta_t\rangle}{\lVert\Delta_t\rVert^2}`,
  third_order: String.raw`\Delta L_t=\langle g_t,\Delta_t\rangle
    +\frac12\Delta_t^\top H_t\Delta_t
    +\frac16\nabla^3L_t[\Delta_t,\Delta_t,\Delta_t]+O(\lVert\Delta_t\rVert^4)`,
  wiener: String.raw`\underset{w}{\operatorname{minimize}}\;
    \sum_{\omega}\left[S_{\mathrm{noise}}(\omega)|H_w(\omega)|^2
    +S_{\mathrm{signal}}(\omega)|1-H_w(\omega)|^2\right]`,
};

let html = fs.readFileSync(new URL("index.template.html", import.meta.url), "utf8");
html = html.replace(/\{\{(inline|display):([A-Za-z0-9_]+)\}\}/g, (_, mode, name) => {
  if (!(name in formulas)) throw new Error(`Unknown formula: ${name}`);
  return katex.renderToString(formulas[name], {displayMode: mode === "display", output: "htmlAndMathml", throwOnError: true, strict: "error"});
});
if (html.includes("{{inline:") || html.includes("{{display:")) throw new Error("Unrendered formula");
fs.writeFileSync(new URL("index.html", import.meta.url), html);
