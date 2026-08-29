import fs from "node:fs";
import katex from "katex";

const formulas = {
  W: String.raw`W`,
  W_flip: String.raw`W(-1)`,
  W_omega: String.raw`W(\omega)`,
  sigma_sum: String.raw`\sigma_i + \sigma_j`,
  eta_scaling: String.raw`\eta^{2.4}`,
  eta_linear: String.raw`\eta^1`,
  eta_lambda: String.raw`\eta\lambda`,
  linear_characteristic: String.raw`r-a+\eta\lambda W(r^{-1})=0`,
  muon_update: String.raw`Z_t=W(q)G_t, \qquad X_{t+1}=aX_t-\eta s\,\operatorname{NS}(Z_t)`,
  expected_loss: String.raw`\mathbb{E}[\Delta L]\approx
    -\eta\,\mathbb{E}\!\left[\left\langle G_t,\operatorname{NS}(Z_t)\right\rangle\right]
    +\frac{\eta^2}{2}\,\mathbb{E}\!\left[\left\langle\operatorname{NS}(Z_t),H\operatorname{NS}(Z_t)\right\rangle\right]`,
};

let html = fs.readFileSync(new URL("index.template.html", import.meta.url), "utf8");
html = html.replace(/\{\{(inline|display):([A-Za-z0-9_]+)\}\}/g, (_, mode, name) => {
  if (!(name in formulas)) throw new Error(`Unknown formula placeholder: ${name}`);
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
