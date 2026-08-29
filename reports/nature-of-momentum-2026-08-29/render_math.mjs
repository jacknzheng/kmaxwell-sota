import fs from "node:fs";
import katex from "katex";

const formulas = {
  kernel: String.raw`Z_t=\sum_{k=0}^{t} w_t(k)\,G_{t-k}`,
  ema_kernel: String.raw`w(k)\propto\beta^k`,
  msign_scale: String.raw`\operatorname{msign}(cZ)=\operatorname{msign}(Z),\qquad c>0`,
  two_modes: String.raw`Z=\alpha A+\beta B`,
  ratio_modes: String.raw`\beta/\alpha`,
  W_def: String.raw`W(q)=\sum_{k\ge 0}w(k)q^k`,
  W_flip: String.raw`W(-1)`,
  eta_lambda_flip: String.raw`\eta\lambda W(-1)`,
  linear_characteristic: String.raw`r-a+\eta\lambda W(r^{-1})=0`,
  scalar_model: String.raw`g_t=\lambda x_t,\qquad z_t=\sum_{k\ge0}w(k)g_{t-k},\qquad x_{t+1}=a x_t-\eta z_t`,
  mean_age: String.raw`\bar{k}=\frac{\sum_k k\,w(k)}{\sum_k w(k)}`,
  noise_gain: String.raw`N=\frac{\sum_k w(k)^2}{\left(\sum_k w(k)\right)^2}`,
  delayed_mode: String.raw`x_{t-k}=r^{-k}x_t`,
  filtered_mode: String.raw`z_t=\lambda x_t W(r^{-1})`,
  delta_x: String.raw`\Delta X=-\eta U`,
  taylor: String.raw`L(X+\Delta X)-L(X)
    =-\eta\langle \nabla L,U\rangle
    +\frac{\eta^2}{2}\langle U,HU\rangle
    +O\!\left(\eta^3\lVert U\rVert^3\right)`,
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
