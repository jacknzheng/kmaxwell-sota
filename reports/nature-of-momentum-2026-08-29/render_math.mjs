import fs from "node:fs";
import katex from "katex";

const formulas = {
  gradient_modes: String.raw`G_t=\sum_j \operatorname{Re}\!\left(A_j e^{i\omega_j t}\right)`,
  filtered_modes: String.raw`Z_t=\sum_{k\ge 0}w_kG_{t-k}
    =\sum_j\operatorname{Re}\!\left(H(\omega_j)A_je^{i\omega_jt}\right)`,
  H_def: String.raw`H(\omega)=\sum_{k\ge0}w_ke^{-i\omega k}`,
  muon_update: String.raw`U_t=\operatorname{msign}(Z_t)`,
  response_ratio: String.raw`\frac{H(\omega_1)}{H(\omega_2)}`,
  scalar_update: String.raw`x_{t+1}=a x_t-\eta\lambda\sum_{k\ge0}w_kx_{t-k}`,
  period2_state: String.raw`x_{t+1}=-x_t,\quad x_{t-k}=(-1)^kx_t`,
  period2_substitution: String.raw`-1=a-\eta\lambda\sum_{k\ge0}(-1)^kw_k`,
  Uw_def: String.raw`U_w=\operatorname{msign}(Z_w)`,
  taylor_local: String.raw`L(\theta-\eta U_w)-L(\theta)\approx
    -\eta\langle g,U_w\rangle
    +\frac{\eta^2}{2}\langle U_w,\mathcal H U_w\rangle`,
  controller_objective: String.raw`w^*(s_t)=\arg\min_w\mathbb E\!\left[
    -\eta_t\langle g_t,U_w\rangle
    +\frac{\eta_t^2}{2}\langle U_w,\mathcal H_tU_w\rangle
    \,\middle|\,s_t\right]`,
  general_characteristic: String.raw`r-a+\eta\lambda\sum_{k\ge0}w_kr^{-k}=0`,
  kernel: String.raw`Z_t=\sum_{k=0}^{t} w_t(k)\,G_{t-k}`,
  ema_kernel: String.raw`w(k)\propto\beta^k`,
  msign_scale: String.raw`\operatorname{msign}(cZ)=\operatorname{msign}(Z),\qquad c>0`,
  two_modes: String.raw`Z=\alpha A+\beta B`,
  ratio_modes: String.raw`\beta/\alpha`,
  W_def: String.raw`W(q)=\sum_{k\ge 0}w(k)q^k`,
  W_poly: String.raw`W(s)=\sum_{k\ge 0}w(k)s^k`,
  W_flip: String.raw`W(-1)`,
  eta_lambda_flip: String.raw`\eta\lambda W(-1)`,
  linear_characteristic: String.raw`r-a+\eta\lambda W(r^{-1})=0`,
  scalar_model: String.raw`g_t=\lambda x_t,\qquad z_t=\sum_{k\ge0}w(k)g_{t-k},\qquad x_{t+1}=a x_t-\eta z_t`,
  mean_age: String.raw`\bar{k}=\frac{\sum_k k\,w(k)}{\sum_k w(k)}`,
  noise_gain: String.raw`N=\frac{\sum_k w(k)^2}{\left(\sum_k w(k)\right)^2}`,
  periodic_input: String.raw`g_t=A\cos\!\left(\frac{2\pi t}{P}\right)`,
  endpoint_response: String.raw`\left|W\!\left(e^{-i2\pi/P}\right)\right|`,
  eos_sgd: String.raw`\eta\lambda_{\mathrm{crit}}=2`,
  flip_response: String.raw`F=\sum_{k\ge 0}(-1)^k w(k)`,
  eos_momentum: String.raw`\eta\lambda_{\mathrm{crit}}F=1+a`,
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
