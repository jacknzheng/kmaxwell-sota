import fs from "node:fs";
import katex from "katex";

const formulas = {
  normalized_curvature_fit: String.raw`\overline{\widetilde{\lambda}}
    =1.32\,s^{-1.28}`,
  two_component_history: String.raw`G_t=A_{\mathrm{slow}}s_t+A_{\mathrm{fast}}f_t`,
  local_fourier_coefficient: String.raw`A_{\omega,t}=\frac{2}{L}
    \sum_{\ell=0}^{L-1}G_{t-L+1+\ell}
    e^{-i\omega(t-L+1+\ell)}`,
  one_temporal_component: String.raw`G_t^{(\omega)}=\operatorname{Re}\!\left(A_\omega e^{i\omega t}\right)`,
  delayed_component_derivation: String.raw`\begin{aligned}
    G_{t-k}^{(\omega)}
      &=\operatorname{Re}\!\left(A_\omega e^{i\omega(t-k)}\right)\\
      &=\operatorname{Re}\!\left(e^{-i\omega k}A_\omega e^{i\omega t}\right),\\[4pt]
    Z_t^{(\omega)}
      &=\sum_{k\ge0}w_kG_{t-k}^{(\omega)}\\
      &=\operatorname{Re}\!\left(H(\omega)A_\omega e^{i\omega t}\right),
      \qquad H(\omega)=\sum_{k\ge0}w_ke^{-i\omega k}.
    \end{aligned}`,
  two_temporal_components: String.raw`Z_t=\operatorname{Re}\!\left(
    H(\omega_1)A_1e^{i\omega_1t}+H(\omega_2)A_2e^{i\omega_2t}
    \right)`,
  H_def: String.raw`H(\omega)=\sum_{k\ge0}w_ke^{-i\omega k}`,
  muon_update: String.raw`P_t=\operatorname{msign}(Z_t)`,
  response_ratio: String.raw`\frac{H(\omega_1)}{H(\omega_2)}`,
  scalar_quadratic_derivation: String.raw`\begin{aligned}
    L(x)&=\tfrac12\lambda x^2,
      &\nabla L(x_t)&=\lambda x_t,\\
    m_t&=\sum_{k\ge0}w_k\lambda x_{t-k},\\
    x_{t+1}&=(1-\eta\rho)x_t-\eta m_t\\
      &=a x_t-\eta\lambda\sum_{k\ge0}w_kx_{t-k},
      &a&=1-\eta\rho.
    \end{aligned}`,
  period2_full_derivation: String.raw`\begin{aligned}
    x_t&=c(-1)^t,
      &\frac{x_{t+1}}{x_t}&=-1,
      &\frac{x_{t-k}}{x_t}&=(-1)^k,\\
    -1&=a-\eta\lambda\sum_{k\ge0}(-1)^kw_k,\\
    \eta\lambda_{\mathrm{crit}}F&=1+a,
      &F=H(\pi)&=\sum_{k\ge0}(-1)^kw_k.
    \end{aligned}`,
  period2_state: String.raw`x_{t+1}=-x_t,\quad x_{t-k}=(-1)^kx_t`,
  period2_substitution: String.raw`-1=a-\eta\lambda\sum_{k\ge0}(-1)^kw_k`,
  Pw_def: String.raw`P_w=\operatorname{msign}(Z_w)`,
  taylor_local: String.raw`L(\theta-\eta P_w)-L(\theta)\approx
    -\eta\langle g,P_w\rangle
    +\frac{\eta^2}{2}\langle P_w,\mathcal H P_w\rangle`,
  controller_objective: String.raw`w^*(q_t)=\arg\min_w\mathbb E\!\left[
    -\eta_t\langle g_t,P_w\rangle
    +\frac{\eta_t^2}{2}\langle P_w,\mathcal H_tP_w\rangle
    \,\middle|\,q_t\right]`,
  general_characteristic: String.raw`r-a+\eta\lambda\sum_{k\ge0}w_kr^{-k}=0`,
  dc_normalization: String.raw`H(0)=\sum_{k\ge0}w_k=1`,
  lambda_inverse_F: String.raw`\lambda_{\mathrm{crit}}\propto F^{-1}`,
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
  flip_response: String.raw`F=H(\pi)=\sum_{k\ge 0}(-1)^k w_k`,
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
