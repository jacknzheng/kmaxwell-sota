# Momentum plus Newton: formulation

## 1. The question

A momentum kernel and an inverse Hessian can both suppress directions that
oscillate under gradient descent. This creates a concrete design problem:

> How much of a damped Newton direction can a single temporal kernel reconstruct
> from gradient history, and what inverse-curvature correction remains after
> that reconstruction?

The answer determines whether momentum and Newton are redundant. If a kernel
already reconstructs most of the Newton direction, the optimizer should apply
only the residual correction. If it reconstructs little, temporal estimation
and spatial curvature correction are mostly separate jobs.

This question has to be asked in the true Hessian eigenbasis. Activation
covariance is not a substitute for the Hessian here.

## 2. The objects being compared

Freeze a training state at step t. Let H be the Hessian at that state, with
eigenpairs

    H y_i = lambda_i y_i.

Throughout this document, lambda_i is curvature. The damping constant is rho.

Project each gradient in the recorded history onto each curvature direction:

    a_i(tau) = <g_(t-tau), y_i>,       tau = 0, 1, ..., T.

A temporal kernel with taps c_tau produces coefficient

    q_i(c) = sum_tau c_tau a_i(tau).

Now choose a reference momentum estimate z from the gradient history and solve

    (H + rho I) p = z.

The vector p is the damped Newton direction for that reference estimate. Its
coefficient along y_i is

    b_i = <p, y_i> = <z, y_i> / (lambda_i + rho)

whenever the eigenpair is exact.

The exercise is therefore a regression with a precise target:

    c* = argmin_c sum_(matrices i) [q_i(c) - b_i]^2.

One shared kernel must fit many curvature directions and many parameter
matrices. It cannot choose a separate gain for each lambda_i. It succeeds only
when the gradient time series carries curvature information in a consistent
form—for example, when sharper directions oscillate at predictably higher
frequencies.

## 3. The requested percentage

Let A contain the history coefficients a_i(tau), and let b contain the damped
Newton coefficients b_i. On held-out matrices and checkpoints, define

    residual fraction R_raw = ||A c* - b||^2 / ||b||^2,
    kernel-expressible share X_raw = 1 - R_raw.

This is the literal answer to “what percentage of damped Newton can a momentum
kernel express?” under the registered least-squares definition.

It contains one easy component. With strong damping, (H + rho I)^(-1) is close
to a scalar multiple of the identity over much of the spectrum. A lag-zero tap
can reproduce that common scalar without learning any curvature dependence.
Muon's matrix sign also discards a common positive scale. We therefore need a
second score.

For each matrix, fit the best scalar s multiplying the reference momentum z and
remove it from the Newton direction:

    p_aniso = p - s z.

Repeat the held-out regression against p_aniso. Its residual fraction R_aniso
measures the part of the anisotropic Newton correction that cannot be written
as a shared temporal kernel. The headline scientific answer is

    X_aniso = 1 - R_aniso.

Both numbers will be reported. X_raw answers the literal vector-reconstruction
question. X_aniso answers whether momentum has learned the curvature-dependent
part that could change Muon's update.

The present preregistration is X_aniso between 0.15 and 0.40. The pending
regression will fill in the measured value. A substantially larger value would
show that oscillation history encodes much more of inverse curvature than we
currently expect.

## 4. “Apply the remaining Newton” means adding a residual vector

The least-squares kernel produces

    p_K = sum_tau c*_tau g_(t-tau).

The part of damped Newton it misses is

    r = p - p_K.

This is the residual inverse-curvature correction. It is generally a
cross-basis operator because the Hessian eigenvectors need not coincide with
the singular vectors of the momentum matrix. There is no need to describe the
remainder as an unstable ratio such as b_i/q_i. The additive decomposition

    p = p_K + r

is exact.

For Muon, the corresponding update is

    Delta_Newton-Muon = -eta msign(p_K + r).

A family of partial corrections is

    Delta(alpha) = -eta msign(p_K + alpha r),     0 <= alpha <= 1.

Alpha = 0 uses only what the temporal kernel can express. Alpha = 1 recovers
the damped-Newton direction before matrix sign. Intermediate alpha values make
“apply the remaining percentage” operational without pretending that the
percentage is a scalar gain shared by all directions.

The vector-space percentage is not automatically the update-space percentage,
because matrix sign is nonlinear and invariant to common positive scale. Every
reported X must therefore be accompanied by the actual post-msign difference:

    cosine(msign(p_K), msign(p))

and, more importantly, curvature along each resulting update. The current CG
results demonstrate why: the polar factors have cosine about 0.99, yet the
damped inverse changes curvature along the update by roughly a factor of two.

## 5. Where order enters

There are two serious placements of temporal averaging and inverse curvature:

    A. p_A = P_t [sum_tau c_tau g_(t-tau)]

    B. p_B = sum_tau c_tau [P_(t-tau) g_(t-tau)],

where P_t = (H_t + rho I)^(-1).

If curvature is fixed, A and B are identical because both operations are
linear. Their difference in training measures Hessian drift over the kernel's
memory horizon. This gives the order decision a simple state variable: compare
the curvature-refresh time with the kernel's effective memory.

Matrix sign comes after the candidate pre-polar direction in the primary
Newton-Muon construction. Applying an inverse-curvature map after matrix sign
defines a different optimizer; it is not the Newton direction for the momentum
estimate. It can be tested as an alternative geometry, but it should not be
silently identified with A or B.

The meta-optimizer chooses between A and B, the kernel taps, damping rho, and
the residual strength alpha by held-out continuation loss at fixed compute.
The fixed-H identity above prevents a meaningless order search when the two
orders should agree.

## 6. Minibatch awareness

Curvature correction is useful only when curvature can be estimated reliably.
Let S_t summarize directional signal-to-noise ratio measured at the actual
minibatch size, using cross-replica gradient variation. Let alpha_t be a
monotone reliability gate:

    alpha_t = phi(S_t),       phi(0) = 0,       phi(infinity) = 1.

Then use

    Delta_t = -eta_t msign(p_K + alpha_t r_hat),

where r_hat is the residual computed from the current curvature estimate.
This one knob governs the interaction between momentum and Newton. At small
token count, S_t falls, alpha_t approaches zero, and the uncertain Newton
correction becomes passthrough. Momentum can lengthen its memory to improve the
temporal estimate. As token count grows and curvature estimates stabilize,
alpha_t permits more of the measured residual correction.

The other required boundary condition concerns conditioning. When the Hessian
is isotropic, its inverse differs from identity only by a scalar. Then r is zero
after removing common scale. In a stationary deterministic problem, the
optimal temporal kernel consequently collapses to its lag-zero tap: no
momentum is needed. This is a derived boundary of the same decomposition, not
an extra heuristic.

## 7. The complete selection problem

At each refresh, select

    (kernel c, order O, damping rho, residual gate alpha, refresh cadence C)

to minimize held-out continuation loss plus measured compute cost, subject to:

1. the kernel uses only past gradients;
2. inverse solves satisfy a recorded residual tolerance and exclude breakdowns;
3. alpha is determined from minibatch-scale reliability, rather than training
   step;
4. the isotropic deterministic limit returns lag-zero momentum;
5. the token-starved limit returns a scalar spatial passthrough;
6. candidate choices are evaluated after matrix sign, using loss and curvature
   along the realized update.

The high-dimensional solution is an instrument, not the final optimizer. The
research goal is to discover whether it compresses to one state variable—most
plausibly directional SNR relative to curvature staleness—and one interaction
knob alpha.

## 8. What the new measurement already establishes

The damped solves reached the residual gate in a median of three CG iterations.
At the three checkpoints, the pass counts were 72/72, 70/72, and 66/72;
breakdown events increased with training age from 0 to 4 to 11. This makes two
points immediately.

First, a residual-verified damped inverse can be cheap enough to refresh during
training: approximately three Hessian-vector products in these states. Second,
the fixed damping rule becomes less safe late in training, where more negative
curvature crosses the shift. Damping and solver choice must therefore respond
to state; breakdown is part of the signal, not an implementation nuisance.

Among solves passing the validity gate, matrix sign of the damped-Newton
direction remained close to matrix sign of the momentum estimate: rotation
cosine was about 0.99. Nevertheless, curvature along the realized update fell
by factors 0.47, 0.36, and 0.53 across the three checkpoints. A small angular
change can move the update away from the sharpest curvature directions. Update
cosine alone is consequently an inadequate measure of Newton's contribution.

The pending expressibility regression supplies the missing number: what share
of this anisotropic correction is already reproducible from gradient history.
Until it lands, the evidence supports a practical damped-Newton correction but
does not tell us whether the best momentum kernel makes most of that correction
redundant.
