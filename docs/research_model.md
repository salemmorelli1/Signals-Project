# Executed Joint Torus-Valued State-Space Model

## Filtering distribution

For two emitters, the implemented particle approximation targets

\[
p(\boldsymbol\ell_t,\mathbf f_t,\mathbf z_t,
  \boldsymbol\phi_t,\mathbf h_t,\mathbf H_t\mid y_{1:t}),
\]

where log amplitude and frequency are Euclidean, the hop state is discrete,
phase lies on the flat torus \(\mathbb T^2\), the complex tapped-delay channel
is latent, and \(\mathbf H_t\) is the source lag buffer.

The simulator's realized state and channel paths are not supplied to
`JointParticleFilter.run`. They are used only after inference to calculate
frequency MSE, circular phase RMSE, and channel NMSE.

## Observation law

For clean complex sample \(\mu_t\), the real and imaginary residuals each obey

\[
\varepsilon = G + C,\qquad
G\sim\mathcal N(0,\sigma_G^2),
\]

with \(C\) a centered Cauchy variable truncated to \([-A,A]\). Therefore,

\[
p_\varepsilon(e)=\int_{-A}^{A}
  p_G(e-\tau)p_C(\tau)\,d\tau.
\]

The code evaluates this convolution by Gauss-Legendre quadrature. Normalizers,
quadrature weights, and the Gaussian exponent are combined with `logsumexp`.
The likelihood has no additive density floor.

## Switching transition

Each emitter has a four-channel frequency alphabet. Hop states evolve at
scheduled opportunities. If a hop occurs, frequency is centered on the new
channel. Otherwise, its deviation from the current channel follows an AR(1)
transition. This gives non-negligible prior mass to tactical hops while
retaining continuous local dynamics.

Phase propagation is candidate dependent:

\[
\phi_{k,t}=\left(\phi_{k,t-1}+2\pi f_{k,t}\Delta t+eta_{k,t}\right)
\bmod 2\pi.
\]

The analytical frequency score includes the derivative induced through this
phase-transition location.

## Flat-torus MALA

For phase score \(s_\phi\), the tangent proposal is

\[
v=\tfrac12\epsilon^2s_\phi+\epsilon\xi,
\qquad
\phi^*=\operatorname{Exp}_\phi(v)=(\phi+v)\bmod 2\pi.
\]

The implemented proposal density is the wrapped normal

\[
q_{\mathbb T}(\phi^*\mid\phi)=
\sum_{k\in\mathbb Z}
\mathcal N(\phi^*-\phi+2\pi k;
\tfrac12\epsilon^2s_\phi,\epsilon^2).
\]

For the flat torus, the metric tensor is the identity and the exponential-map
Jacobian is one under Haar volume. Detailed balance follows from the ordinary
Metropolis-Hastings identity after evaluating the actual wrapped forward and
reverse densities.

## Sequential inference

The resample-move filter performs:

1. joint transition propagation;
2. exact convolution likelihood weighting;
3. stable normalization and particle-ESS calculation;
4. systematic resampling below the ESS threshold or at a forced move time;
5. one exact-target MALA rejuvenation step; and
6. weighted posterior summaries.

Particle ESS is an importance-weight diagnostic. It is not MCMC chain ESS, and
rank-normalized R-hat is not reported for a one-step rejuvenation kernel.

## Score architectures

The analytical method uses the exact joint target gradient. The amortized
method uses a SiLU random-feature network trained on independent exact score
targets. Both methods use the exact configured target in the acceptance ratio.
Consequently, score approximation changes proposal efficiency, not the formal
conditional target.

## Evidence scope

The implementation and full factorial experiment establish operationally
representative simulation validation. See
`docs/OPERATIONAL_VALIDATION_PROTOCOL.md` for the evidence required before any
field or operational validation statement is defensible.
