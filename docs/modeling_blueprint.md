# Bayesian Regime Project Blueprint

## Core Question

Can latent market regimes inferred from macro indicators improve dynamic allocation
across diversified ETFs relative to static or non-regime baselines?

## Working Dataset

- Sample frequency: monthly
- Sample window: `2006-03-31` to `2026-04-30`
- Asset return columns:
  - `SPY_ret`, `EFA_ret`, `TLT_ret`, `GLD_ret`, `DBC_ret`, `VNQ_ret`
- Macro signal columns:
  - `T10Y2Y_z`, `VIX_z`, `CREDIT_z`, `UNRATE_z`, `UNRATE_CHG_z`,
    `INDPRO_YOY_z`, `CFNAI_z`

## Recommended Modeling Scope

Start with a Bayesian Hidden Markov Model with `K = 2` and `K = 3` candidate
regimes.

- Latent state: `z_t`, the unobserved market regime at month `t`
- Transition model: first-order Markov chain with a sticky prior that prefers
  regime persistence
- Observation model:
  - Primary version: multivariate Gaussian model on the six asset return series
  - Secondary interpretation layer: macro variables are used to explain and label
    the inferred regimes

This split is the safest first version because:

- it keeps inference dimension manageable
- it directly models the quantity needed for portfolio construction
- it still lets the team interpret each regime economically using macro data

## Why Not Start With Joint Returns + Macro HMM

Jointly modeling all 13 observed variables is possible, but it is more fragile:

- larger covariance matrices are harder to estimate
- MCMC will be slower and less stable
- interpretation becomes harder if the first model already struggles to mix

If the returns-only model works well, you can extend later to a richer joint
specification.

## Priors To Use

For a first implementation in PyMC:

- Initial state probabilities: Dirichlet prior
- Transition matrix rows: Dirichlet prior with extra mass on self-transitions
- Regime-specific return means: weakly informative Normal priors centered near 0
- Regime-specific covariance matrices: LKJ prior with separate scale priors

## Evaluation Design

Use a train/test or expanding-window backtest.

- In-sample goal: recover interpretable latent regimes
- Out-of-sample goal: improve allocation decisions

Compare against:

1. Equal-weight portfolio
2. Static `60/40` proxy, if desired
3. Rolling historical mean-variance without regimes
4. Optional non-Bayesian Gaussian HMM baseline

## Project Deliverables

1. Dataset audit and exploratory analysis
2. Baseline model
3. Bayesian HMM fit for `K = 2` and `K = 3`
4. Regime interpretation plots
5. Allocation rule using posterior regime probabilities
6. Backtest metrics and discussion

## Suggested Team Sequence

1. Finalize preprocessing and EDA
2. Build a simple baseline HMM
3. Implement Bayesian HMM on returns only
4. Interpret regimes using macro overlays
5. Build allocation and backtest layer
6. Write results and limitations

## Immediate Next Decision

Unless the team strongly prefers a more ambitious first model, proceed with:

- Bayesian HMM on the six asset return series
- macro variables used for interpretation and reporting
- `K = 2` and `K = 3` tested side by side
