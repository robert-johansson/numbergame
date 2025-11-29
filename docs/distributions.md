# Probability Distributions in Gen.clj

Gen.clj provides probability distributions as first-class generative functions. They can be sampled from directly, used with `trace!` for probabilistic modeling, and their log probabilities can be computed.

## Distribution Backends

Gen.clj supports two distribution backends:

### Apache Commons Math (Recommended for JVM)

```clojure
(require '[gen.distribution.commons-math :as dist])
```

Full-featured, JVM-only. Best for most use cases.

### Kixi Stats (Cross-platform)

```clojure
(require '[gen.distribution.kixi :as dist])
```

Works in both Clojure and ClojureScript. Slightly fewer distributions.

## Available Distributions

### Discrete Distributions

#### Bernoulli

Binary outcome (true/false) with probability `p`.

```clojure
(dist/bernoulli p)
;; p: probability of true (0 to 1)
;; Returns: true or false

;; Examples
(dist/bernoulli 0.5)   ; fair coin
(dist/bernoulli 0.9)   ; biased coin (90% heads)
(dist/bernoulli 0.0)   ; always false
```

**Use cases:** Coin flips, binary decisions, presence/absence indicators.

#### Binomial

Number of successes in `n` independent Bernoulli trials.

```clojure
(dist/binomial n p)
;; n: number of trials
;; p: probability of success per trial
;; Returns: integer from 0 to n

;; Examples
(dist/binomial 10 0.5)  ; 10 fair coin flips
(dist/binomial 100 0.01) ; rare events in 100 trials
```

**Use cases:** Counting successes, sampling with replacement.

#### Uniform Discrete

Integer uniformly distributed in range `[low, high]` (inclusive).

```clojure
(dist/uniform-discrete low high)
;; low: minimum value (inclusive)
;; high: maximum value (inclusive)
;; Returns: integer in [low, high]

;; Examples
(dist/uniform-discrete 1 6)    ; die roll
(dist/uniform-discrete 0 99)   ; random two-digit number
```

**Use cases:** Dice, random selection from finite set.

#### Categorical

Sample from a discrete distribution over indices or keys.

```clojure
;; With vector (returns index 0, 1, 2, ...)
(dist/categorical [0.5 0.3 0.2])

;; With map (returns key)
(dist/categorical {:a 0.5 :b 0.3 :c 0.2})

;; Probabilities are automatically normalized
(dist/categorical [1 2 3])  ; same as [1/6 2/6 3/6]
```

**Use cases:** Model selection, discrete choices, mixture models.

### Continuous Distributions

#### Normal (Gaussian)

Bell curve distribution with mean `mu` and standard deviation `sigma`.

```clojure
(dist/normal mu sigma)
;; mu: mean (center of distribution)
;; sigma: standard deviation (spread)
;; Returns: real number

;; Examples
(dist/normal 0 1)      ; standard normal
(dist/normal 100 15)   ; IQ distribution
(dist/normal 0 0.1)    ; tight around 0
```

**Properties:**
- Symmetric around mean
- 68% of values within 1 sigma of mean
- 95% within 2 sigma
- 99.7% within 3 sigma

**Use cases:** Measurement noise, natural variation, priors on real-valued parameters.

#### Uniform

Real number uniformly distributed in range `[a, b]`.

```clojure
(dist/uniform a b)
;; a: minimum value
;; b: maximum value
;; Returns: real in [a, b]

;; Examples
(dist/uniform 0 1)     ; unit interval
(dist/uniform -1 1)    ; symmetric around 0
(dist/uniform 0 (* 2 Math/PI))  ; random angle
```

**Use cases:** Uninformative priors on bounded intervals, random angles/phases.

#### Beta

Distribution over `[0, 1]`, parameterized by shape parameters `alpha` and `beta`.

```clojure
(dist/beta alpha beta)
;; alpha: shape parameter > 0
;; beta: shape parameter > 0
;; Returns: real in (0, 1)

;; Examples
(dist/beta 1 1)     ; uniform on [0,1]
(dist/beta 2 2)     ; symmetric, peaked at 0.5
(dist/beta 2 5)     ; skewed toward 0
(dist/beta 5 2)     ; skewed toward 1
(dist/beta 0.5 0.5) ; U-shaped (peaked at 0 and 1)
```

**Properties:**
- Mean = alpha / (alpha + beta)
- Higher alpha+beta = more concentrated

**Use cases:** Probabilities, proportions, rates. Conjugate prior for Bernoulli.

#### Gamma

Distribution over positive reals, parameterized by `shape` and `scale`.

```clojure
(dist/gamma shape scale)
;; shape: k > 0 (also called alpha)
;; scale: theta > 0
;; Returns: positive real

;; Examples
(dist/gamma 1 1)     ; exponential(1)
(dist/gamma 2 1)     ; peaked around 1
(dist/gamma 9 0.5)   ; peaked around 4
```

**Properties:**
- Mean = shape * scale
- Variance = shape * scale^2
- shape=1 gives exponential distribution

**Use cases:** Waiting times, variances, positive quantities. Conjugate prior for Poisson rate.

#### Exponential

Time until first event in a Poisson process.

```clojure
(dist/exponential rate)
;; rate: lambda > 0 (events per unit time)
;; Returns: positive real

;; Examples
(dist/exponential 1)    ; mean = 1
(dist/exponential 0.5)  ; mean = 2
(dist/exponential 10)   ; mean = 0.1
```

**Properties:**
- Mean = 1/rate
- Memoryless: P(X > s+t | X > s) = P(X > t)

**Use cases:** Waiting times, durations, inter-arrival times.

#### Student's t

Heavy-tailed distribution, approaches normal as degrees of freedom increase.

```clojure
(dist/student-t nu)           ; standard (location=0, scale=1)
(dist/student-t nu loc scale) ; with location and scale

;; nu: degrees of freedom > 0
;; Returns: real number

;; Examples
(dist/student-t 1)       ; Cauchy (very heavy tails)
(dist/student-t 3)       ; heavy tails
(dist/student-t 30)      ; approximately normal
```

**Use cases:** Robust regression, handling outliers, small sample sizes.

#### Cauchy

Extremely heavy-tailed distribution. Has no mean or variance.

```clojure
(dist/cauchy location scale)
;; location: center of distribution
;; scale: half-width at half-maximum
;; Returns: real number

(dist/cauchy 0 1)  ; standard Cauchy
```

**Use cases:** Modeling extreme outliers, certain physical phenomena.

## Using Distributions

### Direct Sampling

Call a distribution like a function to get a sample:

```clojure
(dist/normal 0 1)           ; => 0.342...
(dist/bernoulli 0.7)        ; => true
(repeatedly 5 #(dist/uniform-discrete 1 6))  ; => (3 1 6 2 4)
```

### With trace! (Traced Sampling)

Inside a `gen` function, use `trace!` to record the choice:

```clojure
(def my-model
  (gen []
    (let [x (dynamic/trace! :x dist/normal 0 1)
          y (dynamic/trace! :y dist/normal x 1)]
      (+ x y))))
```

### Computing Log Probabilities

Distributions implement the `LogPDF` protocol:

```clojure
(require '[gen.distribution :as d])

;; Create a distribution object
(def normal-dist (dist/normal-distribution 0 1))

;; Compute log probability density
(d/logpdf normal-dist 0.0)   ; => -0.919... (log of ~0.4)
(d/logpdf normal-dist 2.0)   ; => -2.919... (log of ~0.05)

;; For discrete distributions, this is log probability mass
(def bern-dist (dist/bernoulli-distribution 0.7))
(d/logpdf bern-dist true)    ; => -0.357... (log of 0.7)
(d/logpdf bern-dist false)   ; => -1.204... (log of 0.3)
```

### Distribution Objects vs Generative Functions

There are two forms of each distribution:

```clojure
;; Generative function form (for use with trace!)
dist/normal        ; takes args, returns sample
(dist/normal 0 1)  ; => 0.342...

;; Distribution object form (for logpdf, etc.)
(dist/normal-distribution 0 1)  ; returns distribution object
(d/logpdf (dist/normal-distribution 0 1) 0.5)  ; => -1.044...
```

## Distribution Properties

### Log Probability Density/Mass

All distributions can compute `logpdf`:

| Distribution | logpdf formula |
|--------------|----------------|
| Bernoulli | log(p) if true, log(1-p) if false |
| Normal | -0.5*((x-μ)/σ)² - log(σ) - 0.5*log(2π) |
| Uniform | -log(b-a) if x∈[a,b], else -∞ |
| Beta | (α-1)log(x) + (β-1)log(1-x) - logB(α,β) |
| Gamma | (k-1)log(x) - x/θ - k*log(θ) - log(Γ(k)) |

### Support (Valid Range)

| Distribution | Support |
|--------------|---------|
| Bernoulli | {true, false} |
| Binomial | {0, 1, ..., n} |
| Categorical | {0, 1, ..., n-1} or map keys |
| Normal | (-∞, +∞) |
| Uniform | [a, b] |
| Beta | (0, 1) |
| Gamma | (0, +∞) |
| Exponential | (0, +∞) |

## Common Patterns

### Hierarchical Models

Use distributions to build hierarchical priors:

```clojure
(def hierarchical-model
  (gen [n]
    ;; Hyperprior
    (let [mu (dynamic/trace! :mu dist/normal 0 10)
          sigma (dynamic/trace! :sigma dist/gamma 1 1)]
      ;; Group-level parameters drawn from hyperprior
      (dotimes [i n]
        (dynamic/trace! [:theta i] dist/normal mu sigma)))))
```

### Mixture Models

Use categorical to select among distributions:

```clojure
(def mixture-model
  (gen []
    (let [;; Mixture weights
          component (dynamic/trace! :component dist/categorical [0.3 0.7])]
      ;; Sample from selected component
      (case component
        0 (dynamic/trace! :x dist/normal -2 1)
        1 (dynamic/trace! :x dist/normal 2 1)))))
```

### Conjugate Priors

Some prior-likelihood pairs have closed-form posteriors:

| Prior | Likelihood | Posterior |
|-------|------------|-----------|
| Beta(α,β) | Bernoulli | Beta(α+successes, β+failures) |
| Gamma(α,β) | Poisson | Gamma(α+Σx, β+n) |
| Normal(μ₀,σ₀) | Normal (known σ) | Normal(...) |

## Tips and Best Practices

1. **Choose appropriate priors**: Match the support of your prior to the domain of your parameter.

2. **Use log probabilities**: Gen.clj works in log space to avoid numerical underflow.

3. **Parameterization matters**: Some distributions have multiple parameterizations (e.g., Gamma can use rate or scale).

4. **Test with simulation**: Before doing inference, simulate from your model to verify it generates sensible data.

5. **Watch for numerical issues**: Very small or very large parameter values can cause numerical problems.

## See Also

- [Generative Functions](generative-functions.md) - Using distributions in models
- [Inference](inference.md) - Computing posteriors
- [API Reference](api-reference.md) - Complete distribution API
