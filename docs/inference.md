# Inference in Gen.clj

Inference is the process of reasoning backwards from observations to the latent variables that could have generated them. Gen.clj provides building blocks for implementing various inference algorithms.

## Table of Contents

1. [The Inference Problem](#the-inference-problem)
2. [Key Operations](#key-operations)
3. [Importance Sampling](#importance-sampling)
4. [Metropolis-Hastings MCMC](#metropolis-hastings-mcmc)
5. [Model Comparison](#model-comparison)
6. [Posterior Predictive Checking](#posterior-predictive-checking)
7. [Building Custom Inference](#building-custom-inference)

## The Inference Problem

Given:
- A **model** (generative function) that makes random choices
- **Observations** (constraints on some of those choices)

Find:
- The **posterior distribution** over the unconstrained (latent) choices

### Bayes' Rule

```
P(latent | observed) ∝ P(observed | latent) × P(latent)
     posterior           likelihood           prior
```

In Gen.clj terms:
- **P(latent)**: The prior, encoded by the model's distributions
- **P(observed | latent)**: The likelihood, from traced choices
- **P(latent | observed)**: The posterior, what we want to compute

## Key Operations

### gf/generate - Constrained Execution

The fundamental inference primitive:

```clojure
(gf/generate model args constraints)
;; Returns: {:trace ... :weight ...}
```

- Runs the model with some choices fixed to given values
- Returns a trace and an **importance weight**
- Weight = log probability of the constrained choices

```clojure
(def model
  (gen []
    (let [bias (dynamic/trace! :bias dist/beta 1 1)]
      (dotimes [i 5]
        (dynamic/trace! [:flip i] dist/bernoulli bias))
      bias)))

;; Constrain the flips, infer the bias
(def constraints {[:flip 0] true
                  [:flip 1] true
                  [:flip 2] false
                  [:flip 3] true
                  [:flip 4] true})

(def result (gf/generate model [] constraints))
;; (:trace result) - trace with bias sampled from prior, flips constrained
;; (:weight result) - log P(flips | bias)
```

### trace/update - Modify Existing Trace

Propose changes to a trace:

```clojure
(trace/update trace new-constraints)
;; Returns: {:trace ... :weight ... :discard ...}
```

- Weight = log ratio of new/old probability
- Discard = the old values that were replaced

### gf/assess - Compute Log Probability

Compute log probability without creating a full trace:

```clojure
(gf/assess model args complete-choices)
;; Returns: {:weight ... :retval ...}
```

All choices must be specified. More efficient than `generate` when you just need the probability.

## Importance Sampling

The simplest inference algorithm: sample from the prior, weight by likelihood.

### Basic Importance Sampling

```clojure
(defn importance-sample [model args observations n-samples]
  (repeatedly n-samples
              #(gf/generate model args observations)))

;; Returns list of {:trace ... :weight ...}
```

### Importance Resampling

Sample once from the weighted population:

```clojure
(require '[gen.inference.importance :as importance])

(importance/resampling model args observations n-particles)
;; Returns: {:trace ... :weight ...}
;; - trace: one resampled trace
;; - weight: log marginal likelihood estimate
```

### Complete Example: Coin Bias Inference

```clojure
(def coin-model
  (gen [n]
    (let [bias (dynamic/trace! :bias dist/beta 1 1)]
      (dotimes [i n]
        (dynamic/trace! [:flip i] dist/bernoulli bias))
      bias)))

;; Observed: 7 heads, 3 tails
(def observations
  (into {} (concat
            (for [i (range 7)] [[:flip i] true])
            (for [i (range 7 10)] [[:flip i] false]))))

;; Run importance sampling
(defn infer-bias [n-particles]
  (let [result (importance/resampling coin-model [10] observations n-particles)
        choices (trace/get-choices (:trace result))]
    (choicemap/get-value choices :bias)))

;; Multiple runs to see distribution
(repeatedly 10 #(infer-bias 1000))
;; => (0.72 0.68 0.71 0.69 0.74 ...)  ; Should cluster around 0.7
```

### When Importance Sampling Works

✅ Good for:
- Low-dimensional problems
- When prior is close to posterior
- Quick approximations

❌ Poor for:
- High-dimensional problems
- When prior and posterior differ greatly
- Complex multimodal posteriors

## Metropolis-Hastings MCMC

For more complex problems, use Markov chain Monte Carlo.

### The MH Algorithm

1. Start with an initial trace
2. Propose a change to some choices
3. Accept or reject based on probability ratio
4. Repeat to generate samples from posterior

### Simple MH Implementation

```clojure
(defn mh-step
  "Single MH step with random walk proposal on given address."
  [trace addr proposal-std]
  (let [choices (trace/get-choices trace)
        current-val (choicemap/get-value choices addr)
        ;; Random walk proposal
        proposed-val (+ current-val (dist/normal 0 proposal-std))
        ;; Update trace
        update-result (trace/update trace {addr proposed-val})
        ;; Accept/reject
        log-accept (min 0.0 (:weight update-result))]
    (if (< (Math/log (rand)) log-accept)
      (:trace update-result)  ; Accept
      trace)))                 ; Reject

(defn run-mh
  "Run MH chain for n steps."
  [initial-trace addr proposal-std n-steps]
  (loop [trace initial-trace
         samples []
         i 0]
    (if (< i n-steps)
      (let [new-trace (mh-step trace addr proposal-std)
            value (choicemap/get-value (trace/get-choices new-trace) addr)]
        (recur new-trace (conj samples value) (inc i)))
      samples)))
```

### Example: Linear Regression with MH

```clojure
(def linear-model
  (gen [xs]
    (let [slope (dynamic/trace! :slope dist/normal 0 2)
          intercept (dynamic/trace! :intercept dist/normal 0 5)
          noise (dynamic/trace! :noise dist/gamma 1 1)]
      (doseq [[i x] (map-indexed vector xs)]
        (dynamic/trace! [:y i] dist/normal
                        (+ (* slope x) intercept)
                        noise)))))

(def xs [0 1 2 3 4])
(def ys [1.1 3.0 4.9 7.1 8.8])
(def observations (into {} (map-indexed (fn [i y] [[:y i] y]) ys)))

;; Initialize
(def initial-trace
  (:trace (gf/generate linear-model [xs] observations)))

;; Run MH on slope
(def slope-samples
  (run-mh initial-trace :slope 0.5 1000))

;; Analyze results
(defn mean [xs] (/ (reduce + xs) (count xs)))
(defn variance [xs]
  (let [m (mean xs)]
    (/ (reduce + (map #(Math/pow (- % m) 2) xs)) (count xs))))

(println "Mean slope:" (mean slope-samples))
(println "Std slope:" (Math/sqrt (variance slope-samples)))
```

### Multi-site MH

Update multiple addresses in one step:

```clojure
(defn multi-site-mh-step [trace addrs proposal-stds]
  (let [choices (trace/get-choices trace)
        proposals (into {}
                        (map (fn [[addr std]]
                               [addr (+ (choicemap/get-value choices addr)
                                       (dist/normal 0 std))])
                             (map vector addrs proposal-stds)))
        update-result (trace/update trace proposals)
        log-accept (min 0.0 (:weight update-result))]
    (if (< (Math/log (rand)) log-accept)
      (:trace update-result)
      trace)))
```

### Block Updates

For correlated parameters, update them together:

```clojure
(defn block-update [trace param-addrs]
  ;; Propose all parameters jointly
  (let [proposals (into {}
                        (map (fn [addr]
                               [addr (dist/normal 0 1)])  ; Fresh samples
                             param-addrs))
        result (trace/update trace proposals)]
    (if (< (Math/log (rand)) (min 0 (:weight result)))
      (:trace result)
      trace)))
```

## Model Comparison

Compare models using marginal likelihood estimates.

### Bayes Factor

```
BF(M1, M2) = P(data | M1) / P(data | M2)
```

- BF > 1: M1 is favored
- BF < 1: M2 is favored
- BF > 10: Strong evidence for M1

### Computing Marginal Likelihood

Importance sampling provides an estimate:

```clojure
(defn estimate-marginal-likelihood [model args observations n-samples]
  (let [results (repeatedly n-samples
                            #(gf/generate model args observations))
        log-weights (map :weight results)
        max-w (apply max log-weights)
        ;; Log-sum-exp trick for numerical stability
        weights (map #(Math/exp (- % max-w)) log-weights)]
    (+ max-w (Math/log (/ (reduce + weights) n-samples)))))
```

### Example: Linear vs Quadratic

```clojure
(def linear-model
  (gen [xs]
    (let [a (dynamic/trace! :a dist/normal 0 2)
          b (dynamic/trace! :b dist/normal 0 5)]
      (doseq [[i x] (map-indexed vector xs)]
        (dynamic/trace! [:y i] dist/normal (+ (* a x) b) 0.5)))))

(def quadratic-model
  (gen [xs]
    (let [a (dynamic/trace! :a dist/normal 0 1)
          b (dynamic/trace! :b dist/normal 0 2)
          c (dynamic/trace! :c dist/normal 0 5)]
      (doseq [[i x] (map-indexed vector xs)]
        (dynamic/trace! [:y i] dist/normal
                        (+ (* a x x) (* b x) c)
                        0.5)))))

;; Data generated from linear relationship
(def xs [0 1 2 3 4])
(def ys [1.0 3.1 4.9 7.0 9.1])
(def obs (into {} (map-indexed (fn [i y] [[:y i] y]) ys)))

;; Compare models
(def ml-linear (estimate-marginal-likelihood linear-model [xs] obs 1000))
(def ml-quad (estimate-marginal-likelihood quadratic-model [xs] obs 1000))

(println "Log ML (linear):" ml-linear)
(println "Log ML (quadratic):" ml-quad)
(println "Bayes factor (linear/quad):" (Math/exp (- ml-linear ml-quad)))
```

## Posterior Predictive Checking

Validate your model by generating new data from inferred parameters.

### Process

1. Infer parameters from observed data
2. Generate new data using inferred parameters
3. Compare generated data to observed data

```clojure
(defn predict-new-data [model xs inferred-trace param-addrs new-xs]
  (let [;; Extract inferred parameters
        choices (trace/get-choices inferred-trace)
        constraints (into {}
                          (map (fn [addr]
                                 [addr (choicemap/get-value choices addr)])
                               param-addrs))
        ;; Generate with fixed parameters
        result (gf/generate model [new-xs] constraints)
        new-trace (:trace result)]
    ;; Extract generated y values
    (mapv #(choicemap/get-value (trace/get-choices new-trace) [:y %])
          (range (count new-xs)))))

;; Infer
(def inferred (:trace (importance/resampling
                        linear-model [xs] obs 1000)))

;; Predict
(def new-xs [5 6 7 8])
(def predictions (predict-new-data
                   linear-model xs inferred
                   [:slope :intercept :noise]
                   new-xs))

(println "Predictions for" new-xs ":" predictions)
```

### Multiple Posterior Samples

Get uncertainty by using multiple posterior samples:

```clojure
(defn posterior-predictive [model xs obs param-addrs new-xs n-samples]
  (repeatedly n-samples
              #(let [trace (:trace (importance/resampling
                                     model [xs] obs 100))]
                 (predict-new-data model xs trace param-addrs new-xs))))

(def all-predictions
  (posterior-predictive linear-model xs obs
                        [:slope :intercept :noise]
                        new-xs 20))

;; Compute mean and std of predictions at each point
(defn transpose [matrix]
  (apply map vector matrix))

(doseq [[i preds] (map-indexed vector (transpose all-predictions))]
  (println "x =" (nth new-xs i)
           "mean =" (mean preds)
           "std =" (Math/sqrt (variance preds))))
```

## Building Custom Inference

### Generic Framework

```clojure
(defn custom-inference
  [model args observations
   {:keys [n-iterations
           proposal-fn
           initialize-fn
           callback-fn]}]

  ;; Initialize
  (let [initial-trace (or (initialize-fn model args observations)
                          (:trace (gf/generate model args observations)))]

    ;; Main loop
    (loop [trace initial-trace
           samples []
           i 0]
      (if (< i n-iterations)
        (let [;; Propose new trace
              proposed (proposal-fn trace)
              ;; Update
              result (trace/update trace proposed)
              ;; Accept/reject
              new-trace (if (< (Math/log (rand))
                              (min 0 (:weight result)))
                         (:trace result)
                         trace)]
          ;; Callback for monitoring
          (when callback-fn
            (callback-fn i new-trace))
          ;; Continue
          (recur new-trace
                 (conj samples new-trace)
                 (inc i)))
        samples))))
```

### Adaptive Proposals

Tune proposal distributions based on acceptance rate:

```clojure
(defn adaptive-mh
  [trace addr initial-std target-accept-rate n-steps]
  (loop [trace trace
         std initial-std
         accepts 0
         samples []
         i 0]
    (if (< i n-steps)
      (let [;; Propose
            current (choicemap/get-value (trace/get-choices trace) addr)
            proposed (+ current (dist/normal 0 std))
            result (trace/update trace {addr proposed})
            accepted? (< (Math/log (rand)) (min 0 (:weight result)))
            new-trace (if accepted? (:trace result) trace)
            ;; Adapt every 100 steps
            new-std (if (and (pos? i) (zero? (mod i 100)))
                     (let [rate (/ accepts 100)]
                       (cond
                         (< rate (- target-accept-rate 0.1)) (* std 0.9)
                         (> rate (+ target-accept-rate 0.1)) (* std 1.1)
                         :else std))
                     std)
            new-accepts (if (zero? (mod i 100))
                         (if accepted? 1 0)
                         (if accepted? (inc accepts) accepts))]
        (recur new-trace new-std new-accepts
               (conj samples (choicemap/get-value
                               (trace/get-choices new-trace) addr))
               (inc i)))
      {:samples samples :final-std std})))
```

## Common Pitfalls

### 1. Insufficient Particles

```clojure
;; Too few - high variance
(importance/resampling model args obs 10)

;; Better
(importance/resampling model args obs 1000)
```

### 2. Poor Proposals

```clojure
;; Proposal too narrow - slow mixing
(mh-step trace :x 0.001)

;; Proposal too wide - low acceptance
(mh-step trace :x 100)

;; Need to tune!
```

### 3. Ignoring Burn-in

```clojure
;; Discard initial samples
(def samples (run-mh initial 1000))
(def after-burnin (drop 200 samples))
```

### 4. Correlated Samples

```clojure
;; Thin the chain
(def thinned (take-nth 10 samples))
```

## Summary

| Algorithm | Pros | Cons | Use When |
|-----------|------|------|----------|
| Importance Sampling | Simple, parallelizable | High variance in high-D | Quick estimates, low-D |
| MH MCMC | Works in high-D | Sequential, tuning needed | Complex posteriors |
| Importance Resampling | Easy to implement | May need many particles | General purpose |

## See Also

- [Generative Functions](generative-functions.md) - Defining models
- [Traces and Choice Maps](traces-and-choicemaps.md) - Working with traces
- [API Reference](api-reference.md) - Complete API
