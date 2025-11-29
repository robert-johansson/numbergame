# Generative Functions in Gen.clj

Generative functions are the core abstraction in Gen.clj. They are probabilistic programs that can make random choices, be executed to produce traces, and be conditioned on observations for inference.

## Table of Contents

1. [The gen Macro](#the-gen-macro)
2. [Making Random Choices with trace!](#making-random-choices-with-trace)
3. [Addresses](#addresses)
4. [Composing Generative Functions](#composing-generative-functions)
5. [Execution Modes](#execution-modes)
6. [The Generative Function Interface](#the-generative-function-interface)
7. [Common Patterns](#common-patterns)
8. [Best Practices](#best-practices)

## The gen Macro

The `gen` macro transforms ordinary Clojure code into a generative function:

```clojure
(require '[gen.dynamic :as dynamic :refer [gen]])

(def my-model
  (gen [arg1 arg2]
    ;; Body can contain any Clojure code
    (let [x (+ arg1 arg2)]
      ;; Plus traced random choices
      (dynamic/trace! :result dist/normal x 1))))
```

### Syntax

```clojure
;; Anonymous generative function
(gen [args] body)

;; Named (useful for recursion)
(gen my-name [args] body)
```

### What gen Does

The `gen` macro:
1. Creates a `DynamicDSLFunction` record
2. Rewrites `trace!` and `splice!` calls to interact with the tracing system
3. Makes the result callable as both a regular function and a generative function

## Making Random Choices with trace!

Inside a `gen` body, use `dynamic/trace!` to make traced random choices:

```clojure
(dynamic/trace! address distribution args...)
```

### Basic Usage

```clojure
(def simple-model
  (gen []
    ;; Sample from a normal distribution, record at address :x
    (let [x (dynamic/trace! :x dist/normal 0 1)]
      ;; Sample from another normal, using x as the mean
      (dynamic/trace! :y dist/normal x 0.5))))
```

### What trace! Does

1. **Samples** a value from the distribution
2. **Records** the choice at the given address in the trace
3. **Accumulates** the log probability into the trace's score
4. **Returns** the sampled value

### trace! vs Direct Sampling

```clojure
;; Traced - choice is recorded
(let [x (dynamic/trace! :x dist/normal 0 1)] ...)

;; Untraced - choice is NOT recorded
(let [x (dist/normal 0 1)] ...)
```

Untraced choices:
- Don't appear in the choice map
- Don't contribute to the score
- Can't be constrained during inference

## Addresses

Every traced choice needs a unique address. Addresses can be any Clojure value.

### Address Types

```clojure
;; Keywords (most common)
(dynamic/trace! :slope dist/normal 0 1)

;; Vectors (for indexed data)
(dynamic/trace! [:y 0] dist/normal 0 1)
(dynamic/trace! [:y 1] dist/normal 0 1)

;; Nested vectors
(dynamic/trace! [:group :a :param] dist/normal 0 1)

;; Strings
(dynamic/trace! "my-choice" dist/normal 0 1)

;; Integers
(dynamic/trace! 42 dist/normal 0 1)
```

### Address Uniqueness

**Each address must be unique within a single execution.** Reusing an address throws an error:

```clojure
;; ERROR: Address collision
(gen []
  (dynamic/trace! :x dist/normal 0 1)
  (dynamic/trace! :x dist/normal 0 1))  ; Throws!
```

### Addressing Patterns

#### Indexed Observations

```clojure
(def data-model
  (gen [n]
    (dotimes [i n]
      (dynamic/trace! [:obs i] dist/normal 0 1))))
```

#### Hierarchical Grouping

```clojure
(def grouped-model
  (gen []
    ;; Parameters
    (dynamic/trace! [:params :mean] dist/normal 0 10)
    (dynamic/trace! [:params :std] dist/gamma 1 1)
    ;; Observations
    (dynamic/trace! [:data :x] dist/normal 0 1)
    (dynamic/trace! [:data :y] dist/normal 0 1)))
```

## Composing Generative Functions

Generative functions can call other generative functions in three ways:

### 1. Regular Call (Untraced)

```clojure
(def inner (gen [] (dynamic/trace! :x dist/normal 0 1)))

(def outer
  (gen []
    ;; NOT traced - :x won't appear in outer's trace
    (inner)))
```

Use when you don't care about the inner choices.

### 2. trace! with Address (Hierarchical)

```clojure
(def inner (gen [] (dynamic/trace! :x dist/normal 0 1)))

(def outer
  (gen []
    ;; Inner's choices appear under :sub namespace
    ;; Accessible as [:sub :x]
    (dynamic/trace! :sub inner)))
```

The inner choices are **nested** under the given address. Use when:
- You want to organize choices hierarchically
- The inner function might be called multiple times

### 3. splice! (Flat)

```clojure
(def inner (gen [] (dynamic/trace! :x dist/normal 0 1)))

(def outer
  (gen []
    ;; Inner's choices imported directly
    ;; Accessible as :x (not [:sub :x])
    (dynamic/splice! inner)))
```

The inner choices are **imported directly** into the outer trace. Use when:
- You want a flat address space
- The inner function is called only once

### Comparison

```clojure
(def inner (gen [] (dynamic/trace! :a dist/normal 0 1)))

;; Using trace! - hierarchical
(def outer-traced
  (gen []
    (dynamic/trace! :inner inner)))

(trace/get-choices (gf/simulate outer-traced []))
;; => {:inner {:a #gen/choice 0.5}}
;; Access: (get-in choices [:inner :a])

;; Using splice! - flat
(def outer-spliced
  (gen []
    (dynamic/splice! inner)))

(trace/get-choices (gf/simulate outer-spliced []))
;; => {:a #gen/choice 0.5}
;; Access: (get choices :a)
```

### Avoiding Address Collisions with trace!

When calling a function multiple times, use trace! with different addresses:

```clojure
(def sample-point (gen [] {:x (dynamic/trace! :x dist/normal 0 1)
                           :y (dynamic/trace! :y dist/normal 0 1)}))

(def multi-point
  (gen [n]
    (vec (for [i (range n)]
           ;; Each call gets its own namespace
           (dynamic/trace! [:point i] sample-point)))))
```

## Execution Modes

Generative functions support multiple execution modes:

### Direct Call (Untraced)

```clojure
(my-model arg1 arg2)
```

- Runs the function like a normal Clojure function
- Random choices are made but NOT recorded
- Returns only the return value

### gf/simulate (Traced, Unconstrained)

```clojure
(gf/simulate my-model [arg1 arg2])
```

- Runs the function and records all choices
- Returns a complete trace
- No constraints - all choices are sampled freely

### gf/generate (Traced, Constrained)

```clojure
(gf/generate my-model [arg1 arg2] constraints)
```

- Runs the function with some choices fixed to given values
- Unconstrained choices are sampled freely
- Returns `{:trace ... :weight ...}`
- Weight is the log probability of the constrained choices

### gf/assess (Score Only)

```clojure
(gf/assess my-model [arg1 arg2] complete-choices)
```

- Computes log probability of a complete set of choices
- Does NOT create a full trace (more efficient)
- Returns `{:weight ... :retval ...}`
- All choices must be specified

### gf/propose (Sample + Score)

```clojure
(gf/propose my-model [arg1 arg2])
```

- Samples all choices and returns them with log probability
- Returns `{:choices ... :weight ... :retval ...}`
- Primarily useful for primitive distributions

## The Generative Function Interface

Generative functions implement the `IGenerativeFunction` protocol:

```clojure
(defprotocol IGenerativeFunction
  (simulate [gf args])
  (has-argument-grads [gf])
  (accepts-output-grad? [gf])
  (get-params [gf]))

(defprotocol IGenerate
  (-generate [gf args constraints]))

(defprotocol IAssess
  (-assess [gf args choices]))

(defprotocol IPropose
  (propose [gf args]))
```

### Distributions as Generative Functions

Primitive distributions are also generative functions:

```clojure
;; These are equivalent:
(dynamic/trace! :x dist/normal 0 1)
(dynamic/trace! :x (dist/normal-distribution 0 1))

;; dist/normal is a GenerativeFn that wraps normal-distribution
```

This means you can:
- Use `gf/simulate` on distributions
- Use `gf/generate` to compute probability of a value
- Mix distributions and gen functions seamlessly

## Common Patterns

### Bayesian Linear Regression

```clojure
(def linear-regression
  (gen [xs]
    (let [;; Priors on parameters
          slope (dynamic/trace! :slope dist/normal 0 2)
          intercept (dynamic/trace! :intercept dist/normal 0 5)
          noise (dynamic/trace! :noise dist/gamma 1 1)]
      ;; Likelihood
      (doseq [[i x] (map-indexed vector xs)]
        (let [y-mean (+ (* slope x) intercept)]
          (dynamic/trace! [:y i] dist/normal y-mean noise)))
      ;; Return the line function
      (fn [x] (+ (* slope x) intercept)))))
```

### Mixture Models

```clojure
(def gaussian-mixture
  (gen [n k]
    (let [;; Component parameters
          means (vec (for [j (range k)]
                       (dynamic/trace! [:mean j] dist/normal 0 10)))
          ;; Mixture weights (use Dirichlet in full implementation)
          weights (vec (repeat k (/ 1.0 k)))]
      ;; Generate data
      (dotimes [i n]
        (let [z (dynamic/trace! [:z i] dist/categorical weights)
              mu (nth means z)]
          (dynamic/trace! [:x i] dist/normal mu 1))))))
```

### Hidden Markov Models

```clojure
(def hmm
  (gen [n num-states]
    (let [;; Transition probabilities (simplified)
          trans-probs (vec (repeat num-states (/ 1.0 num-states)))
          ;; Emission parameters
          emit-means (vec (range num-states))]
      ;; Initial state
      (loop [t 0
             state (dynamic/trace! [:state 0] dist/categorical trans-probs)]
        (when (< t n)
          ;; Emit observation
          (dynamic/trace! [:obs t] dist/normal (nth emit-means state) 1)
          ;; Transition
          (when (< (inc t) n)
            (recur (inc t)
                   (dynamic/trace! [:state (inc t)] dist/categorical trans-probs))))))))
```

### Recursive Models

```clojure
(def random-tree
  (gen tree [depth max-depth]
    (if (or (>= depth max-depth)
            (dynamic/trace! [:stop depth] dist/bernoulli 0.3))
      ;; Leaf
      (dynamic/trace! [:value depth] dist/normal 0 1)
      ;; Branch
      {:left (dynamic/trace! [:left depth] tree (inc depth) max-depth)
       :right (dynamic/trace! [:right depth] tree (inc depth) max-depth)})))
```

### Conditional Structure

```clojure
(def conditional-model
  (gen [use-complex]
    (if use-complex
      ;; Complex path - more parameters
      (let [a (dynamic/trace! :a dist/normal 0 1)
            b (dynamic/trace! :b dist/normal 0 1)
            c (dynamic/trace! :c dist/normal 0 1)]
        (+ a b c))
      ;; Simple path - fewer parameters
      (dynamic/trace! :simple dist/normal 0 1))))
```

## Best Practices

### 1. Give Meaningful Addresses

```clojure
;; Good
(dynamic/trace! :slope dist/normal 0 1)
(dynamic/trace! [:observation i] dist/normal mu sigma)

;; Avoid
(dynamic/trace! :x1 dist/normal 0 1)
(dynamic/trace! 42 dist/normal 0 1)
```

### 2. Use Consistent Address Schemes

```clojure
;; Consistent scheme for parameters vs data
(dynamic/trace! [:param :slope] ...)
(dynamic/trace! [:param :intercept] ...)
(dynamic/trace! [:data i] ...)
```

### 3. Keep Models Modular

```clojure
;; Break complex models into composable pieces
(def prior-model (gen [] ...))
(def likelihood-model (gen [params data] ...))

(def full-model
  (gen [data]
    (let [params (dynamic/trace! :prior prior-model)]
      (dynamic/trace! :likelihood likelihood-model params data))))
```

### 4. Test with Simulation First

```clojure
;; Always test that your model generates sensible data
(dotimes [_ 10]
  (let [tr (gf/simulate my-model [args])]
    (println (trace/get-choices tr))))
```

### 5. Document Expected Choices

```clojure
(def my-model
  "Model for X.

  Choices:
    :slope     - slope parameter (real)
    :intercept - intercept parameter (real)
    [:y i]     - observation i (real)
  "
  (gen [xs] ...))
```

## Debugging Generative Functions

### Inspecting Traces

```clojure
(let [tr (gf/simulate my-model [args])]
  (println "Args:" (trace/get-args tr))
  (println "Return:" (trace/get-retval tr))
  (println "Choices:" (trace/get-choices tr))
  (println "Score:" (trace/get-score tr)))
```

### Finding Address Collisions

Address collisions throw exceptions with the duplicate address:

```
ExceptionInfo: Subtrace already present at address.
{:addr :x}
```

### Checking Score Computation

```clojure
;; Score should equal sum of individual log probabilities
(let [tr (gf/simulate simple-model [])
      choices (trace/get-choices tr)
      manual-score (+ (d/logpdf (dist/normal-distribution 0 1)
                                (choicemap/get-value choices :x))
                      ...)]
  (println "Trace score:" (trace/get-score tr))
  (println "Manual score:" manual-score))
```

## See Also

- [Distributions](distributions.md) - Available probability distributions
- [Traces and Choice Maps](traces-and-choicemaps.md) - Working with execution traces
- [Inference](inference.md) - Using generative functions for inference
- [API Reference](api-reference.md) - Complete API documentation
