# Gen.clj Documentation

Gen.clj is a Clojure implementation of [Gen](https://www.gen.dev/), a multi-paradigm platform for probabilistic modeling and inference. It provides a powerful framework for building probabilistic programs that can reason under uncertainty.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Quick Start](#quick-start)
5. [Documentation Index](#documentation-index)

## Overview

Gen.clj enables you to:

- **Define probabilistic models** as generative functions that make random choices
- **Perform inference** to reason backwards from observations to latent variables
- **Compose models** hierarchically using traced function calls
- **Implement custom inference algorithms** using low-level trace manipulation

### What is Probabilistic Programming?

Probabilistic programming combines programming languages with probabilistic modeling. Instead of writing deterministic programs, you write programs that make random choices. The system then helps you answer questions like:

- "Given this data, what parameters probably generated it?" (inference)
- "What data would this model generate?" (simulation)
- "How likely is this particular configuration?" (scoring)

### Gen.clj vs Gen.jl

Gen.clj is a Clojure port of the original Julia implementation (Gen.jl). Currently, Gen.clj supports:

- ✅ Dynamic DSL for generative functions
- ✅ Importance sampling inference
- ✅ Trace updates (for MCMC)
- ✅ Multiple distribution backends (Commons Math, Kixi Stats)
- ⚠️ Partial: Gradients and optimization (stubs present)
- ❌ Not yet: Static DSL, variational inference, neural network integration

## Installation

### Using deps.edn

Add Gen.clj as a Git dependency:

```clojure
{:deps
 {io.github.probcomp/gen.clj
  {:git/url "https://github.com/probcomp/Gen.clj"
   :git/sha "LATEST_SHA"}}}
```

Or as a local dependency (if you have it as a submodule):

```clojure
{:deps
 {io.github.probcomp/gen.clj {:local/root "Gen.clj"}}}
```

### Required Namespaces

```clojure
(require '[gen.dynamic :as dynamic :refer [gen]]
         '[gen.distribution.commons-math :as dist]
         '[gen.generative-function :as gf]
         '[gen.trace :as trace]
         '[gen.choicemap :as choicemap]
         '[gen.inference.importance :as importance])
```

## Core Concepts

### 1. Generative Functions

A **generative function** is a probabilistic program that can make random choices. Define one using the `gen` macro:

```clojure
(def my-model
  (gen [x]
    (let [slope (dynamic/trace! :slope dist/normal 0 1)
          noise (dynamic/trace! :noise dist/gamma 1 1)]
      (dynamic/trace! :y dist/normal (* slope x) noise))))
```

Key points:
- `gen` creates a generative function from Clojure code
- `dynamic/trace!` makes a random choice at a named **address**
- The function can be called directly (untraced) or via `gf/simulate` (traced)

### 2. Traces

A **trace** records the execution of a generative function, including:
- The arguments passed to the function
- All random choices made (stored in a **choice map**)
- The return value
- The log probability (score) of the execution

```clojure
(def tr (gf/simulate my-model [2.0]))

(trace/get-args tr)      ; => [2.0]
(trace/get-retval tr)    ; => 1.847...
(trace/get-choices tr)   ; => {:slope ..., :noise ..., :y ...}
(trace/get-score tr)     ; => -4.23...
```

### 3. Choice Maps

A **choice map** is a (possibly hierarchical) map from addresses to values:

```clojure
;; Simple choice map
{:x 1.0 :y 2.0}

;; With vector addresses
{[:obs 0] 1.0, [:obs 1] 2.0}

;; Hierarchical (from nested trace! calls)
{:outer {:inner 3.0}}
```

Choice maps are used both to read choices from traces and to specify constraints during inference.

### 4. Inference

**Inference** is reasoning backwards from observations to latent variables. The key operation is `gf/generate`, which runs a model while constraining some choices:

```clojure
;; Constrain :y to be 5.0, sample :slope and :noise
(def result (gf/generate my-model [2.0] {:y 5.0}))

(:trace result)   ; The trace with :y constrained
(:weight result)  ; Log probability of the constraint
```

## Quick Start

Here's a complete example: inferring the bias of a coin from observed flips.

```clojure
(ns coin-example
  (:require [gen.dynamic :as dynamic :refer [gen]]
            [gen.distribution.commons-math :as dist]
            [gen.generative-function :as gf]
            [gen.trace :as trace]
            [gen.choicemap :as choicemap]
            [gen.inference.importance :as importance]))

;; Model: unknown bias, observe flips
(def coin-model
  (gen [n]
    (let [bias (dynamic/trace! :bias dist/beta 1 1)]  ; Prior: uniform on [0,1]
      (dotimes [i n]
        (dynamic/trace! [:flip i] dist/bernoulli bias))
      bias)))

;; Observed data: 7 heads, 3 tails
(def observations
  {[:flip 0] true,  [:flip 1] true,  [:flip 2] false,
   [:flip 3] true,  [:flip 4] true,  [:flip 5] true,
   [:flip 6] false, [:flip 7] true,  [:flip 8] false,
   [:flip 9] true})

;; Run importance sampling
(def result (importance/resampling coin-model [10] observations 1000))

;; Get the inferred bias
(def inferred-bias
  (choicemap/get-value (trace/get-choices (:trace result)) :bias))

(println "Inferred bias:" inferred-bias)
;; Should be around 0.7 (7/10 heads)
```

## Documentation Index

- **[Distributions](distributions.md)** - Probability distributions and how to use them
- **[Generative Functions](generative-functions.md)** - Creating and composing probabilistic programs
- **[Traces and Choice Maps](traces-and-choicemaps.md)** - Understanding execution traces
- **[Inference](inference.md)** - Techniques for posterior inference
- **[API Reference](api-reference.md)** - Complete function reference

## Resources

- [Gen.clj GitHub Repository](https://github.com/probcomp/Gen.clj)
- [Original Gen (Julia) Documentation](https://www.gen.dev/docs/stable/)
- [Gen Paper (PLDI 2019)](https://dl.acm.org/doi/10.1145/3314221.3314642)
