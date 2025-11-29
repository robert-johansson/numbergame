# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run Gen.clj learning examples
clj -M -m ex01-distributions       # Probability distributions
clj -M -m ex02-generative-functions # Generative functions and traces
clj -M -m ex03-choicemaps          # Choice maps and addressing
clj -M -m ex04-inference           # Importance sampling inference
clj -M -m ex05-trace-updates       # Trace updates and MH

# Run main example
clj -M -m numbergame.example

# Start REPL
clj

# Run tests (when added)
clj -X:test
```

## Project Structure

- `Gen.clj/` - Gen.clj probabilistic programming library (git submodule)
- `src/numbergame/` - Main source code
- `examples/` - Gen.clj learning examples (ex01-ex05)
- `deps.edn` - Clojure deps configuration (Gen.clj included as local dependency)

## Project Overview

This is a Clojure project for building probabilistic and Bayesian models of cognition. The primary example is Tenenbaum's "number game" for concept learning, but the architecture is designed to support other learning, generalization, and inference tasks.

## Architecture

The project has three layers:

1. **Core Bayesian Layer** - Pure Clojure logic for hypotheses, belief updates, and predictions. Everything is EDN data (maps, vectors) that can be inspected, logged, and serialized. No hidden mutable state in modeling code.

2. **Probabilistic Programming Layer** - Built on Gen.clj for complex models requiring approximate inference (importance sampling, resampling). Used when closed-form Bayesian updates are insufficient.

3. **Interactive Layer** - Notebooks and REPL-driven exploration for visualizing belief dynamics.

## Namespace Organization (planned)

- **Hypothesis space namespace** - Defines candidate concepts as data (identifier, member set, prior weight). For the number game: rule-based hypotheses ("multiples of 10", "powers of 2") and interval-based hypotheses ("between 16 and 23").

- **Inference namespace** - `update-posterior` function that takes prior + new example → posterior using size principle (consistent examples contribute 1/|h| to likelihood).

- **Generalization namespace** - Computes probability that a probe belongs to the concept by averaging over hypotheses weighted by posterior.

- **Gen.clj namespaces** - Generative functions that sample hypotheses from priors, then sample examples/responses. Enables hierarchical priors, noise, graded membership.

## Gen.clj Usage Patterns

```clojure
(require '[gen.dynamic :as dynamic :refer [gen]]
         '[gen.distribution.commons-math :as dist]
         '[gen.generative-function :as gf]
         '[gen.trace :as trace])

;; Define generative function with traced random choices
(def my-model
  (gen [args]
    (let [x (dynamic/trace! :x dist/normal 0.0 1.0)]
      x)))

;; Simulate (unconstrained)
(gf/simulate my-model [args])

;; Generate with constraints (for inference)
(gf/generate my-model [args] {:x 0.5})

;; Access trace data
(trace/get-choices tr)   ; choice map
(trace/get-retval tr)    ; return value
(trace/get-score tr)     ; log probability
```

## Design Principles

- State passed explicitly as arguments and returned as values
- Beliefs (posteriors, particles, parameters) are always Clojure maps/vectors
- Mutability for live experimental loops handled outside core logic via atoms
- Tests verify: posterior normalization, sensible generalization curves, explicit handling of inconsistent observations
