# Gen.clj API Reference

Quick reference for all Gen.clj functions and macros.

## Table of Contents

1. [Namespaces](#namespaces)
2. [gen.dynamic](#gendynamic)
3. [gen.generative-function](#gengenerative-function)
4. [gen.trace](#gentrace)
5. [gen.choicemap](#genchoicemap)
6. [gen.distribution](#gendistribution)
7. [gen.distribution.commons-math](#gendistributioncommons-math)
8. [gen.inference.importance](#geninferenceimportance)

---

## Namespaces

```clojure
;; Core namespaces
(require '[gen.dynamic :as dynamic :refer [gen]])
(require '[gen.generative-function :as gf])
(require '[gen.trace :as trace])
(require '[gen.choicemap :as choicemap])

;; Distributions
(require '[gen.distribution :as d])
(require '[gen.distribution.commons-math :as dist])
;; or
(require '[gen.distribution.kixi :as dist])

;; Inference
(require '[gen.inference.importance :as importance])
```

---

## gen.dynamic

### gen

```clojure
(gen [args] body)
(gen name [args] body)
```

Macro that creates a generative function from Clojure code.

**Parameters:**
- `name` - Optional. Symbol for recursive calls.
- `args` - Vector of argument symbols.
- `body` - Clojure expressions, may contain `trace!` and `splice!` calls.

**Returns:** A `DynamicDSLFunction` implementing `IGenerativeFunction`.

**Example:**
```clojure
(def my-model
  (gen [x]
    (let [slope (dynamic/trace! :slope dist/normal 0 1)]
      (* slope x))))

;; With name for recursion
(def recursive-model
  (gen recur-fn [depth]
    (if (zero? depth)
      (dynamic/trace! :leaf dist/normal 0 1)
      {:left (dynamic/trace! :left recur-fn (dec depth))
       :right (dynamic/trace! :right recur-fn (dec depth))})))
```

---

### trace!

```clojure
(dynamic/trace! address gen-fn & args)
```

Makes a traced random choice inside a `gen` body.

**Parameters:**
- `address` - Any Clojure value (keyword, vector, string, etc.). Must be unique per execution.
- `gen-fn` - A generative function (distribution or `gen` function).
- `args` - Arguments to pass to `gen-fn`.

**Returns:** The sampled/returned value from `gen-fn`.

**Example:**
```clojure
(gen []
  ;; Trace a distribution
  (dynamic/trace! :x dist/normal 0 1)

  ;; Trace with vector address
  (dynamic/trace! [:obs 0] dist/normal 0 1)

  ;; Trace another gen function
  (dynamic/trace! :sub other-gen-fn arg1 arg2))
```

---

### splice!

```clojure
(dynamic/splice! gen-fn & args)
```

Calls a generative function and imports its choices directly (flat addressing).

**Parameters:**
- `gen-fn` - A generative function.
- `args` - Arguments to pass to `gen-fn`.

**Returns:** The return value of `gen-fn`.

**Example:**
```clojure
(def inner (gen [] (dynamic/trace! :x dist/normal 0 1)))

(def outer
  (gen []
    ;; :x appears directly in outer's trace (not nested)
    (dynamic/splice! inner)))
```

---

### untraced

```clojure
(dynamic/untraced & body)
```

Macro that executes body without tracing.

**Example:**
```clojure
(gen []
  ;; This IS traced
  (dynamic/trace! :x dist/normal 0 1)

  ;; This is NOT traced
  (dynamic/untraced
    (dist/normal 0 1)))  ; Choice not recorded
```

---

### active-trace

```clojure
(dynamic/active-trace)
```

Returns the currently active tracing function. For advanced use.

---

## gen.generative-function

### simulate

```clojure
(gf/simulate gen-fn args)
```

Executes a generative function and returns a complete trace.

**Parameters:**
- `gen-fn` - A generative function.
- `args` - Vector of arguments.

**Returns:** A trace.

**Example:**
```clojure
(def tr (gf/simulate my-model [2.0]))
(trace/get-choices tr)  ; => {:slope ..., :y ...}
```

---

### generate

```clojure
(gf/generate gen-fn args constraints)
```

Executes with some choices constrained to given values.

**Parameters:**
- `gen-fn` - A generative function.
- `args` - Vector of arguments.
- `constraints` - Choice map of address → value.

**Returns:** Map with keys:
- `:trace` - The generated trace.
- `:weight` - Log probability of constrained choices.

**Example:**
```clojure
(def result (gf/generate my-model [2.0] {:y 5.0}))
(:trace result)   ; Trace with :y = 5.0
(:weight result)  ; log P(y=5.0 | slope)
```

---

### assess

```clojure
(gf/assess gen-fn args choices)
```

Computes log probability of a complete set of choices.

**Parameters:**
- `gen-fn` - A generative function.
- `args` - Vector of arguments.
- `choices` - Complete choice map (all addresses must be specified).

**Returns:** Map with keys:
- `:weight` - Log probability of choices.
- `:retval` - Return value.

**Example:**
```clojure
(def result (gf/assess my-model [2.0] {:slope 1.0 :y 3.0}))
(:weight result)  ; Log probability
```

---

### propose

```clojure
(gf/propose gen-fn args)
```

Samples choices and returns them with log probability.

**Parameters:**
- `gen-fn` - A generative function.
- `args` - Vector of arguments.

**Returns:** Map with keys:
- `:choices` - Sampled choice map.
- `:weight` - Log probability.
- `:retval` - Return value.

**Note:** Works well for primitive distributions; dynamic gen functions may have limitations.

---

## gen.trace

### get-args

```clojure
(trace/get-args trace)
```

Returns the arguments the function was called with.

**Returns:** Vector of arguments.

---

### get-retval

```clojure
(trace/get-retval trace)
```

Returns the return value of the function.

---

### get-choices

```clojure
(trace/get-choices trace)
```

Returns the choice map of all traced choices.

**Returns:** Choice map (address → Choice wrapper).

---

### get-score

```clojure
(trace/get-score trace)
```

Returns the log probability of all choices.

**Returns:** Double (log probability).

---

### get-gen-fn

```clojure
(trace/get-gen-fn trace)
```

Returns the generative function that produced this trace.

---

### update

```clojure
(trace/update trace constraints)
```

Modifies a trace by changing some choices.

**Parameters:**
- `trace` - Existing trace.
- `constraints` - Choice map of new values.

**Returns:** Map with keys:
- `:trace` - New trace with updated choices.
- `:weight` - Log ratio of new/old probability.
- `:discard` - Choice map of old values that were replaced.
- `:change` - Diff information.

**Example:**
```clojure
(def result (trace/update tr {:slope 2.0}))
(:trace result)    ; New trace with slope=2.0
(:weight result)   ; log(P(new)/P(old))
(:discard result)  ; {:slope <old-value>}
```

---

## gen.choicemap

### choicemap

```clojure
(choicemap/choicemap)
(choicemap/choicemap m)
```

Creates a choice map.

**Parameters:**
- `m` - Optional. Map of address → value.

**Returns:** A choice map.

**Example:**
```clojure
(choicemap/choicemap {:x 1.0 [:y 0] 2.0})
```

---

### get-value

```clojure
(choicemap/get-value cm address)
```

Gets the raw value at an address (unwraps Choice wrapper).

**Returns:** The underlying value.

**Example:**
```clojure
(def choices (trace/get-choices tr))
(choicemap/get-value choices :slope)  ; => 1.234 (not #gen/choice 1.234)
```

---

### get-submap

```clojure
(choicemap/get-submap cm address)
```

Gets the nested choice map at an address.

**Returns:** Choice map or empty choice map.

---

### has-value?

```clojure
(choicemap/has-value? cm address)
```

Checks if address has a value (not a submap).

**Returns:** Boolean.

---

### has-submap?

```clojure
(choicemap/has-submap? cm address)
```

Checks if address has a nested choice map.

**Returns:** Boolean.

---

### empty?

```clojure
(choicemap/empty? cm)
```

Checks if choice map has no entries.

**Returns:** Boolean.

---

### get-values-shallow

```clojure
(choicemap/get-values-shallow cm)
```

Gets all values at the top level (not nested).

**Returns:** Map of address → raw value.

---

### get-submaps-shallow

```clojure
(choicemap/get-submaps-shallow cm)
```

Gets all submaps at the top level.

**Returns:** Map of address → choice map.

---

### ->map

```clojure
(choicemap/->map cm)
```

Converts choice map to regular Clojure map with raw values.

**Returns:** Regular map.

---

### merge

```clojure
(choicemap/merge cm1 cm2)
```

Merges two choice maps.

**Returns:** Combined choice map.

---

## gen.distribution

### Sample protocol

```clojure
(d/sample distribution)
```

Draws a sample from the distribution.

---

### LogPDF protocol

```clojure
(d/logpdf distribution value)
```

Computes log probability density/mass at value.

---

## gen.distribution.commons-math

All distributions can be used in two ways:
1. As generative functions: `(dist/normal 0 1)` → sample
2. As distribution objects: `(dist/normal-distribution 0 1)` → object for logpdf

### Discrete Distributions

| Function | Signature | Description |
|----------|-----------|-------------|
| `bernoulli` | `(bernoulli p)` | True/false with P(true)=p |
| `binomial` | `(binomial n p)` | Successes in n trials |
| `uniform-discrete` | `(uniform-discrete lo hi)` | Integer in [lo, hi] |
| `categorical` | `(categorical probs)` | Index/key from weights |

### Continuous Distributions

| Function | Signature | Description |
|----------|-----------|-------------|
| `normal` | `(normal mu sigma)` | Gaussian |
| `uniform` | `(uniform a b)` | Uniform on [a, b] |
| `beta` | `(beta alpha beta)` | Beta on (0, 1) |
| `gamma` | `(gamma shape scale)` | Gamma on (0, ∞) |
| `exponential` | `(exponential rate)` | Exponential |
| `student-t` | `(student-t nu)` | Student's t |
| `cauchy` | `(cauchy loc scale)` | Cauchy |

### Distribution Constructors

For each `dist/X`, there's also `dist/X-distribution`:

```clojure
(dist/normal-distribution 0 1)      ; Returns distribution object
(d/logpdf (dist/normal-distribution 0 1) 0.5)  ; Compute logpdf
```

---

## gen.inference.importance

### resampling

```clojure
(importance/resampling model args observations n-particles)
```

Importance sampling with resampling.

**Parameters:**
- `model` - Generative function.
- `args` - Vector of arguments.
- `observations` - Choice map of observed values.
- `n-particles` - Number of particles.

**Returns:** Map with keys:
- `:trace` - Resampled trace.
- `:weight` - Log marginal likelihood estimate.

**Example:**
```clojure
(def result (importance/resampling
              my-model [xs]
              observations
              1000))

(trace/get-choices (:trace result))  ; Inferred values
(:weight result)                      ; Log marginal likelihood
```

---

## Quick Reference Tables

### Trace Operations

| Operation | Code |
|-----------|------|
| Create trace | `(gf/simulate model args)` |
| Constrained trace | `(gf/generate model args constraints)` |
| Get arguments | `(trace/get-args tr)` |
| Get return value | `(trace/get-retval tr)` |
| Get choices | `(trace/get-choices tr)` |
| Get score | `(trace/get-score tr)` |
| Update trace | `(trace/update tr new-constraints)` |

### Choice Map Operations

| Operation | Code |
|-----------|------|
| Create | `(choicemap/choicemap {:x 1})` |
| Get raw value | `(choicemap/get-value cm :x)` |
| Get nested map | `(choicemap/get-submap cm :x)` |
| Check value | `(choicemap/has-value? cm :x)` |
| Check submap | `(choicemap/has-submap? cm :x)` |
| To regular map | `(choicemap/->map cm)` |

### Common Patterns

```clojure
;; Infer and extract parameter
(let [result (importance/resampling model args obs 1000)
      choices (trace/get-choices (:trace result))]
  (choicemap/get-value choices :param))

;; Build observation constraints
(into {} (map-indexed (fn [i y] [[:y i] y]) ys))

;; MH accept/reject
(let [result (trace/update tr {:x new-x})
      accept? (< (Math/log (rand)) (min 0 (:weight result)))]
  (if accept? (:trace result) tr))
```

---

## See Also

- [Gen.clj Overview](gen-clj.md)
- [Distributions](distributions.md)
- [Generative Functions](generative-functions.md)
- [Traces and Choice Maps](traces-and-choicemaps.md)
- [Inference](inference.md)
