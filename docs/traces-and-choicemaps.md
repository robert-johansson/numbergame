# Traces and Choice Maps in Gen.clj

Traces and choice maps are the data structures that capture the execution of generative functions. Understanding them is essential for performing inference and building custom algorithms.

## Table of Contents

1. [What is a Trace?](#what-is-a-trace)
2. [Working with Traces](#working-with-traces)
3. [Choice Maps](#choice-maps)
4. [Creating Choice Maps](#creating-choice-maps)
5. [Hierarchical Choice Maps](#hierarchical-choice-maps)
6. [Trace Updates](#trace-updates)
7. [Common Operations](#common-operations)

## What is a Trace?

A **trace** is a record of a single execution of a generative function. It contains:

| Component | Description | Access Function |
|-----------|-------------|-----------------|
| Arguments | Input arguments to the function | `trace/get-args` |
| Return Value | What the function returned | `trace/get-retval` |
| Choices | Map of address → sampled value | `trace/get-choices` |
| Score | Log probability of all choices | `trace/get-score` |
| Gen Function | The function that was executed | `trace/get-gen-fn` |

### Creating a Trace

Traces are created by executing generative functions:

```clojure
(require '[gen.generative-function :as gf]
         '[gen.trace :as trace])

(def my-model
  (gen [x]
    (let [slope (dynamic/trace! :slope dist/normal 0 1)]
      (dynamic/trace! :y dist/normal (* slope x) 0.5))))

;; Create a trace via simulation
(def tr (gf/simulate my-model [2.0]))
```

## Working with Traces

### Accessing Trace Components

```clojure
(def tr (gf/simulate my-model [2.0]))

;; Arguments
(trace/get-args tr)
;; => [2.0]

;; Return value
(trace/get-retval tr)
;; => 1.234... (the last traced value)

;; Choice map
(trace/get-choices tr)
;; => {:slope #gen/choice 0.567..., :y #gen/choice 1.234...}

;; Log probability (score)
(trace/get-score tr)
;; => -2.456...

;; The generative function itself
(trace/get-gen-fn tr)
;; => #gen.dynamic.DynamicDSLFunction{...}
```

### The Score

The **score** is the sum of log probabilities of all traced choices:

```clojure
(def model
  (gen []
    (dynamic/trace! :a dist/normal 0 1)    ; logp₁
    (dynamic/trace! :b dist/normal 0 1)))  ; logp₂

;; score = logp₁ + logp₂
```

For a Bernoulli with p=0.5:
```clojure
(let [tr (gf/simulate (gen [] (dynamic/trace! :x dist/bernoulli 0.5)) [])]
  (trace/get-score tr))
;; => -0.693... (which is log(0.5))
```

## Choice Maps

A **choice map** maps addresses to values. It's the "record" of random choices made during execution.

### Accessing Values

There are several ways to access choice map values:

```clojure
(def tr (gf/simulate my-model [2.0]))
(def choices (trace/get-choices tr))

;; Method 1: Using get
(get choices :slope)
;; => #gen/choice 0.567...

;; Method 2: Choice map as function
(choices :slope)
;; => #gen/choice 0.567...

;; Method 3: Keyword as function
(:slope choices)
;; => #gen/choice 0.567...

;; For vector addresses
(get choices [:y 0])
(choices [:y 0])
```

### Choice Wrappers

Values in choice maps are wrapped in `Choice` records:

```clojure
(choices :slope)
;; => #gen/choice 0.567...  (this is a Choice wrapper)

;; To get the raw value:
(require '[gen.choicemap :as choicemap])
(choicemap/get-value choices :slope)
;; => 0.567...  (raw number)
```

**Important:** Use `choicemap/get-value` when you need the actual numeric value for computations:

```clojure
;; This will fail (Choice can't be used as Number):
(+ (choices :x) 1)  ; ERROR!

;; This works:
(+ (choicemap/get-value choices :x) 1)  ; OK
```

### Checking for Values

```clojure
;; Check if address has a value
(choicemap/has-value? choices :slope)
;; => true

;; Check if address has a submap (hierarchical)
(choicemap/has-submap? choices :slope)
;; => false
```

## Creating Choice Maps

Choice maps are used as **constraints** in `gf/generate` and `trace/update`.

### From Regular Maps

Regular Clojure maps are automatically converted:

```clojure
(gf/generate my-model [2.0] {:slope 1.0})
;; {:slope 1.0} is used as constraints
```

### Using choicemap/choicemap

For explicit creation:

```clojure
(def cm (choicemap/choicemap {:slope 1.0 :intercept 2.0}))

;; With vector addresses
(def cm (choicemap/choicemap {[:y 0] 1.5
                               [:y 1] 2.3
                               [:y 2] 1.8}))
```

### Building Incrementally

```clojure
(def cm (-> (choicemap/choicemap)
            (assoc :slope 1.0)
            (assoc :intercept 2.0)))
```

### From Observations

Common pattern for building constraints from data:

```clojure
(def observations [1.5 2.3 1.8 2.1])

(def constraints
  (reduce (fn [cm [i y]]
            (assoc cm [:y i] y))
          (choicemap/choicemap)
          (map-indexed vector observations)))

;; Or more concisely:
(def constraints
  (into {} (map-indexed (fn [i y] [[:y i] y]) observations)))
```

## Hierarchical Choice Maps

When using `trace!` to call other generative functions, choice maps become hierarchical:

```clojure
(def inner (gen [] (dynamic/trace! :x dist/normal 0 1)))

(def outer
  (gen []
    (dynamic/trace! :a inner)
    (dynamic/trace! :b inner)))

(def tr (gf/simulate outer []))
(def choices (trace/get-choices tr))

;; Hierarchical structure:
;; {:a {:x #gen/choice ...}
;;  :b {:x #gen/choice ...}}
```

### Accessing Nested Values

```clojure
;; Using get-in
(get-in choices [:a :x])
;; => #gen/choice 0.567...

;; Using choicemap/get-submap
(choicemap/get-submap choices :a)
;; => {:x #gen/choice 0.567...}

;; Getting nested value directly
(choicemap/get-value (choicemap/get-submap choices :a) :x)
;; => 0.567...
```

### Creating Nested Constraints

```clojure
;; For hierarchical models, nest your constraints:
(def constraints
  {:a {:x 1.0}
   :b {:x 2.0}})

(gf/generate outer [] constraints)
```

## Trace Updates

The `trace/update` function modifies an existing trace by changing some choices:

```clojure
(trace/update trace new-constraints)
;; Returns: {:trace new-trace
;;           :weight log-weight-change
;;           :discard old-choices
;;           :change diff-info}
```

### Basic Update

```clojure
(def model
  (gen []
    (let [x (dynamic/trace! :x dist/normal 0 1)]
      (dynamic/trace! :y dist/normal x 1))))

(def initial-trace (gf/simulate model []))
(def initial-x (choicemap/get-value (trace/get-choices initial-trace) :x))
;; Say initial-x is 0.5

;; Update :x to 2.0
(def update-result (trace/update initial-trace {:x 2.0}))

(def new-trace (:trace update-result))
(choicemap/get-value (trace/get-choices new-trace) :x)
;; => 2.0

;; :y is re-sampled given the new :x
```

### Understanding the Weight

The **weight** is the log ratio of probabilities:

```
weight = log(p(new-trace) / p(old-trace))
       = new-score - old-score
```

This is used for acceptance probability in MCMC:

```clojure
(def accept-prob (min 1.0 (Math/exp (:weight update-result))))
```

### The Discard

The **discard** contains the old values that were replaced:

```clojure
(:discard update-result)
;; => {:x #gen/choice 0.5}  (the old value)
```

The discard is useful for:
1. **Reversibility**: You can reverse an update using the discard
2. **Debugging**: See what changed

### Reversible Updates

```clojure
;; Forward update
(def forward (trace/update initial-trace {:x 2.0}))
(def forward-trace (:trace forward))
(def forward-discard (:discard forward))

;; Reverse update (restore original)
(def reverse (trace/update forward-trace forward-discard))
(def restored-trace (:trace reverse))

;; restored-trace should have same :x as initial-trace
```

### Structural Updates

Updates handle structural changes automatically:

```clojure
(def conditional-model
  (gen [threshold]
    (let [x (dynamic/trace! :x dist/normal 0 1)]
      (when (> x threshold)
        (dynamic/trace! :extra dist/normal 0 1)))))

;; Initial: x = -1 (below threshold, no :extra)
(def initial (:trace (gf/generate conditional-model [0] {:x -1})))

;; Update: x = 1 (above threshold, :extra appears)
(def updated (trace/update initial {:x 1}))

(trace/get-choices (:trace updated))
;; => {:x #gen/choice 1, :extra #gen/choice 0.3...}
```

## Common Operations

### Converting Choice Map to Regular Map

```clojure
(choicemap/->map choices)
;; Returns a regular Clojure map with raw values
```

### Getting All Shallow Addresses

```clojure
(choicemap/get-values-shallow choices)
;; => {:slope 0.567, :intercept 1.234}

(choicemap/get-submaps-shallow choices)
;; => {:a {...}, :b {...}}
```

### Checking Emptiness

```clojure
(choicemap/empty? (choicemap/choicemap))
;; => true

(choicemap/empty? choices)
;; => false
```

### Merging Choice Maps

```clojure
(choicemap/merge cm1 cm2)
;; Combines two choice maps
```

### Iterating Over Choices

```clojure
;; Choice maps are seqable
(doseq [[addr choice] choices]
  (println addr "->" choice))

;; Get just addresses
(keys choices)

;; Get just values (as Choice wrappers)
(vals choices)
```

## Practical Examples

### Building Observation Constraints

```clojure
(defn make-observation-constraints [ys]
  (into {} (map-indexed (fn [i y] [[:y i] y]) ys)))

(def observations [1.5 2.3 1.8 2.1 2.0])
(def constraints (make-observation-constraints observations))
;; => {[:y 0] 1.5, [:y 1] 2.3, ...}
```

### Extracting Inferred Parameters

```clojure
(defn extract-params [trace param-names]
  (let [choices (trace/get-choices trace)]
    (into {} (map (fn [name]
                    [name (choicemap/get-value choices name)])
                  param-names))))

(extract-params tr [:slope :intercept :noise])
;; => {:slope 1.23, :intercept 0.45, :noise 0.67}
```

### Copying Parameters Between Traces

```clojure
(defn copy-params [from-trace param-names]
  (let [from-choices (trace/get-choices from-trace)]
    (into {} (map (fn [name]
                    [name (choicemap/get-value from-choices name)])
                  param-names))))

;; Use to constrain new generation
(def old-params (copy-params old-trace [:slope :intercept]))
(def new-result (gf/generate model [new-xs] old-params))
```

### Simple MH Step

```clojure
(defn mh-step [current-trace proposal-addr proposal-dist]
  (let [current-val (choicemap/get-value
                      (trace/get-choices current-trace)
                      proposal-addr)
        proposed-val (proposal-dist current-val)
        update-result (trace/update current-trace
                                     {proposal-addr proposed-val})
        accept-prob (min 1.0 (Math/exp (:weight update-result)))]
    (if (< (rand) accept-prob)
      (:trace update-result)
      current-trace)))
```

## Summary Table

| Operation | Function | Returns |
|-----------|----------|---------|
| Get arguments | `trace/get-args` | Vector of args |
| Get return value | `trace/get-retval` | Return value |
| Get choices | `trace/get-choices` | Choice map |
| Get score | `trace/get-score` | Log probability |
| Get choice value | `choicemap/get-value` | Raw value |
| Get nested map | `choicemap/get-submap` | Choice map |
| Check value exists | `choicemap/has-value?` | Boolean |
| Check submap exists | `choicemap/has-submap?` | Boolean |
| Update trace | `trace/update` | Map with :trace, :weight, :discard |
| Create choice map | `choicemap/choicemap` | Choice map |
| Convert to map | `choicemap/->map` | Regular Clojure map |

## See Also

- [Generative Functions](generative-functions.md) - Creating models
- [Inference](inference.md) - Using traces for inference
- [API Reference](api-reference.md) - Complete API documentation
