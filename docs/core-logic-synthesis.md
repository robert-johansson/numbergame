# Program Synthesis with core.logic

This document describes how to implement backward program synthesis using
Clojure's core.logic library. Instead of pre-computing hypothesis sets and
filtering, we define membership **intensionally** (as rules) and let core.logic
find programs that satisfy the constraints.

## The Key Insight

```
EXTENSIONAL (current approach):
  Pre-compute: :pow2 → #{1 2 4 8 16 32 64}
  Query: Does this set contain our examples?

INTENSIONAL (core.logic approach):
  Define: x ∈ pow2 iff x = 2^n for some n
  Query: Find programs where all examples satisfy the membership rule
```

No pre-computation. core.logic searches the space guided by constraints.

## Setup

Add to `deps.edn`:

```clojure
{:deps
 {org.clojure/core.logic {:mvn/version "1.0.1"}}}
```

## Basic core.logic Concepts

```clojure
(require '[clojure.core.logic :as l]
         '[clojure.core.logic.fd :as fd])  ; Finite domains

;; l/run* - find all solutions
;; l/run n - find n solutions
;; l/== - unification (bidirectional equality)
;; l/conde - disjunction (OR)
;; l/fresh - introduce logic variables
;; fd/in - constrain to finite domain
;; fd/+ fd/* fd/** - arithmetic over finite domains
```

## The Grammar as Relations

### Defining Membership Intensionally

```clojure
(ns numbergame.logic
  (:require [clojure.core.logic :as l]
            [clojure.core.logic.fd :as fd]))

;; Helper: x is a power of base k
(defn power-ofo
  "x = k^n for some n in [0, max-exp]"
  [k x max-exp]
  (l/fresh [n]
    (fd/in n (fd/interval 0 max-exp))
    (fd/** k n x)))

;; Helper: x is a multiple of k
(defn mult-ofo
  "x = k*m for some m in [1, max-mult]"
  [k x max-mult]
  (l/fresh [m]
    (fd/in m (fd/interval 1 max-mult))
    (fd/* k m x)))

;; Helper: x is in range [lo, hi]
(defn range-ofo
  "lo <= x <= hi"
  [lo hi x]
  (l/all
    (fd/>= x lo)
    (fd/<= x hi)))

;; Helper: x is prime (simplified - enumerate small primes)
(def primes #{2 3 5 7 11 13 17 19 23 29 31 37 41 43 47
              53 59 61 67 71 73 79 83 89 97})

(defn prime-o
  "x is a prime number in 1-100"
  [x]
  (l/membero x (vec primes)))

;; Helper: x is a perfect square
(defn square-o
  "x = n^2 for some n"
  [x]
  (l/fresh [n]
    (fd/in n (fd/interval 1 10))
    (fd/* n n x)))
```

### The Main Membership Relation

```clojure
(defn in-concept
  "Relation: x is a member of the concept defined by prog.

   prog is a logic variable representing the program structure.
   x is a logic variable representing a number.

   This relation can be run 'backward' - given x, find prog."
  [prog x]
  (l/conde
    ;; Power of k: x = k^n
    [(l/fresh [k]
       (l/== prog {:type :power-of :base k})
       (fd/in k (fd/interval 2 10))
       (fd/in x (fd/interval 1 100))
       (power-ofo k x 7))]

    ;; Multiple of k: x = k*m
    [(l/fresh [k]
       (l/== prog {:type :mult :k k})
       (fd/in k (fd/interval 2 20))
       (fd/in x (fd/interval 1 100))
       (mult-ofo k x 50))]

    ;; Range: lo <= x <= hi
    [(l/fresh [lo hi]
       (l/== prog {:type :range :lo lo :hi hi})
       (fd/in lo (fd/interval 1 100))
       (fd/in hi (fd/interval 1 100))
       (fd/<= lo hi)
       (range-ofo lo hi x))]

    ;; Prime
    [(l/== prog {:type :prime})
     (prime-o x)]

    ;; Square
    [(l/== prog {:type :square})
     (square-o x)]

    ;; Even: x = 2*m
    [(l/== prog {:type :even})
     (l/fresh [m]
       (fd/in m (fd/interval 1 50))
       (fd/* 2 m x))]

    ;; Odd: x = 2*m + 1
    [(l/== prog {:type :odd})
     (l/fresh [m]
       (fd/in m (fd/interval 0 49))
       (fd/+ (fd/* 2 m) 1 x))]

    ;; Union: x in left OR x in right
    [(l/fresh [left right]
       (l/== prog {:type :union :left left :right right})
       (l/conde
         [(in-concept left x)]
         [(in-concept right x)]))]

    ;; Intersect: x in left AND x in right
    [(l/fresh [left right]
       (l/== prog {:type :intersect :left left :right right})
       (in-concept left x)
       (in-concept right x)])))
```

## Finding Programs from Examples

```clojure
(defn find-programs
  "Find all programs (up to n) whose concept contains all examples."
  [examples n]
  (l/run n [prog]
    (l/everyg (fn [ex] (in-concept prog ex)) examples)))

;; Usage:
(find-programs [16 8 2 64] 10)
;; => ({:type :power-of :base 2}
;;     {:type :even}
;;     {:type :mult :k 2}
;;     ...)
```

## Detailed Example

```clojure
;; Query: What programs contain all of [16, 8, 2, 64]?

(l/run 5 [prog]
  (in-concept prog 16)
  (in-concept prog 8)
  (in-concept prog 2)
  (in-concept prog 64))

;; core.logic reasoning:
;;
;; For (in-concept prog 16):
;;   - {:type :power-of :base 2} works (16 = 2^4)
;;   - {:type :power-of :base 4} works (16 = 4^2)
;;   - {:type :mult :k 2} works (16 = 2*8)
;;   - {:type :mult :k 4} works (16 = 4*4)
;;   - {:type :mult :k 8} works (16 = 8*2)
;;   - {:type :even} works
;;   - {:type :range :lo ?lo :hi ?hi} where ?lo <= 16 <= ?hi
;;   - etc.
;;
;; For (in-concept prog 8):
;;   - {:type :power-of :base 2} works (8 = 2^3)
;;   - {:type :mult :k 2} works (8 = 2*4)
;;   - {:type :mult :k 4} works (8 = 4*2)
;;   - {:type :even} works
;;   - {:type :range :lo ?lo :hi ?hi} where ?lo <= 8 <= ?hi
;;   - etc.
;;
;; INTERSECTION of constraints:
;;   - {:type :power-of :base 2} survives (all are powers of 2)
;;   - {:type :even} survives (all are even)
;;   - {:type :mult :k 2} survives (all are multiples of 2)
;;   - {:type :power-of :base 4} FAILS (2 is not a power of 4)
;;   - {:type :mult :k 4} FAILS (2 is not a multiple of 4)
;;   - {:type :range :lo 2 :hi 64} survives
;;   - etc.
```

## Ranking Programs by Size Principle

core.logic finds consistent programs, but we need to rank by likelihood:

```clojure
(defn concept-size
  "Compute the size of a concept (how many numbers it contains)."
  [prog]
  (count
   (l/run* [x]
     (fd/in x (fd/interval 1 100))
     (in-concept prog x))))

(defn rank-programs
  "Rank programs by size principle: smaller = more probable."
  [programs examples]
  (let [n (count examples)]
    (->> programs
         (map (fn [prog]
                (let [size (concept-size prog)
                      log-like (- (* n (Math/log size)))]
                  {:program prog
                   :size size
                   :log-likelihood log-like})))
         (sort-by :log-likelihood >))))
```

## Full Synthesis Pipeline

```clojure
(defn synthesize
  "Complete program synthesis: find and rank programs."
  [examples max-programs]
  (let [programs (find-programs examples max-programs)
        ranked (rank-programs programs examples)]
    {:examples examples
     :n-found (count programs)
     :top-programs (take 10 ranked)}))

;; Usage:
(synthesize [16 8 2 64] 100)
;; => {:examples [16 8 2 64]
;;     :n-found 47
;;     :top-programs
;;     [{:program {:type :power-of :base 2} :size 7 :log-likelihood ...}
;;      {:program {:type :intersect
;;                 :left {:type :power-of :base 2}
;;                 :right {:type :even}} :size 6 ...}
;;      ...]}
```

## Handling Compositional Programs

For union and intersect, the search can explode. Strategies:

### 1. Limit Depth

```clojure
(defn in-concept-bounded
  "in-concept with bounded depth to prevent infinite search."
  [prog x depth]
  (if (zero? depth)
    ;; Base case: only primitives
    (l/conde
      [(l/fresh [k] (l/== prog {:type :power-of :base k}) ...)]
      [(l/fresh [k] (l/== prog {:type :mult :k k}) ...)]
      ...)
    ;; Can also include compositions
    (l/conde
      ;; ... primitives ...
      [(l/fresh [left right]
         (l/== prog {:type :union :left left :right right})
         (l/conde
           [(in-concept-bounded left x (dec depth))]
           [(in-concept-bounded right x (dec depth))]))]
      ...)))
```

### 2. Iterative Deepening

```clojure
(defn find-programs-iterative
  "Find programs, trying increasing depths."
  [examples max-depth max-programs]
  (loop [depth 0
         all-programs []]
    (if (or (> depth max-depth)
            (>= (count all-programs) max-programs))
      all-programs
      (let [new-programs (find-programs-at-depth examples depth)]
        (recur (inc depth)
               (into all-programs new-programs))))))
```

## Comparison with Current Approaches

| Aspect | Enumeration | Sampling | core.logic |
|--------|-------------|----------|------------|
| Pre-computation | All hypotheses | None | None |
| Search | Filter pre-computed | Random + reject | Constraint propagation |
| Completeness | Exact | Approximate | Exact (within depth) |
| Efficiency | O(hypotheses) | O(samples) | Depends on constraints |
| Compositionality | Limited | Natural | Natural |
| Parameter discovery | No | No | Yes (e.g., finds k=2) |

## Advantages of core.logic Approach

1. **No pre-computation**: Define rules, not sets
2. **Parameter discovery**: Finds that k=2 works, not just "pow2"
3. **Natural compositionality**: Union/intersect are just relations
4. **Bidirectional**: Same relation for membership test and synthesis
5. **Constraint propagation**: Prunes search space intelligently
6. **Extensible**: Add new concept types easily

## Limitations

1. **Performance**: Can be slower than pre-computed lookup
2. **Depth limits**: Need to bound recursive concepts
3. **Finite domains**: fd/ only works with integers
4. **Debugging**: Logic programs can be hard to debug

## Implementation Checklist

- [ ] Add core.logic to deps.edn
- [ ] Implement primitive relations (power-ofo, mult-ofo, etc.)
- [ ] Implement in-concept main relation
- [ ] Implement find-programs query
- [ ] Add size computation for ranking
- [ ] Handle bounded depth for compositions
- [ ] Add tests for various example sets
- [ ] Benchmark against current approaches

## Example Session

```clojure
(require '[numbergame.logic :as logic])

;; Powers of 2
(logic/synthesize [16 8 2 64] 50)
;; Top: {:type :power-of :base 2}

;; Interval
(logic/synthesize [16 17 18 19] 50)
;; Top: {:type :range :lo 16 :hi 19}

;; Primes
(logic/synthesize [7 11 13 17] 50)
;; Top: {:type :prime}

;; Multiples of 7
(logic/synthesize [7 14 21 28] 50)
;; Top: {:type :mult :k 7}  ; Discovered k=7!

;; Union (if we allow compositions)
(logic/synthesize [2 4 8 3 9 27] 50)
;; Top: {:type :union
;;       :left {:type :power-of :base 2}
;;       :right {:type :power-of :base 3}}
```

## References

- [core.logic Tutorial](https://github.com/clojure/core.logic/wiki/A-Core.logic-Primer)
- [Finite Domain Constraints](https://github.com/clojure/core.logic/wiki/CLP(FD))
- [miniKanren](http://minikanren.org/)
