# Future Project: Meta-Learning / DreamCoder-like Grammar Induction

This document outlines a future project to implement grammar learning from experience,
inspired by MIT's DreamCoder system. The goal is to start with a minimal grammar and
**learn** the grammar itself from many examples, then use the learned grammar for fast
inference.

## The Core Idea

```
Current approach:
  Human defines grammar → core.logic finds programs

Future approach:
  System LEARNS grammar from experience → core.logic finds programs
```

## The Hierarchy of Learning

```
Level 3: META-GRAMMAR    ← What kinds of grammar rules are possible?
            ↓
Level 2: GRAMMAR         ← What kinds of concepts are possible?
            ↓               (LEARNED from experience)
Level 1: PROGRAMS        ← What concept explains this data?
            ↓               (inferred by core.logic)
Level 0: DATA            ← [16, 8, 2, 64]
```

## The DreamCoder Loop

DreamCoder (Ellis et al., 2020) alternates between two phases:

### Wake Phase: Solve Tasks
- Given current grammar, try to synthesize programs for tasks
- Use enumeration, neural guidance, or MCMC
- Record successful programs

### Dream/Sleep Phase: Learn Abstractions
- Analyze successful programs from wake phase
- Find common patterns/subprograms
- Abstract them into new grammar rules
- Add to grammar library

### Repeat
- With richer grammar, can solve harder tasks
- Can express solutions more concisely
- Virtuous cycle of learning

## Example: Learning the "Power" Abstraction

```
Iteration 1 (minimal grammar: just +, *, constants):

  Task: generate [1, 2, 4, 8]
  Solution found: [1, (* 1 2), (* (* 1 2) 2), (* (* (* 1 2) 2) 2)]

  Task: generate [1, 3, 9, 27]
  Solution found: [1, (* 1 3), (* (* 1 3) 3), (* (* (* 1 3) 3) 3)]

DREAM phase analyzes solutions:
  Pattern detected: repeated multiplication by same constant

  Abstracts into:
    (power-of k n) = k^n

  ADDS TO GRAMMAR.

Iteration 2 (grammar now includes power-of):

  Task: generate [1, 2, 4, 8, 16, 32]
  Solution found: (map (partial power-of 2) [0 1 2 3 4 5])

  Much more concise!
```

## Architecture for Clojure Implementation

### 1. Grammar Representation

```clojure
(def grammar
  {:primitives
   [{:name :const
     :arity 1
     :type [:int :-> :int]
     :impl (fn [c] (constantly c))}
    {:name :+
     :arity 2
     :type [:int :int :-> :int]
     :impl +}
    {:name :*
     :arity 2
     :type [:int :int :-> :int]
     :impl *}]

   :learned-abstractions
   []  ; Start empty, grow through learning

   :combinators
   [{:name :compose
     :type [[:a :-> :b] [:b :-> :c] :-> [:a :-> :c]]}
    {:name :map
     :type [[:a :-> :b] [:list :a] :-> [:list :b]]}]})
```

### 2. Program Synthesis (Wake Phase)

```clojure
(defn synthesize
  "Find a program that produces the target output."
  [grammar target-output max-depth]
  (let [candidates (enumerate-programs grammar max-depth)]
    (first (filter #(= (eval-program %) target-output) candidates))))
```

### 3. Pattern Detection (Dream Phase)

```clojure
(defn find-common-patterns
  "Find repeated subprograms across successful solutions."
  [programs min-frequency]
  (let [all-subprograms (mapcat extract-subprograms programs)
        frequencies (frequencies all-subprograms)]
    (->> frequencies
         (filter (fn [[_ freq]] (>= freq min-frequency)))
         (map first))))

(defn abstract-pattern
  "Turn a concrete pattern into a parameterized abstraction."
  [pattern]
  ;; Find constants that vary, turn them into parameters
  ;; E.g., (* (* x 2) 2) and (* (* x 3) 3)
  ;;       → (power-of ?k x)
  ...)
```

### 4. Grammar Extension

```clojure
(defn extend-grammar
  "Add newly learned abstractions to the grammar."
  [grammar new-abstractions]
  (update grammar :learned-abstractions
          into new-abstractions))
```

### 5. The Main Loop

```clojure
(defn dreamcoder-loop
  "Main learning loop."
  [initial-grammar tasks n-iterations]
  (loop [grammar initial-grammar
         iteration 0]
    (if (>= iteration n-iterations)
      grammar
      (let [;; WAKE: solve tasks
            solutions (doall
                       (for [task tasks]
                         {:task task
                          :program (synthesize grammar task 10)}))
            successful (filter :program solutions)

            ;; DREAM: find patterns
            programs (map :program successful)
            patterns (find-common-patterns programs 3)
            abstractions (map abstract-pattern patterns)

            ;; Extend grammar
            grammar' (extend-grammar grammar abstractions)]

        (println "Iteration" iteration
                 "- Solved:" (count successful)
                 "- New abstractions:" (count abstractions))

        (recur grammar' (inc iteration))))))
```

## Pre-training Dataset

For the Number Game domain, pre-training data would be:

```clojure
(def pretraining-data
  [;; Power patterns
   {:examples [1 2 4 8 16 32 64]     :description "powers of 2"}
   {:examples [1 3 9 27 81]          :description "powers of 3"}
   {:examples [1 4 16 64]            :description "powers of 4"}
   {:examples [1 5 25]               :description "powers of 5"}

   ;; Multiple patterns
   {:examples [2 4 6 8 10 12]        :description "multiples of 2"}
   {:examples [3 6 9 12 15 18]       :description "multiples of 3"}
   {:examples [7 14 21 28 35]        :description "multiples of 7"}

   ;; Range patterns
   {:examples [10 11 12 13 14 15]    :description "range 10-15"}
   {:examples [50 51 52 53]          :description "range 50-53"}

   ;; Compositions
   {:examples [6 12 18 24 30]        :description "multiples of 6 (= mult 2 AND mult 3)"}
   {:examples [2 4 8 16 3 9 27]      :description "powers of 2 OR powers of 3"}

   ;; More complex
   {:examples [2 3 5 7 11 13 17 19]  :description "primes"}
   {:examples [1 4 9 16 25 36]       :description "squares"}
   {:examples [1 1 2 3 5 8 13]       :description "fibonacci"}

   ;; ... thousands more examples from mathematical sequences
   ])
```

## Integration with core.logic

After pre-training, export learned grammar to core.logic format:

```clojure
(defn grammar->core-logic
  "Convert learned grammar to core.logic relations."
  [grammar]
  (for [abstraction (:learned-abstractions grammar)]
    `(defn ~(symbol (str (:name abstraction) "o"))
       [~@(:params abstraction) ~'x]
       ~(generate-relation abstraction))))

;; Generates something like:
(defn power-ofo [k n x]
  (l/fresh [result]
    (fd/in k (fd/interval 2 10))
    (fd/in n (fd/interval 0 7))
    (fd/** k n x)))
```

## Why Clojure is Ideal for This

1. **Homoiconicity**: Programs are data, grammars are data, easy to manipulate
2. **Macros**: Can generate grammar rules programmatically
3. **REPL**: Interactive exploration of learned grammars
4. **Immutability**: Track grammar evolution, easy to compare versions
5. **core.logic**: Use learned grammar for fast inference
6. **Spec**: Could define specs for valid grammar rules
7. **Transducers**: Efficient processing of large training sets

## Challenges to Address

1. **Abstraction finding**: How to detect useful patterns?
   - Version spaces
   - Anti-unification
   - Compression-based metrics

2. **Search efficiency**: Enumeration is slow
   - Neural guidance (train NN to propose programs)
   - Type-directed synthesis
   - Constraint propagation

3. **Compositionality**: How do abstractions compose?
   - Typed lambda calculus
   - Combinator libraries

4. **Evaluation**: How to measure grammar quality?
   - Compression of solutions
   - Generalization to new tasks
   - Human interpretability

## References

- Ellis et al. (2020). "DreamCoder: Growing Generalizable, Interpretable Knowledge with Wake-Sleep Bayesian Program Learning"
- Lake et al. (2015). "Human-level concept learning through probabilistic program induction"
- Liang et al. (2010). "Learning Programs: A Hierarchical Bayesian Approach"

## Next Steps (When We Implement This)

1. Implement basic program enumeration
2. Implement pattern detection via anti-unification
3. Implement abstraction extraction
4. Build the wake/dream loop
5. Create pre-training dataset
6. Train and evaluate
7. Export to core.logic for deployment
