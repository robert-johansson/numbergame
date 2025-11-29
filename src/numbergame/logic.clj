(ns numbergame.logic
  "Program synthesis using core.logic.

   The key insight: define membership INTENSIONALLY (as rules) rather than
   EXTENSIONALLY (as pre-computed sets). core.logic then finds programs
   that satisfy constraints through constraint propagation.

   Advantages:
   - No pre-computation needed
   - Parameter discovery (finds k=2, not just 'pow2')
   - Natural compositionality
   - Bidirectional: same relation for membership test and synthesis"
  (:require [clojure.core.logic :as l]
            [clojure.core.logic.fd :as fd]))

;; =============================================================================
;; Domain Constants
;; =============================================================================

(def domain-min 1)
(def domain-max 100)
(def domain-interval (fd/interval domain-min domain-max))

;; Primes in 1-100 (pre-computed for efficiency)
(def primes
  #{2 3 5 7 11 13 17 19 23 29 31 37 41 43 47
    53 59 61 67 71 73 79 83 89 97})

;; =============================================================================
;; Helper Relations (Primitives)
;; =============================================================================

(defn power-ofo
  "Relation: x = base^exp for some exp in [0, max-exp].
   Example: (power-ofo 2 x 7) constrains x to be a power of 2."
  [base x max-exp]
  (l/fresh [exp result]
    (fd/in exp (fd/interval 0 max-exp))
    (fd/in x domain-interval)
    ;; x = base^exp
    ;; fd/** doesn't exist in core.logic, we need to enumerate
    (l/conde
      [(l/== exp 0) (l/== x 1)]
      [(l/== exp 1) (l/== x base)]
      [(l/fresh [prev]
         (fd/> exp 1)
         (fd/in prev domain-interval)
         (power-ofo base prev (dec max-exp))
         (fd/* prev base x))])))

;; Actually, fd/** doesn't work well for our needs. Let's use enumeration instead.
;; This is more practical for the Number Game domain.

(defn powers-of
  "Returns set of powers of base in domain [1, 100]."
  [base]
  (loop [n 1 acc #{}]
    (let [power (long (Math/pow base n))]
      (if (> power domain-max)
        (conj acc 1)  ; Include base^0 = 1
        (recur (inc n) (conj acc power))))))

(defn multiples-of
  "Returns set of multiples of k in domain [1, 100]."
  [k]
  (set (filter #(<= % domain-max) (map #(* k %) (range 1 (inc (quot domain-max k)))))))

(defn range-set
  "Returns set of integers in [lo, hi]."
  [lo hi]
  (set (range lo (inc hi))))

(def squares
  "Perfect squares in [1, 100]."
  #{1 4 9 16 25 36 49 64 81 100})

(def evens
  "Even numbers in [1, 100]."
  (set (filter even? (range 1 (inc domain-max)))))

(def odds
  "Odd numbers in [1, 100]."
  (set (filter odd? (range 1 (inc domain-max)))))

;; =============================================================================
;; Membership Relations (Intensional Definitions)
;; =============================================================================

(defn power-of-o
  "Relation: x is a power of base (base in [2, 10]).
   Enumerates all valid bases since core.logic can't project unbound vars."
  [base x]
  (l/conde
    [(l/== base 2) (l/membero x (vec (powers-of 2)))]
    [(l/== base 3) (l/membero x (vec (powers-of 3)))]
    [(l/== base 4) (l/membero x (vec (powers-of 4)))]
    [(l/== base 5) (l/membero x (vec (powers-of 5)))]
    [(l/== base 6) (l/membero x (vec (powers-of 6)))]
    [(l/== base 7) (l/membero x (vec (powers-of 7)))]
    [(l/== base 8) (l/membero x (vec (powers-of 8)))]
    [(l/== base 9) (l/membero x (vec (powers-of 9)))]
    [(l/== base 10) (l/membero x (vec (powers-of 10)))]))

(defn mult-of-o
  "Relation: x is a multiple of k (k in [2, 20]).
   Enumerates all valid k values."
  [k x]
  (l/conde
    [(l/== k 2) (l/membero x (vec (multiples-of 2)))]
    [(l/== k 3) (l/membero x (vec (multiples-of 3)))]
    [(l/== k 4) (l/membero x (vec (multiples-of 4)))]
    [(l/== k 5) (l/membero x (vec (multiples-of 5)))]
    [(l/== k 6) (l/membero x (vec (multiples-of 6)))]
    [(l/== k 7) (l/membero x (vec (multiples-of 7)))]
    [(l/== k 8) (l/membero x (vec (multiples-of 8)))]
    [(l/== k 9) (l/membero x (vec (multiples-of 9)))]
    [(l/== k 10) (l/membero x (vec (multiples-of 10)))]
    [(l/== k 11) (l/membero x (vec (multiples-of 11)))]
    [(l/== k 12) (l/membero x (vec (multiples-of 12)))]
    [(l/== k 13) (l/membero x (vec (multiples-of 13)))]
    [(l/== k 14) (l/membero x (vec (multiples-of 14)))]
    [(l/== k 15) (l/membero x (vec (multiples-of 15)))]
    [(l/== k 16) (l/membero x (vec (multiples-of 16)))]
    [(l/== k 17) (l/membero x (vec (multiples-of 17)))]
    [(l/== k 18) (l/membero x (vec (multiples-of 18)))]
    [(l/== k 19) (l/membero x (vec (multiples-of 19)))]
    [(l/== k 20) (l/membero x (vec (multiples-of 20)))]))

(defn range-o
  "Relation: x is in range [lo, hi].
   lo and hi should be ground when checking membership."
  [lo hi x]
  (l/all
    (fd/in lo (fd/interval domain-min domain-max))
    (fd/in hi (fd/interval domain-min domain-max))
    (fd/in x (fd/interval domain-min domain-max))
    (fd/<= lo hi)
    (fd/>= x lo)
    (fd/<= x hi)))

(defn prime-o
  "Relation: x is a prime number in [1, 100]."
  [x]
  (l/membero x (vec primes)))

(defn square-o
  "Relation: x is a perfect square in [1, 100]."
  [x]
  (l/membero x (vec squares)))

(defn even-o
  "Relation: x is even in [1, 100]."
  [x]
  (l/membero x (vec evens)))

(defn odd-o
  "Relation: x is odd in [1, 100]."
  [x]
  (l/membero x (vec odds)))

;; =============================================================================
;; The Main in-concept Relation
;; =============================================================================

(declare in-concept-bounded)

(defn in-concept-primitives
  "Relation: x is a member of the concept defined by prog (primitives only).
   Order matters: specific rules first, general ranges last."
  [prog x]
  (l/conde
    ;; Power of k: x = k^n (most specific, try first)
    [(l/fresh [k]
       (l/== prog [:power-of k])
       (power-of-o k x))]

    ;; Prime (specific)
    [(l/== prog [:prime])
     (prime-o x)]

    ;; Square (specific)
    [(l/== prog [:square])
     (square-o x)]

    ;; Multiple of k: x = k*m
    [(l/fresh [k]
       (l/== prog [:mult k])
       (mult-of-o k x))]

    ;; Even
    [(l/== prog [:even])
     (even-o x)]

    ;; Odd
    [(l/== prog [:odd])
     (odd-o x)]

    ;; Range [lo, hi] (most general, try last)
    [(l/fresh [lo hi]
       (l/== prog [:range lo hi])
       (range-o lo hi x))]))

(defn in-concept-bounded
  "Relation: x is a member of the concept defined by prog.
   depth bounds recursive compositions to prevent infinite search."
  [prog x depth]
  (if (zero? depth)
    ;; Base case: only primitives
    (in-concept-primitives prog x)
    ;; Can also include compositions
    (l/conde
      ;; All primitives
      [(in-concept-primitives prog x)]

      ;; Union: x in left OR x in right
      [(l/fresh [left right]
         (l/== prog [:union left right])
         (l/conde
           [(in-concept-bounded left x (dec depth))]
           [(in-concept-bounded right x (dec depth))]))]

      ;; Intersect: x in left AND x in right
      [(l/fresh [left right]
         (l/== prog [:intersect left right])
         (in-concept-bounded left x (dec depth))
         (in-concept-bounded right x (dec depth)))])))

(defn in-concept
  "Default in-concept with depth 1 (allows one level of composition)."
  [prog x]
  (in-concept-bounded prog x 1))

;; =============================================================================
;; Program Synthesis from Examples
;; =============================================================================

(defn find-programs-by-type
  "Find programs of a specific type that contain all examples.
   type-relation is one of the primitive relations."
  [examples type-key type-relation n]
  (l/run n [prog]
    (type-relation prog)
    (l/everyg (fn [ex] (in-concept-primitives prog ex)) examples)))

;; Individual type finders
(defn find-power-programs [examples n]
  (l/run n [prog]
    (l/fresh [k]
      (l/== prog [:power-of k])
      (l/everyg (fn [ex] (power-of-o k ex)) examples))))

(defn find-mult-programs [examples n]
  (l/run n [prog]
    (l/fresh [k]
      (l/== prog [:mult k])
      (l/everyg (fn [ex] (mult-of-o k ex)) examples))))

(defn find-range-programs [examples n]
  "Generate range programs ordered by size (tightest first)."
  (let [lo (apply min examples)
        hi (apply max examples)]
    ;; Generate all valid ranges and sort by size
    (->> (for [rlo (range domain-min (inc lo))
               rhi (range hi (inc domain-max))]
           [:range rlo rhi])
         (sort-by (fn [[_ rlo rhi]] (- rhi rlo)))
         (take n))))

(defn find-fixed-programs [examples]
  "Find fixed-type programs (prime, square, even, odd) that contain all examples."
  (let [check (fn [members] (every? members examples))]
    (cond-> []
      (check primes)  (conj [:prime])
      (check squares) (conj [:square])
      (check evens)   (conj [:even])
      (check odds)    (conj [:odd]))))

(defn find-programs-primitives
  "Find primitive programs (no compositions) that contain all examples.
   Searches each type separately to avoid fd flooding."
  [examples n]
  (let [;; Fixed types (fast check)
        fixed (find-fixed-programs examples)
        ;; Parameterized types
        powers (find-power-programs examples 20)
        mults (find-mult-programs examples 30)
        ;; Ranges (limit to avoid explosion)
        ranges (find-range-programs examples (max 10 (- n (count fixed) (count powers) (count mults))))]
    ;; Combine and dedupe
    (vec (distinct (concat fixed powers mults ranges)))))

(defn find-programs
  "Find programs (up to depth d) whose concept contains all examples."
  ([examples n] (find-programs examples n 1))
  ([examples n depth]
   (l/run n [prog]
     (l/everyg (fn [ex] (in-concept-bounded prog ex depth)) examples))))

;; =============================================================================
;; Computing Concept Size (for Size Principle Ranking)
;; =============================================================================

(defn concept-extension
  "Compute the extension (set of members) of a program.
   Direct computation - doesn't use core.logic (much faster)."
  [prog]
  (case (first prog)
    :power-of (powers-of (second prog))
    :mult (multiples-of (second prog))
    :range (range-set (second prog) (nth prog 2))
    :prime primes
    :square squares
    :even evens
    :odd odds
    :union (clojure.set/union
            (concept-extension (second prog))
            (concept-extension (nth prog 2)))
    :intersect (clojure.set/intersection
                (concept-extension (second prog))
                (concept-extension (nth prog 2)))
    #{}))

(defn concept-size
  "Compute the size of a concept (how many numbers it contains)."
  [prog]
  (count (concept-extension prog)))

;; =============================================================================
;; Priors (as data!)
;; =============================================================================

(def uniform-prior
  "Equal weight to all program types."
  {:power-of 1.0
   :mult 1.0
   :range 1.0
   :prime 1.0
   :square 1.0
   :even 1.0
   :odd 1.0
   :union 1.0
   :intersect 1.0})

(def rule-biased-prior
  "Strongly prefer mathematical rules over intervals.
   This makes 'primes' beat 'range 7-17' for [7 11 13 17]."
  {:power-of 10.0   ; Mathematical pattern
   :mult 10.0       ; Mathematical pattern
   :range 1.0       ; Generic interval (baseline)
   :prime 10.0      ; Named mathematical set
   :square 10.0     ; Named mathematical set
   :even 5.0        ; Simple rule
   :odd 5.0         ; Simple rule
   :union 0.5       ; Penalize compositions slightly
   :intersect 0.5})

(def specificity-prior
  "Prefer more specific/constrained concepts."
  {:power-of 20.0   ; Very specific pattern
   :mult 5.0        ; Moderately specific
   :range 1.0       ; Can be anything
   :prime 15.0      ; Specific named set
   :square 15.0     ; Specific named set
   :even 2.0        ; Very broad
   :odd 2.0         ; Very broad
   :union 0.3
   :intersect 0.3})

(defn log-prior
  "Compute log prior for a program given a prior map.

   Key insight from Tenenbaum: The prior for a SPECIFIC interval [a,b]
   should be much lower than for a named rule like 'primes', because
   there are ~5000 possible intervals but only one 'primes' concept.

   We model this by dividing interval priors by the number of intervals."
  [prior prog]
  (let [prog-type (first prog)
        base-weight (get prior prog-type 1.0)
        ;; Intervals get divided by approximate count of possible intervals
        ;; This concentrates probability on named rules
        adjusted-weight (if (= prog-type :range)
                          (/ base-weight 5000.0)
                          base-weight)]
    (Math/log adjusted-weight)))

;; =============================================================================
;; Ranking by Posterior (Prior × Likelihood)
;; =============================================================================

(defn rank-programs
  "Rank programs by posterior: prior × likelihood.
   Uses size principle for likelihood, optional prior map.
   Returns sorted list with scores."
  ([programs examples] (rank-programs programs examples uniform-prior))
  ([programs examples prior]
   (let [n (count examples)
         ;; Normalize prior weights to sum to 1
         total-prior (reduce + (vals prior))
         norm-prior (into {} (map (fn [[k v]] [k (/ v total-prior)]) prior))]
     (->> programs
          (map (fn [prog]
                 (let [size (concept-size prog)
                       safe-size (max 1 size)
                       log-like (- (* n (Math/log safe-size)))
                       log-pr (log-prior norm-prior prog)
                       log-post (+ log-pr log-like)]
                   {:program prog
                    :size size
                    :log-prior log-pr
                    :log-likelihood log-like
                    :log-posterior log-post})))
          (sort-by :log-posterior >)))))

;; =============================================================================
;; Full Synthesis Pipeline
;; =============================================================================

(defn synthesize
  "Complete program synthesis: find and rank programs.

   Arguments:
   - examples: vector of positive examples (e.g., [16 8 2 64])
   - max-programs: maximum number of programs to find
   - depth: composition depth (0 = primitives only, 1 = one level of union/intersect)
   - prior: prior map (default: uniform-prior)

   Returns map with:
   - :examples - input examples
   - :n-found - number of programs found
   - :top-programs - ranked programs with scores"
  ([examples] (synthesize examples 100 1 uniform-prior))
  ([examples max-programs] (synthesize examples max-programs 1 uniform-prior))
  ([examples max-programs depth] (synthesize examples max-programs depth uniform-prior))
  ([examples max-programs depth prior]
   (let [programs (find-programs examples max-programs depth)
         ranked (rank-programs programs examples prior)]
     {:examples examples
      :prior (if (= prior uniform-prior) :uniform
               (if (= prior rule-biased-prior) :rule-biased
                 :custom))
      :n-found (count programs)
      :top-programs (take 10 ranked)})))

(defn synthesize-primitives
  "Synthesize using only primitive concepts (no compositions).
   Faster and often sufficient.

   Arguments:
   - examples: vector of positive examples
   - max-programs: maximum programs to find (default 50)
   - prior: prior map (default: uniform-prior)"
  ([examples] (synthesize-primitives examples 50 uniform-prior))
  ([examples max-programs] (synthesize-primitives examples max-programs uniform-prior))
  ([examples max-programs prior]
   (let [programs (find-programs-primitives examples max-programs)
         ranked (rank-programs programs examples prior)]
     {:examples examples
      :prior (cond (= prior uniform-prior) :uniform
                   (= prior rule-biased-prior) :rule-biased
                   (= prior specificity-prior) :specificity
                   :else :custom)
      :n-found (count programs)
      :top-programs (take 10 ranked)})))

;; =============================================================================
;; Demo / REPL Helpers
;; =============================================================================

(defn demo
  "Run synthesis demo with classic Number Game stimuli."
  []
  (println "\n" (apply str (repeat 70 "=")) "\n")
  (println "CORE.LOGIC PROGRAM SYNTHESIS")
  (println (apply str (repeat 70 "=")) "\n")

  (println "The core.logic approach defines membership INTENSIONALLY.")
  (println "No pre-computed hypothesis sets - core.logic finds programs.\n")

  (doseq [[name examples] [["Powers of 2" [16 8 2 64]]
                           ["Interval" [16 17 18 19]]
                           ["Primes" [7 11 13 17]]
                           ["Multiples of 7" [7 14 21 28]]]]
    (println (str name ": " examples))
    (let [{:keys [n-found top-programs]} (synthesize-primitives examples 20)]
      (println (format "  Found %d consistent programs" n-found))
      (doseq [{:keys [program size log-posterior]} (take 3 top-programs)]
        (println (format "    %-25s size: %3d  log-post: %.2f"
                        (pr-str program) size log-posterior))))
    (println)))

(defn demo-priors
  "Demonstrate how different priors affect hypothesis ranking."
  []
  (println "\n" (apply str (repeat 70 "=")) "\n")
  (println "EFFECT OF PRIORS ON HYPOTHESIS RANKING")
  (println (apply str (repeat 70 "=")) "\n")

  (println "The same examples can yield different top hypotheses")
  (println "depending on the prior over program types.\n")

  (let [examples [7 11 13 17]]
    (println "Examples:" examples "\n")

    (println "1. UNIFORM PRIOR (all types equal)")
    (println "   Size principle dominates - smaller hypotheses win")
    (let [{:keys [top-programs]} (synthesize-primitives examples 30 uniform-prior)]
      (doseq [{:keys [program size]} (take 3 top-programs)]
        (println (format "     %-20s size: %d" (pr-str program) size))))

    (println "\n2. RULE-BIASED PRIOR (prefer mathematical concepts)")
    (println "   Rules get 10x weight over intervals")
    (let [{:keys [top-programs]} (synthesize-primitives examples 30 rule-biased-prior)]
      (doseq [{:keys [program size]} (take 3 top-programs)]
        (println (format "     %-20s size: %d" (pr-str program) size))))

    (println "\n3. SPECIFICITY PRIOR (prefer constrained concepts)")
    (println "   Powers get 20x, primes get 15x")
    (let [{:keys [top-programs]} (synthesize-primitives examples 30 specificity-prior)]
      (doseq [{:keys [program size]} (take 3 top-programs)]
        (println (format "     %-20s size: %d" (pr-str program) size)))))

  (println "\n" (apply str (repeat 70 "-")) "\n")

  (println "Custom priors are just maps - very Clojure!")
  (println "  (synthesize-primitives [7 11 13 17] 30 {:prime 100.0 :range 1.0 ...})\n")

  (println "Or compose priors with merge/update:")
  (println "  (merge uniform-prior {:prime 50.0})  ; boost just primes")
  (println "  (update rule-biased-prior :range * 0.1)  ; penalize ranges more"))

(defn show-extension
  "Show the extension (members) of a program."
  [prog]
  (let [ext (concept-extension prog)]
    {:program prog
     :size (count ext)
     :members (vec (sort ext))}))

(comment
  ;; REPL examples

  ;; Find programs for powers of 2
  (find-programs-primitives [16 8 2 64] 10)
  ;; => ([:power-of 2] [:even] [:mult 2] ...)

  ;; Full synthesis with ranking
  (synthesize [16 8 2 64])
  ;; => {:top-programs [{:program [:power-of 2] :size 7 ...} ...]}

  ;; What does [:power-of 2] mean?
  (show-extension [:power-of 2])
  ;; => {:program [:power-of 2], :size 7, :members [1 2 4 8 16 32 64]}

  ;; Discover parameter: what k makes all examples multiples of k?
  (find-programs-primitives [14 28 42 56] 10)
  ;; => ([:mult 2] [:mult 7] [:mult 14] [:even] ...)
  ;; core.logic DISCOVERED that k=7 and k=14 work!

  ;; Run demo
  (demo)
  )
