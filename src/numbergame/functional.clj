(ns numbergame.functional
  "Idiomatic Clojure approach to program synthesis.

   Key insight: Programs ARE predicates (functions).
   Synthesis is just finding which predicates match all examples.

   This is deeply functional:
   - Programs as first-class functions
   - Grammar as a map of constructors
   - Synthesis as higher-order filtering
   - Composition via function composition"
  (:require [gen.dynamic :as dynamic :refer [gen]]
            [gen.distribution.commons-math :as dist]
            [gen.generative-function :as gf]
            [gen.trace :as trace]))

;; =============================================================================
;; The Grammar: Functions that build predicates
;; =============================================================================

(def domain (set (range 1 101)))

;; Primitive predicates (no parameters)
(def primitive-predicates
  {:prime   (let [ps #{2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97}]
              (fn [x] (contains? ps x)))
   :square  (let [sq #{1 4 9 16 25 36 49 64 81 100}]
              (fn [x] (contains? sq x)))
   :even    (fn [x] (even? x))
   :odd     (fn [x] (odd? x))})

;; Parameterized predicates (functions that return predicates)
(defn power-of
  "Returns predicate: is x a power of base?"
  [base]
  (let [powers (set (take-while #(<= % 100) (iterate #(* % base) 1)))]
    (fn [x] (contains? powers x))))

(defn mult-of
  "Returns predicate: is x a multiple of k?"
  [k]
  (fn [x] (and (pos? x) (<= x 100) (zero? (mod x k)))))

(defn in-range
  "Returns predicate: is lo <= x <= hi?"
  [lo hi]
  (fn [x] (<= lo x hi)))

;; Combinators (higher-order: predicates → predicate)
(defn union
  "Returns predicate: p1(x) OR p2(x)"
  [p1 p2]
  (fn [x] (or (p1 x) (p2 x))))

(defn intersection
  "Returns predicate: p1(x) AND p2(x)"
  [p1 p2]
  (fn [x] (and (p1 x) (p2 x))))

;; =============================================================================
;; The Grammar as Data
;; =============================================================================

(defn all-programs
  "Lazily enumerate all programs in the grammar.
   Each program is {:name ... :pred (fn [x] ...) :size ...}"
  []
  (concat
   ;; Primitives
   (for [[name pred] primitive-predicates]
     {:name name
      :pred pred
      :size (count (filter pred domain))})

   ;; Powers of k
   (for [k (range 2 11)]
     {:name [:power-of k]
      :pred (power-of k)
      :size (count (filter (power-of k) domain))})

   ;; Multiples of k
   (for [k (range 2 21)]
     {:name [:mult k]
      :pred (mult-of k)
      :size (count (filter (mult-of k) domain))})

   ;; Ranges (lazily, sorted by size)
   (for [size (range 1 101)
         lo (range 1 (- 102 size))
         :let [hi (+ lo size -1)]
         :when (<= hi 100)]
     {:name [:range lo hi]
      :pred (in-range lo hi)
      :size size})))

;; =============================================================================
;; Synthesis: The Clojure Way
;; =============================================================================

(defn consistent?
  "Does predicate match all examples?"
  [{:keys [pred]} examples]
  (every? pred examples))

(defn synthesize
  "Find all programs consistent with examples.
   Pure functional style: filter over lazy sequence."
  [examples]
  (->> (all-programs)
       (filter #(consistent? % examples))))

(defn rank-by-size
  "Rank programs by size principle: smaller = better.
   This is just sorting - very Clojure."
  [programs examples]
  (let [n (count examples)]
    (->> programs
         (filter :size)  ; Ensure size exists
         (map #(assoc % :log-likelihood (- (* n (Math/log (max 1 (:size %)))))))
         (sort-by :log-likelihood >))))

(defn infer
  "Full inference: find and rank consistent programs."
  [examples]
  (let [consistent (vec (synthesize examples))
        ranked (rank-by-size consistent examples)]
    (vec (take 10 ranked))))

;; =============================================================================
;; The "Backwards" Part: Data guides the search
;; =============================================================================

(defn data->constraints
  "Extract constraints from data that guide the search.
   This is the 'backwards' direction!"
  [examples]
  (let [lo (apply min examples)
        hi (apply max examples)
        all-even? (every? even? examples)
        all-odd? (every? odd? examples)
        gcd (reduce (fn [a b] (if (zero? b) a (recur b (mod a b)))) examples)]
    {:min lo
     :max hi
     :span (- hi lo)
     :all-even? all-even?
     :all-odd? all-odd?
     :gcd gcd
     :possible-bases (vec (filter #(every? (power-of %) examples) (range 2 11)))
     :possible-mults (vec (filter #(every? (mult-of %) examples) (range 2 21)))}))

(defn guided-synthesize
  "Use data constraints to guide search - more efficient.
   The data tells us where to look!"
  [examples]
  (let [{:keys [min max possible-bases possible-mults all-even? all-odd?]}
        (data->constraints examples)]
    (concat
     ;; Check specific patterns suggested by data
     (when (seq possible-bases)
       (for [b possible-bases]
         {:name [:power-of b] :pred (power-of b)
          :size (count (filter (power-of b) domain))}))

     (when (seq possible-mults)
       (for [k possible-mults]
         {:name [:mult k] :pred (mult-of k)
          :size (count (filter (mult-of k) domain))}))

     (when all-even?
       [{:name :even :pred even? :size 50}])

     (when all-odd?
       [{:name :odd :pred odd? :size 50}])

     ;; Primitives that match
     (for [[name pred] primitive-predicates
           :when (every? pred examples)]
       {:name name :pred pred :size (count (filter pred domain))})

     ;; Ranges (constrained by data bounds)
     (for [lo (range 1 (inc min))
           hi (range max 101)]
       {:name [:range lo hi] :pred (in-range lo hi) :size (inc (- hi lo))}))))

;; =============================================================================
;; Gen.clj Integration: Probabilistic inference
;; =============================================================================

(defn programs->distribution
  "Convert ranked programs to probability distribution."
  [programs examples]
  (let [n (count examples)
        log-weights (mapv #(- (* n (Math/log (:size %)))) programs)
        max-lw (apply max log-weights)
        weights (mapv #(Math/exp (- % max-lw)) log-weights)
        total (reduce + weights)]
    (mapv #(/ % total) weights)))

(def inference-model
  "Gen.clj model for probabilistic program selection."
  (gen [examples]
    (let [programs (vec (infer examples))
          probs (programs->distribution programs examples)
          idx (dynamic/trace! :program dist/categorical probs)]
      (nth programs idx))))

(defn sample-posterior
  "Sample from posterior over programs using Gen.clj."
  [examples n-samples]
  (let [programs (vec (infer examples))
        probs (programs->distribution programs examples)]
    {:programs programs
     :probs (zipmap (map :name programs) probs)
     :samples (frequencies
               (repeatedly n-samples
                 #(:name (nth programs
                           (let [tr (gf/simulate inference-model [examples])]
                             (:name (trace/get-retval tr)))))))}))

;; =============================================================================
;; Demo
;; =============================================================================

(defn demo []
  (println "\n" (apply str (repeat 60 "=")) "\n")
  (println "FUNCTIONAL SYNTHESIS: Programs as Predicates")
  (println (apply str (repeat 60 "=")) "\n")

  (println "The Clojure insight: Programs ARE functions.")
  (println "Synthesis is just: (filter #(every? (:pred %) examples) programs)\n")

  (let [examples [16 8 2 64]]
    (println "Examples:" examples)
    (println "\nData-derived constraints (backwards!):")
    (let [constraints (data->constraints examples)]
      (doseq [[k v] (dissoc constraints :possible-bases :possible-mults)]
        (println (format "  %-12s %s" (name k) v)))
      (println (format "  %-12s %s" "bases" (:possible-bases constraints)))
      (println (format "  %-12s %s" "mults" (:possible-mults constraints))))

    (println "\nTop programs (ranked by size principle):")
    (doseq [{:keys [name size]} (take 5 (infer examples))]
      (println (format "  %-20s size: %d" (pr-str name) size))))

  (println "\n" (apply str (repeat 60 "-")) "\n")

  (let [examples [7 11 13 17]]
    (println "Examples:" examples)
    (println "\nData-derived constraints:")
    (let [constraints (data->constraints examples)]
      (println (format "  all-odd? %s" (:all-odd? constraints)))
      (println (format "  possible primes? %s"
                      (every? (:prime primitive-predicates) examples))))

    (println "\nTop programs:")
    (doseq [{:keys [name size]} (take 5 (infer examples))]
      (println (format "  %-20s size: %d" (pr-str name) size))))

  (println "\n" (apply str (repeat 60 "=")) "\n")
  (println "Key Clojure idioms used:")
  (println "  - Programs as first-class functions (predicates)")
  (println "  - Lazy sequences for grammar enumeration")
  (println "  - Higher-order filtering for synthesis")
  (println "  - Data-driven constraints (backwards from data)")
  (println "  - Gen.clj only for probabilistic sampling"))

(comment
  (demo)

  ;; The "backwards" insight: data tells us what to look for
  (data->constraints [16 8 2 64])
  ;; => {:possible-bases [2], :possible-mults [2 4 8], ...}

  ;; Synthesis is just filtering predicates
  (take 5 (infer [16 8 2 64]))

  ;; Programs ARE functions - we can call them
  (let [{:keys [pred]} (first (infer [16 8 2 64]))]
    (pred 32))  ;; => true
  )
