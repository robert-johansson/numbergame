(ns numbergame.synthesis
  "Program synthesis using spec + Gen.clj (no core.logic).

   Demonstrates that for finite grammars, we can:
   - Use spec for grammar definition and validation
   - Use pure Clojure for consistent program finding
   - Use Gen.clj for probabilistic inference

   core.logic would add value for infinite domains or complex
   relational constraints, but isn't needed for the Number Game."
  (:require [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as sgen]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.distribution.commons-math :as dist]
            [gen.generative-function :as gf]
            [gen.trace :as trace]
            [gen.choicemap :as choicemap]))

;; =============================================================================
;; Domain
;; =============================================================================

(def domain (set (range 1 101)))

;; Pre-computed concept extensions
(defn powers-of [base]
  (set (take-while #(<= % 100) (iterate #(* % base) 1))))

(defn multiples-of [k]
  (set (filter #(<= % 100) (map #(* k %) (range 1 101)))))

(def primes #{2 3 5 7 11 13 17 19 23 29 31 37 41 43 47
              53 59 61 67 71 73 79 83 89 97})

(def squares #{1 4 9 16 25 36 49 64 81 100})
(def evens (set (filter even? domain)))
(def odds (set (filter odd? domain)))

;; =============================================================================
;; spec: Program Grammar
;; =============================================================================

(s/def ::base (s/int-in 2 11))
(s/def ::mult-k (s/int-in 2 21))
(s/def ::lo (s/int-in 1 101))
(s/def ::hi (s/int-in 1 101))

(s/def ::power-of (s/cat :type #{:power-of} :base ::base))
(s/def ::mult (s/cat :type #{:mult} :k ::mult-k))
(s/def ::range (s/and (s/cat :type #{:range} :lo ::lo :hi ::hi)
                      #(<= (:lo %) (:hi %))))
(s/def ::fixed #{[:prime] [:square] [:even] [:odd]})

(s/def ::program
  (s/or :power-of ::power-of
        :mult ::mult
        :range ::range
        :fixed ::fixed))

;; =============================================================================
;; Pure Clojure: Program Extension & Consistency
;; =============================================================================

(defn extension
  "Get the set of numbers a program covers."
  [[type & args]]
  (case type
    :power-of (powers-of (first args))
    :mult (multiples-of (first args))
    :range (set (range (first args) (inc (second args))))
    :prime primes
    :square squares
    :even evens
    :odd odds
    #{}))

(defn consistent?
  "Does program contain all examples?"
  [prog examples]
  (let [ext (extension prog)]
    (every? ext examples)))

;; =============================================================================
;; Pure Clojure: Enumerate All Consistent Programs
;; =============================================================================

(defn all-primitive-programs
  "Enumerate all primitive programs in the grammar."
  []
  (concat
   ;; Fixed types
   [[:prime] [:square] [:even] [:odd]]
   ;; Power of k for k in 2..10
   (for [k (range 2 11)] [:power-of k])
   ;; Multiple of k for k in 2..20
   (for [k (range 2 21)] [:mult k])
   ;; Ranges - but we'll generate these lazily based on examples
   ))

(defn consistent-ranges
  "Generate ranges consistent with examples, ordered by size."
  [examples n]
  (let [lo (apply min examples)
        hi (apply max examples)]
    (->> (for [rlo (range 1 (inc lo))
               rhi (range hi 101)]
           [:range rlo rhi])
         (sort-by (fn [[_ lo hi]] (- hi lo)))
         (take n))))

(defn find-consistent
  "Find all programs consistent with examples.
   Pure Clojure - no core.logic needed!"
  [examples max-programs]
  (let [fixed-and-param (filter #(consistent? % examples) (all-primitive-programs))
        ranges (take (- max-programs (count fixed-and-param))
                     (consistent-ranges examples 100))]
    (vec (concat fixed-and-param ranges))))

;; =============================================================================
;; Gen.clj: Probabilistic Model
;; =============================================================================

(def type-prior
  "Prior weights over program types."
  {:power-of 10.0
   :mult 10.0
   :range 0.0002  ; Divided by ~5000 intervals
   :prime 10.0
   :square 10.0
   :even 5.0
   :odd 5.0})

(defn log-prior [prog]
  (let [ptype (first prog)]
    (Math/log (get type-prior ptype 1.0))))

(defn log-likelihood [prog examples]
  (let [size (count (extension prog))
        n (count examples)]
    (- (* n (Math/log (max 1 size))))))

(defn log-posterior [prog examples]
  (+ (log-prior prog) (log-likelihood prog examples)))

(defn posterior-probs
  "Compute normalized posterior probabilities."
  [programs examples]
  (let [log-posts (mapv #(log-posterior % examples) programs)
        max-lp (apply max log-posts)
        weights (mapv #(Math/exp (- % max-lp)) log-posts)
        total (reduce + weights)]
    (mapv #(/ % total) weights)))

;; Gen.clj generative model
(def number-game-model
  "Generative model for the Number Game.

   Forward direction: sample program → sample examples
   Uses Gen.clj tracing for probabilistic inference."
  (gen [n-examples]
    (let [;; Sample program type
          type-idx (dynamic/trace! :type dist/categorical [0.15 0.15 0.3 0.1 0.1 0.1 0.1])
          prog-type ([:power-of :mult :range :prime :square :even :odd] type-idx)

          ;; Sample parameters based on type
          prog (case prog-type
                 :power-of (let [base (+ 2 (dynamic/trace! :base dist/categorical
                                                          (vec (repeat 9 (/ 1.0 9)))))]
                            [:power-of base])
                 :mult (let [k (+ 2 (dynamic/trace! :mult-k dist/categorical
                                                   (vec (repeat 19 (/ 1.0 19)))))]
                         [:mult k])
                 :range (let [lo (inc (dynamic/trace! :lo dist/categorical
                                                     (vec (repeat 100 0.01))))
                              hi (+ lo (dynamic/trace! :span dist/categorical
                                                       (vec (repeat (- 101 lo) (/ 1.0 (- 101 lo))))))]
                          [:range lo hi])
                 [:prime] prog-type  ; Fixed types
                 [:square] prog-type
                 [:even] prog-type
                 [:odd] prog-type)

          ;; Sample examples from program's extension
          ext (vec (extension prog))
          n-ext (count ext)
          examples (when (pos? n-ext)
                     (vec (for [i (range n-examples)]
                            (let [idx (dynamic/trace! (keyword (str "ex" i))
                                                     dist/categorical
                                                     (vec (repeat n-ext (/ 1.0 n-ext))))]
                              (nth ext idx)))))]

      {:program prog
       :extension-size n-ext
       :examples examples})))

;; Gen.clj inference model
(def inference-model
  "Model for posterior inference given examples."
  (gen [examples]
    (let [programs (find-consistent examples 50)
          probs (posterior-probs programs examples)
          idx (dynamic/trace! :program dist/categorical probs)]
      {:selected (nth programs idx)
       :all-programs programs
       :probabilities (zipmap programs probs)})))

;; =============================================================================
;; Inference Functions
;; =============================================================================

(defn infer
  "Run inference to find most likely program.
   Returns posterior distribution over consistent programs."
  [examples & {:keys [n-samples] :or {n-samples 1000}}]
  (let [programs (find-consistent examples 50)
        probs (posterior-probs programs examples)

        ;; Sample from posterior using Gen.clj
        sample-fn (fn []
                    (let [tr (gf/simulate inference-model [examples])]
                      (:selected (trace/get-retval tr))))
        samples (frequencies (repeatedly n-samples sample-fn))]

    {:examples examples
     :n-consistent (count programs)
     :posterior (->> (zipmap programs probs)
                     (sort-by val >)
                     (take 10))
     :samples (->> samples (sort-by val >) (take 5))}))

(defn simulate-and-infer
  "Full cycle: simulate data, then infer back.
   Demonstrates Gen.clj's traced execution."
  []
  (let [;; Forward: generate concept and examples
        forward-trace (gf/simulate number-game-model [4])
        generated (trace/get-retval forward-trace)

        ;; Backward: infer concept from examples
        examples (:examples generated)
        inference-result (when examples (infer examples :n-samples 500))]

    {:generated generated
     :forward-score (trace/get-score forward-trace)
     :inference inference-result
     :recovered? (when inference-result
                   (= (:program generated)
                      (first (first (:posterior inference-result)))))}))

;; =============================================================================
;; Demo
;; =============================================================================

(defn demo []
  (println "\n" (apply str (repeat 70 "=")) "\n")
  (println "SYNTHESIS WITH spec + Gen.clj (no core.logic)")
  (println (apply str (repeat 70 "=")) "\n")

  (println "1. SPEC: Grammar Definition")
  (println "   Valid? [:power-of 2] =>" (s/valid? ::program [:power-of 2]))
  (println "   Valid? [:power-of 99] =>" (s/valid? ::program [:power-of 99]))
  (println)

  (println "2. PURE CLOJURE: Find Consistent Programs")
  (let [examples [16 8 2 64]
        progs (find-consistent examples 10)]
    (println "   Examples:" examples)
    (println "   Consistent:" (take 5 progs)))
  (println)

  (println "3. GEN.CLJ: Posterior Inference")
  (let [result (infer [7 11 13 17] :n-samples 500)]
    (println "   Examples: [7 11 13 17]")
    (println "   Top by posterior:")
    (doseq [[prog prob] (take 3 (:posterior result))]
      (println (format "     %-20s %.4f" (pr-str prog) prob))))
  (println)

  (println "4. GEN.CLJ: Simulate → Infer Round-Trip")
  (let [{:keys [generated inference recovered?]} (simulate-and-infer)]
    (println "   Generated:" (:program generated) "→" (:examples generated))
    (when inference
      (println "   Inferred:" (first (first (:posterior inference))))
      (println "   Recovered correctly?" recovered?)))
  (println)

  (println (apply str (repeat 70 "-")))
  (println "\ncore.logic would add value for:")
  (println "  - Infinite domains (we have finite 1-100)")
  (println "  - Complex relational constraints")
  (println "  - Bidirectional reasoning")
  (println "\nFor the Number Game, spec + Gen.clj is sufficient!"))

(comment
  ;; REPL examples

  ;; Find consistent programs (pure Clojure)
  (find-consistent [16 8 2 64] 10)
  ;; => [[:power-of 2] [:even] [:mult 2] [:mult 4] ...]

  ;; Posterior inference
  (infer [7 11 13 17])

  ;; Simulate from generative model
  (let [tr (gf/simulate number-game-model [4])]
    (trace/get-retval tr))

  ;; Full round-trip
  (simulate-and-infer)

  ;; Run demo
  (demo)
  )
