(ns numbergame.model
  "Gen.clj generative model for the Number Game.

   This module provides a probabilistic programming approach to the
   number game, complementing the analytic inference in inference.clj.

   The generative story:
     1. Sample a hypothesis h ~ prior
     2. Sample n examples uniformly from h's members
     3. Return the examples

   Use cases:
     - Forward simulation (generating synthetic data)
     - Importance sampling inference
     - Embedding in larger hierarchical models
     - Model comparison and validation"
  (:require [numbergame.hypotheses :as h]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.distribution.commons-math :as dist]
            [gen.generative-function :as gf]
            [gen.trace :as trace]
            [gen.choicemap :as choicemap]
            [gen.inference.importance :as importance]))

;; =============================================================================
;; Core Generative Model
;; =============================================================================

(def number-game-model
  "Gen.clj generative function for the number game.

   Latent choices:
     :hypothesis-idx - index into h/all-hypotheses

   Observable choices:
     [:example i] - the i-th example number

   Args: [n-examples]
   Returns: vector of example numbers"
  (gen [n-examples]
    ;; 1. Sample hypothesis from prior
    (let [h-idx (dynamic/trace! :hypothesis-idx
                                dist/categorical
                                h/uniform-prior)
          {:keys [members]} (nth h/all-hypotheses h-idx)
          members-vec (vec members)
          n-members (count members-vec)]

      ;; 2. Sample n examples uniformly from the hypothesis
      (vec
       (for [i (range n-examples)]
         (let [member-idx (dynamic/trace! [:example i]
                                          dist/uniform-discrete
                                          0 (dec n-members))]
           (nth members-vec member-idx)))))))

(def number-game-model-with-prior
  "Number game model that accepts a custom prior.

   Args: [n-examples prior]
   Returns: vector of example numbers"
  (gen [n-examples prior]
    (let [h-idx (dynamic/trace! :hypothesis-idx
                                dist/categorical
                                prior)
          {:keys [members]} (nth h/all-hypotheses h-idx)
          members-vec (vec members)
          n-members (count members-vec)]
      (vec
       (for [i (range n-examples)]
         (let [member-idx (dynamic/trace! [:example i]
                                          dist/uniform-discrete
                                          0 (dec n-members))]
           (nth members-vec member-idx)))))))

;; =============================================================================
;; Hierarchical Model (Rule vs Interval)
;; =============================================================================

(def n-rule-hypotheses (count h/rule-hypotheses))
(def n-interval-hypotheses (count h/interval-hypotheses))

(defn make-rule-prior
  "Create uniform prior over rule hypotheses only."
  []
  (vec (repeat n-rule-hypotheses (/ 1.0 n-rule-hypotheses))))

(defn make-interval-prior
  "Create uniform prior over interval hypotheses only."
  []
  (vec (repeat n-interval-hypotheses (/ 1.0 n-interval-hypotheses))))

(def hierarchical-number-game
  "Hierarchical model: first choose rule vs interval, then specific hypothesis.

   Latent choices:
     :is-rule?      - boolean, true if rule-based concept
     :rule-idx      - index into rule hypotheses (if rule)
     :interval-idx  - index into interval hypotheses (if interval)
     [:example i]   - the i-th example

   Args: [n-examples rule-prob]
         rule-prob: prior probability of rule-based concept (default 0.5)"
  (gen [n-examples rule-prob]
    (let [is-rule? (dynamic/trace! :is-rule? dist/bernoulli rule-prob)

          ;; Select hypothesis from appropriate subset
          {:keys [members]}
          (if is-rule?
            (let [idx (dynamic/trace! :rule-idx
                                      dist/categorical
                                      (make-rule-prior))]
              (nth h/rule-hypotheses idx))
            (let [idx (dynamic/trace! :interval-idx
                                      dist/categorical
                                      (make-interval-prior))]
              (nth h/interval-hypotheses idx)))

          members-vec (vec members)
          n-members (count members-vec)]

      ;; Sample examples
      (vec
       (for [i (range n-examples)]
         (let [member-idx (dynamic/trace! [:example i]
                                          dist/uniform-discrete
                                          0 (dec n-members))]
           (nth members-vec member-idx)))))))

;; =============================================================================
;; Simulation
;; =============================================================================

(defn simulate
  "Simulate the number game model.
   Returns a trace containing the hypothesis and examples."
  ([n-examples]
   (gf/simulate number-game-model [n-examples]))
  ([n-examples prior]
   (gf/simulate number-game-model-with-prior [n-examples prior])))

(defn simulate-examples
  "Simulate and return just the examples (not the full trace)."
  [n-examples]
  (trace/get-retval (simulate n-examples)))

(defn simulate-with-hypothesis
  "Simulate and return both hypothesis and examples."
  [n-examples]
  (let [tr (simulate n-examples)
        choices (trace/get-choices tr)
        h-idx (choicemap/get-value choices :hypothesis-idx)]
    {:hypothesis (nth h/all-hypotheses h-idx)
     :examples (trace/get-retval tr)}))

;; =============================================================================
;; Gen.clj Inference
;; =============================================================================

(defn examples->constraints
  "Convert example numbers to Gen.clj constraints.

   This requires knowing which hypothesis we're constraining to,
   since we trace member indices, not the numbers directly."
  [hypothesis examples]
  (let [members-vec (vec (:members hypothesis))]
    (into {}
          (map-indexed
           (fn [i example]
             [[:example i] (.indexOf members-vec example)])
           examples))))

(defn importance-sample-hypothesis
  "Use importance sampling to infer the hypothesis.

   Note: This is mainly for demonstration. The analytic posterior
   in inference.clj is exact and faster for this model.

   Returns: map with :hypothesis and :weight"
  [examples n-particles]
  ;; We can't directly constrain on example values since we trace indices.
  ;; Instead, we use the analytic posterior for comparison.
  ;; This function demonstrates how you would do it with Gen.clj
  ;; if the model were more complex.

  ;; For now, simulate and filter (rejection-style importance sampling)
  (let [n-examples (count examples)
        example-set (set examples)
        samples (repeatedly
                 n-particles
                 (fn []
                   (let [tr (simulate n-examples)
                         result (trace/get-retval tr)
                         choices (trace/get-choices tr)
                         h-idx (choicemap/get-value choices :hypothesis-idx)]
                     {:hypothesis-idx h-idx
                      :examples result
                      :matches? (= (set result) example-set)
                      :score (trace/get-score tr)})))
        matching (filter :matches? samples)]
    (if (empty? matching)
      {:error "No matching samples found"
       :n-tried n-particles}
      (let [selected (rand-nth matching)]
        {:hypothesis (nth h/all-hypotheses (:hypothesis-idx selected))
         :n-matching (count matching)
         :n-particles n-particles}))))

;; =============================================================================
;; Model Validation
;; =============================================================================

(defn validate-model
  "Validate that the generative model produces sensible output."
  []
  (println "Validating Number Game Model...")

  ;; Test 1: Simulation produces valid numbers
  (let [examples (simulate-examples 5)]
    (assert (= 5 (count examples)) "Should produce 5 examples")
    (assert (every? #(<= 1 % 100) examples) "Examples should be in 1-100")
    (println "  [OK] Simulation produces valid numbers"))

  ;; Test 2: Hypothesis is valid
  (let [{:keys [hypothesis examples]} (simulate-with-hypothesis 3)]
    (assert (contains? hypothesis :id) "Hypothesis should have :id")
    (assert (contains? hypothesis :members) "Hypothesis should have :members")
    (assert (every? (:members hypothesis) examples)
            "Examples should be in hypothesis")
    (println "  [OK] Examples are consistent with hypothesis"))

  ;; Test 3: Trace has expected structure
  (let [tr (simulate 3)
        choices (trace/get-choices tr)]
    (assert (choicemap/has-value? choices :hypothesis-idx)
            "Trace should have :hypothesis-idx")
    (assert (choicemap/has-value? choices [:example 0])
            "Trace should have [:example 0]")
    (println "  [OK] Trace has expected structure"))

  ;; Test 4: Score is finite
  (let [tr (simulate 3)
        score (trace/get-score tr)]
    (assert (Double/isFinite score) "Score should be finite")
    (println "  [OK] Score is finite"))

  (println "All validations passed!"))

;; =============================================================================
;; Demo
;; =============================================================================

(defn demo
  "Demonstrate the Gen.clj number game model."
  []
  (println "\n=== Gen.clj Number Game Model Demo ===\n")

  (println "1. Simulating from the model...")
  (dotimes [_ 5]
    (let [{:keys [hypothesis examples]} (simulate-with-hypothesis 4)]
      (println (format "   Hypothesis: %-20s Examples: %s"
                       (:id hypothesis) examples))))

  (println "\n2. Examining a trace...")
  (let [tr (simulate 3)
        choices (trace/get-choices tr)]
    (println "   Return value:" (trace/get-retval tr))
    (println "   Log score:" (trace/get-score tr))
    (println "   Hypothesis idx:" (choicemap/get-value choices :hypothesis-idx))
    (println "   Example indices:" (mapv #(choicemap/get-value choices [:example %])
                                         (range 3))))

  (println "\n3. Hierarchical model (rule vs interval)...")
  (dotimes [_ 5]
    (let [tr (gf/simulate hierarchical-number-game [3 0.5])
          choices (trace/get-choices tr)
          is-rule? (choicemap/get-value choices :is-rule?)]
      (println (format "   Rule-based? %-5s Examples: %s"
                       is-rule? (trace/get-retval tr)))))

  (println "\nDemo complete!"))
