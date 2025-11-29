(ns numbergame.generalization
  "Generalization predictions for the Number Game.

   Computes p(y in concept | examples) - the probability that
   a test number belongs to the same concept as the observed examples."
  (:require [numbergame.hypotheses :as h]
            [numbergame.inference :as infer]))

;; =============================================================================
;; Core Generalization
;; =============================================================================

(defn p-in-concept
  "Compute p(y in concept | examples).

   This is the posterior predictive probability that test number y
   belongs to the same concept as the observed examples.

   p(y in C | X) = sum_h 1[y in h] * p(h | X)"
  ([examples y] (p-in-concept examples y h/prior))
  ([examples y prior]
   (let [post (infer/posterior examples prior)]
     (transduce
      (comp (filter (fn [[hyp _]] (contains? (:members hyp) y)))
            (map second))
      + 0.0
      (map vector h/all-hypotheses post)))))

(defn p-in-concept-from-posterior
  "Compute p(y in concept) given a pre-computed posterior.
   More efficient when testing many y values."
  [posterior y]
  (transduce
   (comp (filter (fn [[hyp _]] (contains? (:members hyp) y)))
         (map second))
   + 0.0
   (map vector h/all-hypotheses posterior)))

;; =============================================================================
;; Generalization Curves
;; =============================================================================

(defn generalization-curve
  "Compute generalization probability for all numbers 1-100.
   Returns a map {number -> probability}."
  ([examples] (generalization-curve examples h/prior))
  ([examples prior]
   (let [post (infer/posterior examples prior)]
     (into {}
           (map (fn [y] [y (p-in-concept-from-posterior post y)]))
           h/domain))))

(defn generalization-vector
  "Compute generalization probability for all numbers 1-100.
   Returns a vector of 100 probabilities (index 0 = number 1)."
  ([examples] (generalization-vector examples h/prior))
  ([examples prior]
   (let [post (infer/posterior examples prior)]
     (mapv #(p-in-concept-from-posterior post %) (range 1 101)))))

;; =============================================================================
;; Analysis Utilities
;; =============================================================================

(defn most-likely-members
  "Return numbers most likely to be in the concept.
   Returns seq of [number probability] pairs, sorted by probability."
  ([examples] (most-likely-members examples 20))
  ([examples top-k] (most-likely-members examples top-k h/prior))
  ([examples top-k prior]
   (->> (generalization-curve examples prior)
        (sort-by val >)
        (take top-k))))

(defn generalization-stats
  "Compute statistics about the generalization distribution."
  ([examples] (generalization-stats examples h/prior))
  ([examples prior]
   (let [curve (generalization-curve examples prior)
         probs (vals curve)
         sorted-probs (sort > probs)
         above-threshold (fn [t] (count (filter #(> % t) probs)))]
     {:examples examples
      :max-prob (apply max probs)
      :min-prob (apply min probs)
      :mean-prob (/ (reduce + probs) (count probs))
      :above-50% (above-threshold 0.5)
      :above-10% (above-threshold 0.1)
      :above-1% (above-threshold 0.01)
      :effective-size (reduce + probs)})))

;; =============================================================================
;; Comparison with Human Data
;; =============================================================================

(defn compare-to-targets
  "Compute generalization probabilities for specific test numbers.
   Useful for comparing to human behavioral data."
  ([examples test-numbers] (compare-to-targets examples test-numbers h/prior))
  ([examples test-numbers prior]
   (let [post (infer/posterior examples prior)]
     (mapv (fn [y]
             {:number y
              :probability (p-in-concept-from-posterior post y)})
           test-numbers))))

;; =============================================================================
;; Visualization Helpers
;; =============================================================================

(defn curve-to-ascii
  "Create a simple ASCII visualization of generalization curve.
   Returns a string."
  ([examples] (curve-to-ascii examples 50))
  ([examples width]
   (let [curve (generalization-vector examples)
         max-p (apply max curve)
         scale (if (zero? max-p) 1 (/ width max-p))]
     (clojure.string/join
      "\n"
      (for [row (partition 10 (map-indexed vector curve))]
        (str (format "%3d-%3d: " (inc (* 10 (quot (ffirst row) 10)))
                     (+ 10 (* 10 (quot (ffirst row) 10))))
             (clojure.string/join
              ""
              (for [[_ p] row]
                (let [bar-len (int (* p scale))]
                  (apply str (repeat bar-len "#")))))))))))

(defn print-curve
  "Print a formatted generalization curve to stdout."
  ([examples] (print-curve examples h/prior))
  ([examples prior]
   (let [curve (generalization-curve examples prior)
         sorted (sort-by key curve)]
     (println "Generalization curve for examples:" examples)
     (println (str "Number\tP(in concept)"))
     (println (str "------\t-------------"))
     (doseq [[n p] sorted]
       (when (> p 0.001)
         (println (format "%3d\t%.4f %s"
                          n p
                          (apply str (repeat (int (* p 50)) "*")))))))))

;; =============================================================================
;; Sequential Generalization
;; =============================================================================

(defn sequential-generalization
  "Compute generalization curves after each example in sequence.
   Returns vector of generalization curves."
  ([examples] (sequential-generalization examples h/prior))
  ([examples prior]
   (let [posteriors (infer/sequential-posteriors examples prior)]
     (mapv (fn [post]
             (mapv #(p-in-concept-from-posterior post %) (range 1 101)))
           (rest posteriors)))))  ; skip the prior (before any examples)
