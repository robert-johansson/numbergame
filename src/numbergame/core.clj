(ns numbergame.core
  "Main entry point for the Number Game implementation.

   The Number Game (Tenenbaum, 1999) is a concept learning task where:
   - A hidden concept is a subset of integers 1-100
   - Examples are shown from the concept
   - The learner must generalize to new numbers

   This implementation demonstrates:
   - Analytic Bayesian inference with the size principle
   - Gen.clj probabilistic programming
   - Generalization predictions"
  (:require [numbergame.hypotheses :as h]
            [numbergame.inference :as infer]
            [numbergame.generalization :as gen]
            [numbergame.model :as model]))

;; =============================================================================
;; Classic Number Game Stimuli (from Tenenbaum's experiments)
;; =============================================================================

(def classic-stimuli
  "Classic stimuli from Tenenbaum's number game experiments."
  {:powers-of-2     [16 8 2 64]
   :single-16       [16]
   :interval-like   [16 23 19 20]
   :multiples-of-10 [60 80 10 30]
   :single-60       [60]
   :primes          [7 3 11 17]})

;; =============================================================================
;; Demo Functions
;; =============================================================================

(defn demo-hypothesis-space
  "Demonstrate the hypothesis space."
  []
  (println "\n" (apply str (repeat 70 "=")) "\n")
  (println "HYPOTHESIS SPACE")
  (println (apply str (repeat 70 "=")) "\n")

  (println "Domain: integers 1 to 100")
  (println (format "Total hypotheses: %,d" h/hypothesis-count))
  (println (format "  - Rule-based: %d" (count h/rule-hypotheses)))
  (println (format "  - Interval-based: %,d" (count h/interval-hypotheses)))

  (println "\nSample rule hypotheses:")
  (doseq [hyp (take 8 h/rule-hypotheses)]
    (println (format "  %-20s size: %3d  members: %s"
                     (:id hyp)
                     (count (:members hyp))
                     (if (> (count (:members hyp)) 10)
                       (str (vec (take 8 (sort (:members hyp)))) "...")
                       (vec (sort (:members hyp)))))))

  (println "\nSample interval hypotheses:")
  (doseq [hyp (take 5 (drop 10 h/interval-hypotheses))]
    (println (format "  %-20s size: %3d"
                     (:id hyp)
                     (count (:members hyp))))))

(defn demo-size-principle
  "Demonstrate the size principle in action."
  []
  (println "\n" (apply str (repeat 70 "=")) "\n")
  (println "THE SIZE PRINCIPLE")
  (println (apply str (repeat 70 "=")) "\n")

  (println "The size principle: smaller consistent hypotheses get more support.")
  (println "Likelihood: P(examples | h) = |h|^(-n) if consistent, else 0\n")

  (let [examples [16 8 2 64]
        small-h (h/hypothesis-by-id :powers-of-2)
        large-h (h/hypothesis-by-id :even)]

    (println "Examples:" examples)
    (println)
    (println "Comparing two consistent hypotheses:")
    (println (format "  %-15s size: %3d  log-likelihood: %.3f"
                     (:id small-h)
                     (count (:members small-h))
                     (infer/log-likelihood small-h examples)))
    (println (format "  %-15s size: %3d  log-likelihood: %.3f"
                     (:id large-h)
                     (count (:members large-h))
                     (infer/log-likelihood large-h examples)))
    (println)
    (println "The smaller hypothesis (powers-of-2) has MUCH higher likelihood!")
    (println "This is why specific rules dominate when examples fit them perfectly.")))

(defn demo-posterior
  "Demonstrate posterior inference."
  []
  (println "\n" (apply str (repeat 70 "=")) "\n")
  (println "POSTERIOR INFERENCE")
  (println (apply str (repeat 70 "=")) "\n")

  (doseq [[name examples] [[:powers-of-2 [16 8 2 64]]
                           [:single-16 [16]]
                           [:interval-like [16 23 19 20]]]]
    (println "Examples:" examples)
    (println "Top hypotheses by posterior probability:")
    (doseq [[hyp prob] (infer/top-hypotheses examples 5)]
      (println (format "  %-25s prob: %.4f  size: %3d"
                       (:id hyp) prob (count (:members hyp)))))
    (println)))

(defn demo-generalization
  "Demonstrate generalization predictions."
  []
  (println "\n" (apply str (repeat 70 "=")) "\n")
  (println "GENERALIZATION PREDICTIONS")
  (println (apply str (repeat 70 "=")) "\n")

  (println "P(y in concept | examples) - probability that test number y")
  (println "belongs to the same concept as the examples.\n")

  (let [test-numbers [2 4 8 16 20 32 50 64]]
    (doseq [[name examples] [[:powers-of-2 [16 8 2 64]]
                             [:interval-like [16 23 19 20]]]]
      (println (str "Examples: " examples))
      (println "Test number predictions:")
      (let [preds (gen/compare-to-targets examples test-numbers)]
        (doseq [{:keys [number probability]} preds]
          (let [bar (apply str (repeat (int (* probability 40)) "*"))]
            (println (format "  %3d: %.4f %s" number probability bar)))))
      (println))))

(defn demo-sequential-learning
  "Demonstrate how beliefs update with each new example."
  []
  (println "\n" (apply str (repeat 70 "=")) "\n")
  (println "SEQUENTIAL BELIEF UPDATING")
  (println (apply str (repeat 70 "=")) "\n")

  (let [examples [16 8 2 64]]
    (println "Showing top hypothesis after each example:")
    (println)
    (loop [seen []
           remaining examples]
      (when (seq remaining)
        (let [new-example (first remaining)
              seen' (conj seen new-example)
              top (first (infer/top-hypotheses seen' 1))]
          (println (format "After seeing %s:"
                           (vec seen')))
          (println (format "  Top: %-25s (prob: %.4f)"
                           (:id (first top)) (second top)))
          (println)
          (recur seen' (rest remaining)))))))

(defn demo-gen-clj-model
  "Demonstrate the Gen.clj generative model."
  []
  (println "\n" (apply str (repeat 70 "=")) "\n")
  (println "GEN.CLJ GENERATIVE MODEL")
  (println (apply str (repeat 70 "=")) "\n")

  (println "The Gen.clj model lets us:")
  (println "  - Simulate synthetic data from the generative process")
  (println "  - Use probabilistic inference tools")
  (println "  - Embed in larger models\n")

  (println "Simulating 5 concept/example pairs:")
  (dotimes [i 5]
    (let [{:keys [hypothesis examples]} (model/simulate-with-hypothesis 4)]
      (println (format "  %d. %-20s -> %s"
                       (inc i) (:id hypothesis) examples))))

  (println "\nValidating model consistency...")
  (model/validate-model))

(defn demo-comparison
  "Compare predictions across different stimuli."
  []
  (println "\n" (apply str (repeat 70 "=")) "\n")
  (println "COMPARING DIFFERENT STIMULI")
  (println (apply str (repeat 70 "=")) "\n")

  (let [test-numbers [4 8 16 32 64 17 18 19 20 21 22 23 50]]
    (println "How do generalization patterns differ?\n")
    (println (format "%-20s %s" "Test Number" (vec test-numbers)))
    (println (apply str (repeat 70 "-")))

    (doseq [[name examples] classic-stimuli]
      (let [preds (mapv #(:probability %)
                        (gen/compare-to-targets examples test-numbers))]
        (println (format "%-20s %s"
                         name
                         (mapv #(format "%.2f" %) preds))))))

  (println "\nKey observations:")
  (println "  - [16 8 2 64] strongly predicts other powers of 2")
  (println "  - [16 23 19 20] predicts numbers in that range")
  (println "  - Single examples [16] spread probability more broadly"))

;; =============================================================================
;; Main
;; =============================================================================

(defn run-all-demos
  "Run all demonstrations."
  []
  (println "\n")
  (println (apply str (repeat 70 "*")))
  (println "       THE NUMBER GAME: BAYESIAN CONCEPT LEARNING")
  (println "       (Tenenbaum, 1999)")
  (println (apply str (repeat 70 "*")))

  (demo-hypothesis-space)
  (demo-size-principle)
  (demo-posterior)
  (demo-generalization)
  (demo-sequential-learning)
  (demo-gen-clj-model)
  (demo-comparison)

  (println "\n" (apply str (repeat 70 "=")) "\n")
  (println "DEMO COMPLETE")
  (println (apply str (repeat 70 "=")) "\n"))

(defn -main
  "Main entry point."
  [& args]
  (run-all-demos))
