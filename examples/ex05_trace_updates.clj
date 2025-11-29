(ns ex05-trace-updates
  "Example 5: Trace Updates

   Trace updates allow you to modify an existing trace by:
   - Changing constrained values
   - Re-running parts of the model

   This is the foundation for MCMC algorithms like Metropolis-Hastings,
   where you propose changes to traces and accept/reject based on weights.

   Key function: trace/update"
  (:require [clojure.math :as math]
            [gen.distribution.commons-math :as dist]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.generative-function :as gf]
            [gen.trace :as trace]
            [gen.choicemap :as choicemap]))

;; =============================================================================
;; PART 1: Basic Trace Update
;; =============================================================================

(def simple-model
  (gen []
    (let [x (dynamic/trace! :x dist/normal 0 1)
          y (dynamic/trace! :y dist/normal x 1)]
      {:x x :y y})))

(defn demo-basic-update []
  (println "\n=== Basic Trace Update ===\n")

  ;; Get an initial trace
  (let [initial-trace (gf/simulate simple-model [])
        initial-choices (trace/get-choices initial-trace)]

    (println "Initial trace:")
    (println "  x =" (:x initial-choices))
    (println "  y =" (:y initial-choices))
    (println "  score =" (trace/get-score initial-trace))

    ;; Update: change the value of :x
    (let [new-constraints {:x 2.0}
          update-result (trace/update initial-trace new-constraints)
          new-trace (:trace update-result)
          new-choices (trace/get-choices new-trace)]

      (println "\nAfter update with {:x 2.0}:")
      (println "  x =" (:x new-choices))
      (println "  y =" (:y new-choices) "(re-sampled given new x)")
      (println "  score =" (trace/get-score new-trace))
      (println "  weight =" (:weight update-result))
      (println "  discard =" (:discard update-result) "(old values)"))))

;; =============================================================================
;; PART 2: Understanding the Weight
;; =============================================================================

(defn demo-weight []
  (println "\n=== Understanding Update Weight ===\n")

  ;; The weight from update is:
  ;; log( p(new trace) / p(old trace) * q(old | new) / q(new | old) )
  ;;
  ;; For deterministic updates (just changing a value):
  ;; weight = log p(new trace) - log p(old trace)

  (let [;; Start with a known trace
        initial (gf/simulate simple-model [])
        initial-x (get (trace/get-choices initial) :x)
        initial-score (trace/get-score initial)

        ;; Update x to a specific value
        update-result (trace/update initial {:x 0.0})
        new-trace (:trace update-result)
        new-score (trace/get-score new-trace)
        weight (:weight update-result)]

    (println "Initial x:" initial-x)
    (println "Initial score:" initial-score)
    (println "\nUpdated x: 0.0")
    (println "New score:" new-score)
    (println "Weight:" weight)
    (println "\nThe weight reflects the change in probability")))

;; =============================================================================
;; PART 3: The Discard
;; =============================================================================

(defn demo-discard []
  (println "\n=== The Discard Choice Map ===\n")

  (let [initial (gf/simulate simple-model [])
        initial-choices (trace/get-choices initial)

        update-result (trace/update initial {:x 5.0})
        discard (:discard update-result)]

    (println "Initial choices:" initial-choices)
    (println "Update constraints: {:x 5.0}")
    (println "Discard:" discard)
    (println "\nThe discard contains the OLD value of :x")
    (println "This is useful for reversible MCMC moves"))

  ;; You can use discard to reverse an update
  (println "\n--- Reversibility ---")
  (let [initial (gf/simulate simple-model [])
        initial-choices (trace/get-choices initial)

        ;; Forward update
        forward (trace/update initial {:x 5.0})
        forward-trace (:trace forward)
        forward-discard (:discard forward)

        ;; Reverse update (use discard as constraints)
        reverse (trace/update forward-trace forward-discard)
        reverse-trace (:trace reverse)
        reverse-choices (trace/get-choices reverse-trace)]

    (println "Initial x:" (:x initial-choices))
    (println "After forward update x:" (get (trace/get-choices forward-trace) :x))
    (println "Discard from forward:" forward-discard)
    (println "After reverse update x:" (:x reverse-choices))
    (println "(Should match initial x)")))

;; =============================================================================
;; PART 4: Simple Metropolis-Hastings
;; =============================================================================

(def target-model
  (gen []
    (let [x (dynamic/trace! :x dist/normal 0 1)]
      (dynamic/trace! :y dist/normal 0 1)  ; observed
      x)))

(defn mh-step
  "Single Metropolis-Hastings step using random walk proposal on :x"
  [current-trace]
  (let [current-x (choicemap/get-value (trace/get-choices current-trace) :x)
        ;; Propose new x by adding noise (symmetric proposal)
        proposed-x (+ current-x (dist/normal 0 0.5))
        ;; Update trace with proposed value
        update-result (trace/update current-trace {:x proposed-x})
        weight (:weight update-result)
        ;; Accept/reject
        accept-prob (min 1.0 (math/exp weight))]
    (if (< (rand) accept-prob)
      (:trace update-result)  ; accept
      current-trace)))        ; reject

(defn run-mh
  "Run MH for n steps"
  [initial-trace n]
  (loop [trace initial-trace
         samples []
         i 0]
    (if (< i n)
      (let [new-trace (mh-step trace)
            x (choicemap/get-value (trace/get-choices new-trace) :x)]
        (recur new-trace (conj samples x) (inc i)))
      samples)))

(defn demo-mh []
  (println "\n=== Simple Metropolis-Hastings ===\n")

  ;; Set up: we observe y=2.0, want to infer x
  ;; Under the model: x ~ N(0,1), y ~ N(0,1)
  ;; So posterior on x given y is still N(0,1) (they're independent)

  (let [;; Get initial trace with y constrained
        initial-result (gf/generate target-model [] {:y 2.0})
        initial-trace (:trace initial-result)

        ;; Run MH
        samples (run-mh initial-trace 1000)

        ;; Compute statistics
        mean (/ (reduce + samples) (count samples))
        variance (/ (reduce + (map #(math/pow (- % mean) 2) samples))
                    (count samples))]

    (println "Observed y = 2.0")
    (println "Running 1000 MH steps...")
    (println "\nPosterior on x (should be ~N(0,1) since x and y independent):")
    (println "  Sample mean:" mean "(expected: 0)")
    (println "  Sample variance:" variance "(expected: 1)")
    (println "  First 20 samples:" (take 20 samples))))

;; =============================================================================
;; PART 5: Updates with Structural Changes
;; =============================================================================

(def conditional-model
  (gen [threshold]
    (let [x (dynamic/trace! :x dist/normal 0 1)]
      (if (> x threshold)
        {:x x :extra (dynamic/trace! :extra dist/normal 0 1)}
        {:x x}))))

(defn demo-structural []
  (println "\n=== Updates with Structural Changes ===\n")

  ;; This model has different structure depending on x
  ;; Updates handle this automatically

  ;; Start with x below threshold (no :extra)
  (let [initial-result (gf/generate conditional-model [0] {:x -1.0})
        initial-trace (:trace initial-result)
        initial-choices (trace/get-choices initial-trace)]

    (println "Initial (x=-1, threshold=0):")
    (println "  Choices:" initial-choices)

    ;; Update to x above threshold (now :extra appears)
    (let [update-result (trace/update initial-trace {:x 1.0})
          new-trace (:trace update-result)
          new-choices (trace/get-choices new-trace)]

      (println "\nAfter update (x=1.0):")
      (println "  Choices:" new-choices)
      (println "  (Note: :extra was sampled since x > threshold)")
      (println "  Discard:" (:discard update-result)))))

;; =============================================================================
;; PART 6: gf/assess - Computing Probability of Choices
;; =============================================================================

(defn demo-assess []
  (println "\n=== gf/assess - Computing Log Probability ===\n")

  ;; assess computes the log probability of a complete set of choices
  ;; without generating a full trace

  (let [model (gen []
                (let [x (dynamic/trace! :x dist/normal 0 1)]
                  (dynamic/trace! :y dist/normal x 0.5)))

        choices {:x 0.0 :y 0.0}
        result (gf/assess model [] choices)]

    (println "Model: x ~ N(0,1), y ~ N(x, 0.5)")
    (println "Choices:" choices)
    (println "Result:" result)
    (println "\nLog probability:" (:weight result))
    (println "This is logpdf(x=0|N(0,1)) + logpdf(y=0|N(0,0.5))")))

;; =============================================================================
;; Main
;; =============================================================================

(defn -main [& _]
  (demo-basic-update)
  (demo-weight)
  (demo-discard)
  (demo-mh)
  (demo-structural)
  (demo-assess)
  (println "\nDone!"))

(comment
  (-main)

  ;; Explore interactively
  (let [tr (gf/simulate simple-model [])]
    (trace/update tr {:x 0.0}))
  )
