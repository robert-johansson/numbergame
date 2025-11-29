(ns ex06-advanced
  "Example 6: Advanced Gen.clj Features

   This example covers lesser-known but powerful features:
   - dynamic/untraced: suppress tracing
   - gf/assess: compute log probability without trace
   - gf/propose: sample + return choices and weight
   - Direct trace access as a function
   - Array conversion for gradient-based methods"
  (:require [clojure.math :as math]
            [gen.distribution.commons-math :as dist]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.generative-function :as gf]
            [gen.trace :as trace]
            [gen.choicemap :as choicemap]
            [gen.array :as arr]))

;; =============================================================================
;; PART 1: dynamic/untraced - Suppressing Tracing
;; =============================================================================

;; Sometimes you want to call a generative function without tracing its choices.
;; This is useful for:
;; - Helper computations that shouldn't be part of the model
;; - Deterministic preprocessing

(def helper-model
  (gen []
    (dynamic/trace! :x dist/normal 0 1)))

(def outer-model
  (gen []
    ;; This call IS traced - :helper namespace contains :x
    (let [traced-result (dynamic/trace! :helper helper-model)]
      ;; This call is NOT traced - choices are discarded
      (let [untraced-result (dynamic/untraced (helper-model))]
        {:traced traced-result
         :untraced untraced-result}))))

(defn demo-untraced []
  (println "\n=== dynamic/untraced ===\n")

  (let [tr (gf/simulate outer-model [])
        choices (trace/get-choices tr)]
    (println "Return value:" (trace/get-retval tr))
    (println "Choices:" choices)
    (println "\nNote: Only :helper/:x appears, not the untraced call")))

;; =============================================================================
;; PART 2: gf/assess - Computing Log Probability
;; =============================================================================

;; assess computes the log probability of a set of choices without
;; creating a full trace. Useful when you just need the probability.

(def simple-model
  (gen []
    (let [mu (dynamic/trace! :mu dist/normal 0 10)
          sigma (dynamic/trace! :sigma dist/gamma 1 1)]
      (dynamic/trace! :obs dist/normal mu sigma)
      {:mu mu :sigma sigma})))

(defn demo-assess []
  (println "\n=== gf/assess ===\n")

  ;; Specify complete choices
  (let [choices {:mu 0.0 :sigma 1.0 :obs 0.5}
        result (gf/assess simple-model [] choices)]
    (println "Choices:" choices)
    (println "Result:" result)
    (println "Log probability:" (:weight result))
    (println "Return value:" (:retval result)))

  ;; Compare different choice configurations
  (println "\nComparing configurations:")
  (let [configs [{:mu 0.0 :sigma 1.0 :obs 0.0}    ; obs at mean
                 {:mu 0.0 :sigma 1.0 :obs 3.0}    ; obs 3 std away
                 {:mu 0.0 :sigma 0.1 :obs 0.0}]]  ; tighter sigma, obs at mean
    (doseq [c configs]
      (let [result (gf/assess simple-model [] c)]
        (println "  " c "-> logp =" (:weight result))))))

;; =============================================================================
;; PART 3: gf/propose - Sampling with Choices and Weight
;; =============================================================================

;; propose samples from the model and returns:
;; - :choices - the sampled choice map
;; - :weight - the log probability of those choices
;; - :retval - the return value

;; This is useful for:
;; - Custom proposal distributions in MCMC
;; - Variational inference

(defn demo-propose []
  (println "\n=== gf/propose ===\n")

  ;; Note: gf/propose is primarily used for primitive distributions
  ;; For dynamic gen functions, the implementation may have limitations
  (let [result (gf/propose dist/normal [0 1])]
    (println "Propose on dist/normal [0 1]:")
    (println "  Result keys:" (keys result))
    (println "  Choices:" (:choices result))
    (println "  Weight:" (:weight result))
    (println "  Return value:" (:retval result))

    ;; For primitives, choices is a Choice wrapping the value
    (println "\nThe weight equals log p(sample):")
    (println "  Sample:" (choicemap/get-value (:choices result)))
    (println "  Weight:" (:weight result))))

;; =============================================================================
;; PART 4: Direct Trace Access
;; =============================================================================

;; Traces implement IFn, so you can use them directly as functions
;; to access choices at addresses.

(def data-model
  (gen [n]
    (let [mean (dynamic/trace! :mean dist/normal 0 10)]
      (dotimes [i n]
        (dynamic/trace! [:data i] dist/normal mean 1))
      mean)))

(defn demo-trace-access []
  (println "\n=== Trace and Choice Map Access ===\n")

  (let [tr (gf/simulate data-model [3])
        choices (trace/get-choices tr)]
    (println "Trace type:" (type tr))
    (println "Choices type:" (type choices))

    ;; Access via trace/get-choices + get
    (println "\nVia get on choices:")
    (println "  (get choices :mean) =" (get choices :mean))
    (println "  (get choices [:data 0]) =" (get choices [:data 0]))

    ;; Choice map as function
    (println "\nChoice map as function:")
    (println "  (choices :mean) =" (choices :mean))
    (println "  (choices [:data 0]) =" (choices [:data 0]))

    ;; Keyword lookup on choice map
    (println "\nKeyword on choice map:")
    (println "  (:mean choices) =" (:mean choices))

    ;; Getting the actual value (not the Choice wrapper)
    (println "\nExtracting raw values with choicemap/get-value:")
    (println "  (choicemap/get-value choices :mean) =" (choicemap/get-value choices :mean))
    (println "  (choicemap/get-value choices [:data 0]) =" (choicemap/get-value choices [:data 0]))))

;; =============================================================================
;; PART 5: Array Conversion (for Gradients)
;; =============================================================================

;; Choice maps can be converted to flat arrays and back.
;; This is useful for gradient-based optimization.

(defn demo-array []
  (println "\n=== Array Conversion ===\n")

  ;; Create a choice map
  (let [cm (choicemap/choicemap {:a 1.0 :b 2.0 :c {:nested 3.0}})]
    (println "Original choice map:" cm)

    ;; Convert to array
    (let [arr (arr/to-array cm)]
      (println "As array:" arr)

      ;; Modify the array
      (let [modified-arr (mapv #(* 2 %) arr)]
        (println "Modified array:" modified-arr)

        ;; Convert back (requires template with same structure)
        (let [reconstructed (arr/from-array cm modified-arr)]
          (println "Reconstructed:" reconstructed))))))

;; =============================================================================
;; PART 6: Using assess for Model Comparison
;; =============================================================================

;; A practical use of assess: comparing which model better explains data

(def linear-model
  (gen [xs]
    (let [slope (dynamic/trace! :slope dist/normal 0 2)
          intercept (dynamic/trace! :intercept dist/normal 0 5)]
      (doseq [[i x] (map-indexed vector xs)]
        (dynamic/trace! [:y i] dist/normal (+ (* slope x) intercept) 0.5)))))

(def quadratic-model
  (gen [xs]
    (let [a (dynamic/trace! :a dist/normal 0 1)
          b (dynamic/trace! :b dist/normal 0 1)
          c (dynamic/trace! :c dist/normal 0 5)]
      (doseq [[i x] (map-indexed vector xs)]
        (dynamic/trace! [:y i] dist/normal (+ (* a x x) (* b x) c) 0.5)))))

(defn estimate-marginal-likelihood
  "Estimate marginal likelihood via importance sampling."
  [model args observations n-samples]
  (let [log-weights (repeatedly n-samples
                                #(:weight (gf/generate model args observations)))
        max-w (apply max log-weights)
        weights (map #(math/exp (- % max-w)) log-weights)]
    (+ max-w (math/log (/ (reduce + weights) n-samples)))))

(defn demo-model-comparison []
  (println "\n=== Model Comparison with assess ===\n")

  ;; Generate data from linear model
  (let [xs [0 1 2 3 4]
        ;; True linear relationship: y = 2x + 1 + noise
        ys [0.8 3.1 5.2 6.9 9.1]
        observations (into {} (map-indexed (fn [i y] [[:y i] y]) ys))]

    (println "Data: xs =" xs)
    (println "      ys =" ys)
    (println "(Generated from y = 2x + 1 + noise)")

    ;; Compare models
    (let [linear-ml (estimate-marginal-likelihood linear-model [xs] observations 1000)
          quad-ml (estimate-marginal-likelihood quadratic-model [xs] observations 1000)]

      (println "\nLog marginal likelihoods:")
      (println "  Linear model:" linear-ml)
      (println "  Quadratic model:" quad-ml)
      (println "\nBayes factor (linear/quadratic):" (math/exp (- linear-ml quad-ml)))
      (println "(>1 means linear is favored)"))))

;; =============================================================================
;; Main
;; =============================================================================

(defn -main [& _]
  (demo-untraced)
  (demo-assess)
  (demo-propose)
  (demo-trace-access)
  (demo-array)
  (demo-model-comparison)
  (println "\nDone!"))

(comment
  (-main)

  ;; Individual demos
  (demo-untraced)
  (demo-assess)
  (demo-propose)
  )
