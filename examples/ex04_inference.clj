(ns ex04-inference
  "Example 4: Inference with Gen.clj

   Inference is the process of reasoning backwards from observations
   to the latent variables that could have generated them.

   Key concepts:
   - gf/generate: run model with constraints, get importance weight
   - Importance sampling: sample many traces, weight by likelihood
   - gen.inference.importance/resampling: built-in importance resampling"
  (:require [clojure.math :as math]
            [gen.distribution.commons-math :as dist]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.generative-function :as gf]
            [gen.trace :as trace]
            [gen.choicemap :as choicemap]
            [gen.inference.importance :as importance]))

;; =============================================================================
;; PART 1: The Problem - Forward vs Inverse
;; =============================================================================

;; Forward: Given parameters, generate data
(def forward-model
  (gen [true-mean]
    (let [obs (dynamic/trace! :obs dist/normal true-mean 1.0)]
      obs)))

(defn demo-forward []
  (println "\n=== Forward Sampling (Easy) ===\n")
  (println "Given mean=5, sample observations:")
  (dotimes [_ 5]
    (println "  " (forward-model 5))))

;; Inverse: Given data, infer parameters - this is INFERENCE!

;; =============================================================================
;; PART 2: Using gf/generate for Constrained Generation
;; =============================================================================

;; Model where we want to infer the mean from observations
(def inference-model
  (gen [n]
    (let [mean (dynamic/trace! :mean dist/normal 0 10)  ; prior on mean
          std (dynamic/trace! :std dist/gamma 1 1)]     ; prior on std
      (dotimes [i n]
        (dynamic/trace! [:obs i] dist/normal mean std))
      {:mean mean :std std})))

(defn demo-generate []
  (println "\n=== gf/generate with Constraints ===\n")

  ;; Our observed data
  (let [observations [4.8 5.2 4.9 5.1 5.0]
        n (count observations)

        ;; Build constraints from observations
        constraints (reduce (fn [cm [i y]]
                              (assoc cm [:obs i] y))
                            {}
                            (map-indexed vector observations))]

    (println "Observations:" observations)
    (println "Constraints:" constraints)

    ;; Generate with constraints
    (let [result (gf/generate inference-model [n] constraints)
          tr (:trace result)
          weight (:weight result)]
      (println "\nGenerated trace:")
      (println "  Mean:" (get (trace/get-choices tr) :mean))
      (println "  Std:" (get (trace/get-choices tr) :std))
      (println "  Weight (log):" weight)
      (println "  Weight (linear):" (math/exp weight)))))

;; =============================================================================
;; PART 3: Simple Importance Sampling
;; =============================================================================

(defn simple-importance-sampling
  "Basic importance sampling: generate many constrained traces,
   return one sampled proportional to weight."
  [model args constraints num-samples]
  (let [;; Generate samples with weights
        samples (repeatedly num-samples
                            #(let [result (gf/generate model args constraints)]
                               {:trace (:trace result)
                                :log-weight (:weight result)}))
        ;; Convert to linear weights
        log-weights (map :log-weight samples)
        max-log-w (apply max log-weights)
        weights (map #(math/exp (- % max-log-w)) log-weights)
        total (reduce + weights)
        normalized (map #(/ % total) weights)
        ;; Sample one trace proportional to weight
        idx (dist/categorical (vec normalized))]
    (:trace (nth samples idx))))

(defn demo-simple-importance []
  (println "\n=== Simple Importance Sampling ===\n")

  (let [observations [4.8 5.2 4.9 5.1 5.0]
        n (count observations)
        constraints (reduce (fn [cm [i y]]
                              (assoc cm [:obs i] y))
                            {}
                            (map-indexed vector observations))]

    (println "Observations:" observations)
    (println "True mean should be around 5.0\n")

    ;; Run inference multiple times
    (println "Inferred means (10 runs, 100 samples each):")
    (let [means (repeatedly 10
                            #(let [tr (simple-importance-sampling
                                        inference-model [n] constraints 100)]
                               (choicemap/get-value (trace/get-choices tr) :mean)))]
      (doseq [m means]
        (println "  " m))
      (println "\nAverage inferred mean:" (/ (reduce + means) (count means))))))

;; =============================================================================
;; PART 4: Built-in Importance Resampling
;; =============================================================================

(defn demo-builtin-importance []
  (println "\n=== Built-in Importance Resampling ===\n")

  (let [observations [4.8 5.2 4.9 5.1 5.0]
        n (count observations)
        constraints (reduce (fn [cm [i y]]
                              (assoc cm [:obs i] y))
                            {}
                            (map-indexed vector observations))]

    (println "Using gen.inference.importance/resampling\n")

    ;; Run inference
    (let [result (importance/resampling inference-model [n] constraints 1000)
          tr (:trace result)
          log-ml (:weight result)]
      (println "Inferred parameters:")
      (println "  Mean:" (get (trace/get-choices tr) :mean))
      (println "  Std:" (get (trace/get-choices tr) :std))
      (println "\nLog marginal likelihood estimate:" log-ml))))

;; =============================================================================
;; PART 5: Bayesian Linear Regression
;; =============================================================================

(def linear-model
  (gen [xs]
    (let [slope (dynamic/trace! :slope dist/normal 0 2)
          intercept (dynamic/trace! :intercept dist/normal 0 5)
          noise (dynamic/trace! :noise dist/gamma 1 1)]
      (doseq [[i x] (map-indexed vector xs)]
        (let [y-mean (+ (* slope x) intercept)]
          (dynamic/trace! [:y i] dist/normal y-mean noise)))
      {:slope slope :intercept intercept})))

(defn demo-linear-regression []
  (println "\n=== Bayesian Linear Regression ===\n")

  ;; True parameters
  (let [true-slope 2.0
        true-intercept 1.0
        true-noise 0.5

        ;; Generate synthetic data
        xs [0 1 2 3 4 5]
        ys (mapv (fn [x]
                   (+ (* true-slope x) true-intercept
                      (dist/normal 0 true-noise)))
                 xs)

        ;; Build constraints
        constraints (reduce (fn [cm [i y]]
                              (assoc cm [:y i] y))
                            {}
                            (map-indexed vector ys))]

    (println "True parameters: slope=" true-slope ", intercept=" true-intercept)
    (println "X values:" xs)
    (println "Y values:" ys)

    ;; Run inference
    (println "\nRunning importance sampling (1000 particles)...")
    (let [result (importance/resampling linear-model [xs] constraints 1000)
          tr (:trace result)
          choices (trace/get-choices tr)]
      (println "\nInferred parameters:")
      (println "  Slope:" (:slope choices))
      (println "  Intercept:" (:intercept choices))
      (println "  Noise:" (:noise choices)))

    ;; Multiple runs to see variance
    (println "\n10 inference runs:")
    (let [results (repeatedly 10
                              #(let [result (importance/resampling
                                              linear-model [xs] constraints 500)
                                     choices (trace/get-choices (:trace result))]
                                 {:slope (choicemap/get-value choices :slope)
                                  :intercept (choicemap/get-value choices :intercept)}))]
      (println "  Slopes:" (mapv :slope results))
      (println "  Mean slope:" (/ (reduce + (map :slope results)) 10)))))

;; =============================================================================
;; PART 6: Model Comparison
;; =============================================================================

(def model-a
  (gen []
    (let [x (dynamic/trace! :x dist/normal 0 1)]
      (dynamic/trace! :y dist/normal x 0.5))))

(def model-b
  (gen []
    (let [x (dynamic/trace! :x dist/normal 5 1)]
      (dynamic/trace! :y dist/normal x 0.5))))

(defn demo-model-comparison []
  (println "\n=== Model Comparison via Marginal Likelihood ===\n")

  (let [;; Observed data
        observations {:y 4.5}

        ;; Run importance sampling for each model
        result-a (importance/resampling model-a [] observations 1000)
        result-b (importance/resampling model-b [] observations 1000)]

    (println "Observation: y = 4.5")
    (println "\nModel A: x ~ N(0,1), y ~ N(x, 0.5)")
    (println "  Log marginal likelihood:" (:weight result-a))

    (println "\nModel B: x ~ N(5,1), y ~ N(x, 0.5)")
    (println "  Log marginal likelihood:" (:weight result-b))

    (let [log-bf (- (:weight result-b) (:weight result-a))]
      (println "\nLog Bayes Factor (B vs A):" log-bf)
      (println "Bayes Factor:" (math/exp log-bf))
      (println "(>1 means B is favored, <1 means A is favored)"))))

;; =============================================================================
;; Main
;; =============================================================================

(defn -main [& _]
  (demo-forward)
  (demo-generate)
  (demo-simple-importance)
  (demo-builtin-importance)
  (demo-linear-regression)
  (demo-model-comparison)
  (println "\nDone!"))

(comment
  (-main)

  ;; Explore interactively
  (importance/resampling inference-model [5]
                         {[:obs 0] 5.0 [:obs 1] 4.8 [:obs 2] 5.2
                          [:obs 3] 4.9 [:obs 4] 5.1}
                         1000)
  )
