(ns ex01-distributions
  "Example 1: Overview of probability distributions in Gen.clj

   Gen.clj provides primitive distributions that can be used directly
   as functions to sample, or as generative functions for tracing.

   Two distribution backends are available:
   - gen.distribution.commons-math (Apache Commons Math)
   - gen.distribution.kixi (Kixi Stats)

   Both provide the same interface."
  (:require [gen.distribution.commons-math :as dist]
            [gen.distribution :as d]))

;; =============================================================================
;; PART 1: Direct Sampling
;; =============================================================================
;; Distributions can be called directly as functions to get samples.

(defn demo-direct-sampling []
  (println "\n=== Direct Sampling from Distributions ===\n")

  ;; Bernoulli: coin flip (true/false with probability p)
  (println "Bernoulli (p=0.7):" (dist/bernoulli 0.7))

  ;; Normal/Gaussian: bell curve
  (println "Normal (mean=0, std=1):" (dist/normal 0 1))
  (println "Normal (mean=100, std=15):" (dist/normal 100 15))

  ;; Uniform: equal probability in range
  (println "Uniform (0 to 1):" (dist/uniform 0 1))
  (println "Uniform (10 to 20):" (dist/uniform 10 20))

  ;; Uniform discrete: integers in range [low, high] inclusive
  (println "Uniform-discrete (1 to 6):" (dist/uniform-discrete 1 6))

  ;; Beta: values in [0,1], useful for probabilities
  (println "Beta (alpha=2, beta=5):" (dist/beta 2 5))

  ;; Gamma: positive real values, useful for rates/scales
  (println "Gamma (shape=2, scale=1):" (dist/gamma 2 1))

  ;; Categorical: sample index from probability vector
  (println "Categorical [0.1 0.2 0.7]:" (dist/categorical [0.1 0.2 0.7])))

;; =============================================================================
;; PART 2: The Distribution Protocols
;; =============================================================================
;; Distributions implement two protocols: Sample and LogPDF

(defn demo-protocols []
  (println "\n=== Distribution Protocols ===\n")

  ;; Create a distribution object
  (let [normal-dist (dist/normal-distribution 0 1)]

    ;; Sample from it
    (println "Sample from N(0,1):" (d/sample normal-dist))

    ;; Compute log probability density
    (println "logpdf of 0.0:" (d/logpdf normal-dist 0.0))
    (println "logpdf of 1.0:" (d/logpdf normal-dist 1.0))
    (println "logpdf of 3.0:" (d/logpdf normal-dist 3.0))

    ;; The logpdf tells us how likely a value is under this distribution
    ;; Higher values = more likely
    (println "\nProbability of 0.0:" (Math/exp (d/logpdf normal-dist 0.0)))
    (println "Probability of 3.0:" (Math/exp (d/logpdf normal-dist 3.0)))))

;; =============================================================================
;; PART 3: Categorical Distributions
;; =============================================================================
;; Categorical can work with vectors or maps

(defn demo-categorical []
  (println "\n=== Categorical Distributions ===\n")

  ;; With a vector: returns index 0, 1, 2, ...
  (let [probs [0.5 0.3 0.2]]
    (println "Categorical with vector [0.5 0.3 0.2]:")
    (println "  10 samples:" (repeatedly 10 #(dist/categorical probs))))

  ;; With a map: returns keys
  (let [probs {:heads 0.6 :tails 0.4}]
    (println "\nCategorical with map {:heads 0.6 :tails 0.4}:")
    (println "  10 samples:" (repeatedly 10 #(dist/categorical probs)))))

;; =============================================================================
;; PART 4: Sampling Multiple Values
;; =============================================================================

(defn demo-multiple-samples []
  (println "\n=== Generating Multiple Samples ===\n")

  ;; Generate many samples to see distribution shape
  (let [samples (repeatedly 1000 #(dist/normal 0 1))
        mean (/ (reduce + samples) (count samples))
        sorted (sort samples)
        median (nth sorted 500)]
    (println "1000 samples from N(0,1):")
    (println "  Sample mean:" mean)
    (println "  Sample median:" median)
    (println "  Min:" (first sorted))
    (println "  Max:" (last sorted)))

  ;; Estimate probability via sampling
  (let [samples (repeatedly 10000 #(dist/bernoulli 0.3))
        num-true (count (filter true? samples))]
    (println "\nBernoulli(0.3) - 10000 samples:")
    (println "  Fraction true:" (/ num-true 10000.0))))

;; =============================================================================
;; PART 5: Distribution Summary
;; =============================================================================

(defn demo-all-distributions []
  (println "\n=== All Available Distributions ===\n")

  (println "Discrete distributions:")
  (println "  bernoulli p         - true/false with P(true)=p")
  (println "  binomial n p        - number of successes in n trials")
  (println "  uniform-discrete a b - integer uniformly in [a,b]")
  (println "  categorical probs   - sample index/key from probability vector/map")

  (println "\nContinuous distributions:")
  (println "  normal mu sigma     - Gaussian with mean mu, std sigma")
  (println "  uniform a b         - uniform on [a,b]")
  (println "  beta alpha beta     - values in [0,1], shape params")
  (println "  gamma shape scale   - positive values, shape/scale params")
  (println "  exponential rate    - exponential with given rate")
  (println "  student-t nu [loc scale] - Student's t distribution"))

;; =============================================================================
;; Main
;; =============================================================================

(defn -main [& _]
  (demo-direct-sampling)
  (demo-protocols)
  (demo-categorical)
  (demo-multiple-samples)
  (demo-all-distributions)
  (println "\nDone!"))

(comment
  ;; Run in REPL:
  (-main)

  ;; Or individual demos:
  (demo-direct-sampling)
  (demo-protocols)

  ;; Explore distributions interactively:
  (dist/normal 0 1)
  (dist/bernoulli 0.5)
  (repeatedly 10 #(dist/uniform-discrete 1 6))
  )
