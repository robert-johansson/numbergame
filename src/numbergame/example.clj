(ns numbergame.example
  "A simple Gen.clj example: coin flipping with inference."
  (:require [gen.distribution.commons-math :as dist]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.generative-function :as gf]
            [gen.trace :as trace]
            [gen.choicemap :as choicemap]))

;; === Example 1: Simple coin flip model ===
;; A generative function that flips a biased coin n times

(def flip-coins
  (gen [n bias]
    (loop [i 0
           heads 0]
      (if (< i n)
        (let [flip (dynamic/trace! [:flip i] dist/bernoulli bias)]
          (recur (inc i) (if flip (inc heads) heads)))
        heads))))

;; === Example 2: A simple Bayesian model ===
;; Model: unknown bias, observe some flips, infer the bias
;; Prior: bias ~ Beta(1, 1) = Uniform(0, 1)
;; Likelihood: each flip ~ Bernoulli(bias)

(def coin-model
  (gen [n]
    (let [bias (dynamic/trace! :bias dist/beta 1.0 1.0)]
      (dotimes [i n]
        (dynamic/trace! [:flip i] dist/bernoulli bias))
      bias)))

;; === Helper functions ===

(defn run-example-1
  "Simulate flipping coins and show the trace."
  []
  (println "\n=== Example 1: Simulating coin flips ===")
  (let [tr (gf/simulate flip-coins [5 0.7])]
    (println "Arguments:" (trace/get-args tr))
    (println "Return value (heads count):" (trace/get-retval tr))
    (println "Choices:" (trace/get-choices tr))
    (println "Log probability:" (trace/get-score tr))))

(defn run-example-2
  "Simulate the Bayesian coin model."
  []
  (println "\n=== Example 2: Bayesian coin model (prior) ===")
  (let [tr (gf/simulate coin-model [5])]
    (println "Sampled bias:" (trace/get-retval tr))
    (println "Choices:" (trace/get-choices tr))
    (println "Log probability:" (trace/get-score tr))))

(defn run-example-3
  "Constrained generation: fix some observations and sample the rest."
  []
  (println "\n=== Example 3: Constrained generation ===")
  (println "Constraining flips 0,1,2 to be true (heads)...")
  (let [constraints {:bias 0.8  ; fix the bias
                     [:flip 0] true
                     [:flip 1] true
                     [:flip 2] true}
        result (gf/generate coin-model [5] constraints)
        tr (:trace result)]
    (println "Weight:" (:weight result))
    (println "Sampled bias:" (trace/get-retval tr))
    (println "Choices:" (trace/get-choices tr))))

(defn importance-sampling
  "Simple importance sampling to estimate posterior on bias given observations."
  [observations num-particles]
  (let [n (count observations)
        constraints (into {} (map-indexed (fn [i v] [[:flip i] v]) observations))
        particles (repeatedly num-particles
                              #(let [result (gf/generate coin-model [n] constraints)]
                                 {:trace (:trace result)
                                  :weight (:weight result)}))
        ;; Normalize weights (in log space, then exponentiate)
        log-weights (map :weight particles)
        max-log-w (apply max log-weights)
        weights (map #(Math/exp (- % max-log-w)) log-weights)
        total (reduce + weights)
        normalized (map #(/ % total) weights)
        ;; Compute weighted mean of bias
        biases (map #(trace/get-retval (:trace %)) particles)
        mean-bias (reduce + (map * biases normalized))]
    mean-bias))

(defn run-example-4
  "Importance sampling inference."
  []
  (println "\n=== Example 4: Importance sampling inference ===")
  (let [observations [true true true false true]  ; 4 heads, 1 tail
        num-particles 1000
        estimated-bias (importance-sampling observations num-particles)]
    (println "Observations:" observations)
    (println "Estimated bias (posterior mean):" estimated-bias)
    (println "(With 4/5 heads, we'd expect bias around 0.67)")))

(defn -main
  "Run all examples."
  [& _args]
  (run-example-1)
  (run-example-2)
  (run-example-3)
  (run-example-4)
  (println "\nDone!"))

(comment
  ;; Run in REPL:
  (run-example-1)
  (run-example-2)
  (run-example-3)
  (run-example-4)
  (-main)

  ;; Or run from command line:
  ;; clj -M -m numbergame.example
  )
