(ns ex02-generative-functions
  "Example 2: Generative Functions and Traces

   Generative functions are the core abstraction in Gen.clj.
   They are probabilistic programs that can:
   - Make random choices at named addresses
   - Be simulated to produce traces
   - Be constrained to match observations (for inference)

   The `gen` macro creates generative functions from Clojure code."
  (:require [gen.distribution.commons-math :as dist]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.generative-function :as gf]
            [gen.trace :as trace]))

;; =============================================================================
;; PART 1: Defining Generative Functions
;; =============================================================================

;; A generative function is defined with the `gen` macro.
;; Inside, use `dynamic/trace!` to make random choices at named addresses.

(def simple-model
  (gen []
    ;; trace! takes: address, distribution, and distribution args
    (let [x (dynamic/trace! :x dist/normal 0 1)]
      (* x x))))

;; You can call it directly like a regular function (untraced):
(defn demo-direct-call []
  (println "\n=== Direct Call (Untraced) ===\n")
  (println "Calling simple-model directly:")
  (dotimes [_ 5]
    (println "  Result:" (simple-model))))

;; =============================================================================
;; PART 2: Simulating and Getting Traces
;; =============================================================================

;; To capture the random choices, use gf/simulate

(defn demo-simulate []
  (println "\n=== Simulating with Traces ===\n")

  (let [tr (gf/simulate simple-model [])]
    (println "Trace contents:")
    (println "  Arguments:" (trace/get-args tr))
    (println "  Return value:" (trace/get-retval tr))
    (println "  Choices:" (trace/get-choices tr))
    (println "  Log probability:" (trace/get-score tr))))

;; =============================================================================
;; PART 3: Multi-step Models
;; =============================================================================

;; Models can have multiple traced choices

(def coin-flip-model
  (gen [n bias]
    (loop [i 0
           results []]
      (if (< i n)
        (let [flip (dynamic/trace! [:flip i] dist/bernoulli bias)]
          (recur (inc i) (conj results flip)))
        results))))

(defn demo-multi-choice []
  (println "\n=== Models with Multiple Choices ===\n")

  (let [tr (gf/simulate coin-flip-model [5 0.6])]
    (println "Coin flip model (5 flips, bias=0.6):")
    (println "  Arguments:" (trace/get-args tr))
    (println "  Results:" (trace/get-retval tr))
    (println "  All choices:" (trace/get-choices tr))
    (println "  Log probability:" (trace/get-score tr))))

;; =============================================================================
;; PART 4: Conditional Models (If/When)
;; =============================================================================

;; Models can have conditional structure - different paths make different choices

(def conditional-model
  (gen [threshold]
    (let [x (dynamic/trace! :x dist/normal 0 1)]
      (if (> x threshold)
        ;; This branch makes an additional choice
        (let [y (dynamic/trace! :y dist/normal x 0.5)]
          {:x x :y y :above true})
        ;; This branch does not
        {:x x :above false}))))

(defn demo-conditional []
  (println "\n=== Conditional Structure ===\n")

  (println "Running conditional-model with threshold=0...")
  (dotimes [_ 3]
    (let [tr (gf/simulate conditional-model [0])]
      (println "  Result:" (trace/get-retval tr))
      (println "  Choices:" (trace/get-choices tr))
      (println))))

;; =============================================================================
;; PART 5: Calling Other Generative Functions
;; =============================================================================

;; Gen functions can call other gen functions using trace! or splice!

(def inner-model
  (gen [scale]
    (dynamic/trace! :value dist/normal 0 scale)))

;; Method 1: Using trace! with an address (hierarchical addressing)
(def outer-with-trace
  (gen []
    (let [a (dynamic/trace! :first inner-model 1.0)
          b (dynamic/trace! :second inner-model 2.0)]
      (+ a b))))

;; Method 2: Using splice! (flat addressing - choices imported directly)
(def outer-with-splice
  (gen []
    ;; Note: splice! doesn't take an address
    ;; The inner choices appear at their original addresses
    (dynamic/splice! inner-model 1.0)))

(defn demo-composition []
  (println "\n=== Composing Generative Functions ===\n")

  (println "Using trace! (hierarchical addresses):")
  (let [tr (gf/simulate outer-with-trace [])]
    (println "  Choices:" (trace/get-choices tr))
    ;; Access nested choice with get-in
    (println "  [:first :value]:" (get-in (trace/get-choices tr) [:first :value])))

  (println "\nUsing splice! (flat addresses):")
  (let [tr (gf/simulate outer-with-splice [])]
    (println "  Choices:" (trace/get-choices tr))))

;; =============================================================================
;; PART 6: A More Realistic Model - Linear Regression
;; =============================================================================

(def linear-regression
  (gen [xs]
    (let [;; Prior on parameters
          slope (dynamic/trace! :slope dist/normal 0 2)
          intercept (dynamic/trace! :intercept dist/normal 0 5)
          noise (dynamic/trace! :noise dist/gamma 1 1)]
      ;; Generate y values
      (doseq [[i x] (map-indexed vector xs)]
        (let [mean (+ (* slope x) intercept)]
          (dynamic/trace! [:y i] dist/normal mean noise)))
      ;; Return the line function
      (fn [x] (+ (* slope x) intercept)))))

(defn demo-linear-regression []
  (println "\n=== Linear Regression Model ===\n")

  (let [xs [1 2 3 4 5]
        tr (gf/simulate linear-regression [xs])
        choices (trace/get-choices tr)]
    (println "X values:" xs)
    (println "Slope:" (:slope choices))
    (println "Intercept:" (:intercept choices))
    (println "Noise:" (:noise choices))
    (println "Generated Y values:"
             (mapv #(get choices [:y %]) (range (count xs))))
    (println "Log probability:" (trace/get-score tr))))

;; =============================================================================
;; PART 7: Understanding Log Probability (Score)
;; =============================================================================

(defn demo-score []
  (println "\n=== Understanding Log Probability ===\n")

  ;; The score is the sum of log probabilities of all choices
  (let [tr (gf/simulate simple-model [])]
    (println "Simple model trace:")
    (println "  x =" (get (trace/get-choices tr) :x))
    (println "  score =" (trace/get-score tr))
    (println "  (This is logpdf of x under N(0,1))"))

  ;; For multiple choices, scores add (probabilities multiply)
  (let [tr (gf/simulate coin-flip-model [3 0.5])]
    (println "\nCoin flip model (3 fair coins):")
    (println "  Choices:" (trace/get-choices tr))
    (println "  Score:" (trace/get-score tr))
    (println "  (Should be 3 * log(0.5) =" (* 3 (Math/log 0.5)) ")")))

;; =============================================================================
;; Main
;; =============================================================================

(defn -main [& _]
  (demo-direct-call)
  (demo-simulate)
  (demo-multi-choice)
  (demo-conditional)
  (demo-composition)
  (demo-linear-regression)
  (demo-score)
  (println "\nDone!"))

(comment
  ;; Run in REPL:
  (-main)

  ;; Explore interactively:
  (simple-model)
  (gf/simulate simple-model [])
  (trace/get-choices (gf/simulate coin-flip-model [5 0.7]))
  )
