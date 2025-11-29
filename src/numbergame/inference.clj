(ns numbergame.inference
  "Analytic Bayesian inference for the Number Game.

   Implements the size principle:
     p(examples | h) = |h|^(-n) if all examples in h, else 0

   Uses log-space computation for numerical stability."
  (:require [numbergame.hypotheses :as h]))

;; =============================================================================
;; Size Principle Likelihood
;; =============================================================================

(def ^:const neg-inf
  "Negative infinity for log-space."
  Double/NEGATIVE_INFINITY)

(defn log-likelihood
  "Compute log p(examples | hypothesis) under the size principle.

   If all examples are in the hypothesis, returns -n * log(|h|).
   Otherwise returns negative infinity."
  [{:keys [members]} examples]
  (if (every? members examples)
    (let [n (count examples)
          size (count members)]
      (- (* n (Math/log size))))
    neg-inf))

(defn consistent?
  "Returns true if hypothesis is consistent with all examples."
  [{:keys [members]} examples]
  (every? members examples))

;; =============================================================================
;; Posterior Computation
;; =============================================================================

(defn log-sum-exp
  "Numerically stable computation of log(sum(exp(xs))).
   Uses the max-shift trick to avoid overflow/underflow."
  [xs]
  (if (empty? xs)
    neg-inf
    (let [max-x (apply max xs)]
      (if (= max-x neg-inf)
        neg-inf
        (+ max-x (Math/log (transduce
                           (map #(if (= % neg-inf) 0.0 (Math/exp (- % max-x))))
                           + xs)))))))

(defn posterior-log-weights
  "Compute unnormalized log posterior weights for all hypotheses.
   Returns vector of log(p(h) * p(examples|h))."
  ([examples] (posterior-log-weights examples h/prior))
  ([examples prior]
   (mapv (fn [hyp p-h]
           (let [ll (log-likelihood hyp examples)]
             (if (= ll neg-inf)
               neg-inf
               (+ (Math/log p-h) ll))))
         h/all-hypotheses
         prior)))

(defn normalize-log-weights
  "Convert log weights to normalized probabilities.
   Returns vector of probabilities summing to 1."
  [log-weights]
  (let [log-z (log-sum-exp log-weights)]
    (if (= log-z neg-inf)
      ;; No hypothesis fits - return uniform over all
      (vec (repeat (count log-weights) (/ 1.0 (count log-weights))))
      (mapv #(if (= % neg-inf)
               0.0
               (Math/exp (- % log-z)))
            log-weights))))

(defn posterior
  "Compute posterior distribution over hypotheses given examples.

   Returns a vector of probabilities, one per hypothesis,
   in the same order as h/all-hypotheses."
  ([examples] (posterior examples h/prior))
  ([examples prior]
   (-> (posterior-log-weights examples prior)
       normalize-log-weights)))

;; =============================================================================
;; Posterior Analysis
;; =============================================================================

(defn top-hypotheses
  "Return the top-k hypotheses by posterior probability.
   Returns seq of [hypothesis probability] pairs."
  ([examples] (top-hypotheses examples 10))
  ([examples k] (top-hypotheses examples k h/prior))
  ([examples k prior]
   (let [post (posterior examples prior)]
     (->> (map vector h/all-hypotheses post)
          (sort-by second >)
          (take k)))))

(defn posterior-entropy
  "Compute entropy of posterior distribution (in nats)."
  [post]
  (- (transduce
      (comp (filter pos?)
            (map #(* % (Math/log %))))
      + post)))

(defn posterior-summary
  "Return a summary of the posterior distribution."
  ([examples] (posterior-summary examples h/prior))
  ([examples prior]
   (let [post (posterior examples prior)
         consistent (h/hypotheses-containing-all examples)]
     {:examples examples
      :n-examples (count examples)
      :n-consistent (count consistent)
      :entropy (posterior-entropy post)
      :top-5 (mapv (fn [[hyp p]] {:id (:id hyp) :size (count (:members hyp)) :prob p})
                   (top-hypotheses examples 5 prior))})))

;; =============================================================================
;; Sequential Updates
;; =============================================================================

(defn update-posterior
  "Update a posterior distribution with a new example.
   Takes the current posterior (vector of probs) and returns new posterior."
  [current-posterior new-example]
  (let [log-weights (mapv (fn [hyp p-h]
                            (if (zero? p-h)
                              neg-inf
                              (let [ll (log-likelihood hyp [new-example])]
                                (if (= ll neg-inf)
                                  neg-inf
                                  (+ (Math/log p-h) ll)))))
                          h/all-hypotheses
                          current-posterior)]
    (normalize-log-weights log-weights)))

(defn sequential-posteriors
  "Compute posterior after each example in sequence.
   Returns vector of posteriors, one after each cumulative observation."
  ([examples] (sequential-posteriors examples h/prior))
  ([examples prior]
   (reductions update-posterior prior examples)))

;; =============================================================================
;; Marginal Likelihood
;; =============================================================================

(defn log-marginal-likelihood
  "Compute log p(examples) = log sum_h p(h) p(examples|h).
   This is the evidence or marginal likelihood."
  ([examples] (log-marginal-likelihood examples h/prior))
  ([examples prior]
   (log-sum-exp (posterior-log-weights examples prior))))
