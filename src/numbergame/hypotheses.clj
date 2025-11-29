(ns numbergame.hypotheses
  "Hypothesis space for the Number Game.

   Hypotheses are maps with:
     :id      - unique identifier (keyword or vector)
     :members - set of integers in the concept

   Two types of hypotheses:
     1. Rule-based: mathematical patterns (powers of 2, primes, multiples, etc.)
     2. Interval-based: contiguous ranges [a, b]")

;; =============================================================================
;; Domain
;; =============================================================================

(def domain
  "The universe of possible numbers: 1 to 100."
  (set (range 1 101)))

(def domain-size (count domain))

;; =============================================================================
;; Helper functions
;; =============================================================================

(defn prime?
  "Returns true if n is prime."
  [n]
  (and (> n 1)
       (not-any? #(zero? (rem n %))
                 (range 2 (inc (Math/sqrt n))))))

(defn perfect-square?
  "Returns true if n is a perfect square."
  [n]
  (let [root (Math/sqrt n)]
    (== root (Math/floor root))))

(defn ends-with?
  "Returns true if n ends with digit d."
  [n d]
  (= (rem n 10) d))

;; =============================================================================
;; Rule-based hypotheses
;; =============================================================================

(defn make-hypothesis
  "Create a hypothesis map from an id and a predicate over domain."
  [id pred]
  {:id id
   :members (into #{} (filter pred) domain)})

(defn multiples-of
  "Hypothesis: all multiples of k in domain."
  [k]
  (make-hypothesis
   (keyword (str "multiples-of-" k))
   #(zero? (rem % k))))

(def rule-hypotheses
  "Collection of rule-based hypotheses."
  (concat
   ;; Multiples
   (map multiples-of (range 2 13))

   ;; Mathematical sequences
   [{:id :powers-of-2
     :members #{1 2 4 8 16 32 64}}

    {:id :powers-of-3
     :members #{1 3 9 27 81}}

    {:id :powers-of-4
     :members #{1 4 16 64}}

    {:id :squares
     :members (into #{} (filter perfect-square?) domain)}

    {:id :primes
     :members (into #{} (filter prime?) domain)}

    {:id :odd
     :members (into #{} (filter odd?) domain)}

    {:id :even
     :members (into #{} (filter even?) domain)}]

   ;; Numbers ending in specific digit
   (for [d (range 10)]
     {:id (keyword (str "ends-in-" d))
      :members (into #{} (filter #(ends-with? % d)) domain)})))

;; =============================================================================
;; Interval-based hypotheses
;; =============================================================================

(defn interval-hypothesis
  "Create an interval hypothesis [a, b]."
  [a b]
  {:id [:interval a b]
   :members (set (range a (inc b)))})

(def interval-hypotheses
  "All possible intervals [a, b] where 1 <= a <= b <= 100.
   This is (100 * 101) / 2 = 5050 hypotheses."
  (for [a (range 1 101)
        b (range a 101)]
    (interval-hypothesis a b)))

;; =============================================================================
;; Combined hypothesis space
;; =============================================================================

(def all-hypotheses
  "Vector of all hypotheses: rules + intervals.
   Indexed for efficient lookup."
  (vec (concat rule-hypotheses interval-hypotheses)))

(def hypothesis-count
  "Total number of hypotheses in the space."
  (count all-hypotheses))

(defn hypothesis-by-id
  "Find a hypothesis by its id."
  [id]
  (first (filter #(= (:id %) id) all-hypotheses)))

(defn hypotheses-containing
  "Return all hypotheses that contain the given number."
  [n]
  (filter #(contains? (:members %) n) all-hypotheses))

(defn hypotheses-containing-all
  "Return all hypotheses that contain all given examples."
  [examples]
  (let [example-set (set examples)]
    (filter #(every? (:members %) example-set) all-hypotheses)))

;; =============================================================================
;; Priors
;; =============================================================================

(def uniform-prior
  "Uniform prior: equal probability for all hypotheses."
  (let [p (/ 1.0 hypothesis-count)]
    (vec (repeat hypothesis-count p))))

(defn rule-biased-prior
  "Prior that gives more weight to rule hypotheses.
   rule-weight: relative weight for rule hypotheses (default 10x)."
  ([] (rule-biased-prior 10.0))
  ([rule-weight]
   (let [n-rules (count rule-hypotheses)
         n-intervals (count interval-hypotheses)
         rule-total (* n-rules rule-weight)
         interval-total (* n-intervals 1.0)
         z (+ rule-total interval-total)
         rule-p (/ rule-weight z)
         interval-p (/ 1.0 z)]
     (vec (concat
           (repeat n-rules rule-p)
           (repeat n-intervals interval-p))))))

(defn size-biased-prior
  "Prior that penalizes very large and very small hypotheses.
   Uses a log-normal-like weighting on hypothesis size."
  ([] (size-biased-prior 20.0 1.0))
  ([preferred-size spread]
   (let [weights (mapv (fn [{:keys [members]}]
                         (let [size (count members)
                               log-size (Math/log (max 1 size))
                               log-pref (Math/log preferred-size)
                               diff (- log-size log-pref)]
                           (Math/exp (- (/ (* diff diff) (* 2 spread spread))))))
                       all-hypotheses)
         z (reduce + weights)]
     (mapv #(/ % z) weights))))

;; Default prior
(def prior uniform-prior)
