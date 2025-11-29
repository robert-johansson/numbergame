(ns numbergame.sequential
  "Sequential hypothesis filtering for the Number Game.

   Instead of computing likelihoods for all 5,078 hypotheses,
   we prune impossible hypotheses as each example arrives.

   This is:
   - Faster: only compute likelihoods for consistent hypotheses
   - More cognitively plausible: humans likely prune as they observe
   - More interpretable: watch the hypothesis space shrink"
  (:require [numbergame.hypotheses :as h]
            [clojure.set :as set]))

;; =============================================================================
;; Pre-computed Index: number → hypothesis indices
;; =============================================================================

(def hypotheses-by-number
  "Map from number (1-100) to set of hypothesis indices containing that number.
   Computed once at load time for O(1) lookup."
  (into {}
    (for [n h/domain]
      [n (set (keep-indexed
                (fn [idx hyp]
                  (when (contains? (:members hyp) n) idx))
                h/all-hypotheses))])))

(defn hypotheses-containing
  "Return set of hypothesis indices containing number n."
  [n]
  (get hypotheses-by-number n #{}))

;; =============================================================================
;; Sequential Filtering
;; =============================================================================

(def all-indices
  "Set of all hypothesis indices."
  (set (range h/hypothesis-count)))

(defn filter-step
  "Given current set of hypothesis indices, filter by new example.
   Returns the intersection: hypotheses consistent with all seen examples."
  [current-indices new-example]
  (set/intersection current-indices (hypotheses-containing new-example)))

(defn sequential-filter
  "Process examples one by one, returning sequence of remaining hypothesis sets.
   First element is all hypotheses, then after each example."
  [examples]
  (reductions filter-step all-indices examples))

(defn consistent-hypotheses
  "Return set of hypothesis indices consistent with all examples."
  [examples]
  (reduce filter-step all-indices examples))

(defn consistent-hypothesis-count
  "Count hypotheses consistent with examples."
  [examples]
  (count (consistent-hypotheses examples)))

;; =============================================================================
;; Efficient Posterior (only over consistent hypotheses)
;; =============================================================================

(defn posterior-efficient
  "Compute posterior only over hypotheses that are consistent with examples.
   Much faster than computing over all 5,078 hypotheses."
  ([examples] (posterior-efficient examples h/prior))
  ([examples prior]
   (let [remaining (consistent-hypotheses examples)
         n (count examples)]

     (if (empty? remaining)
       ;; No consistent hypotheses
       {:error "No consistent hypotheses"
        :examples examples}

       ;; Compute log-weights only for consistent hypotheses
       (let [log-weights
             (for [idx remaining]
               (let [hyp (nth h/all-hypotheses idx)
                     size (count (:members hyp))
                     log-prior (Math/log (nth prior idx))
                     log-like (- (* n (Math/log size)))]
                 {:idx idx
                  :hypothesis hyp
                  :log-weight (+ log-prior log-like)}))

             ;; Normalize using log-sum-exp
             max-lw (apply max (map :log-weight log-weights))
             with-weights (map #(assoc % :weight
                                  (Math/exp (- (:log-weight %) max-lw)))
                               log-weights)
             total (reduce + (map :weight with-weights))]

         (->> with-weights
              (map #(assoc % :prob (/ (:weight %) total)))
              (sort-by :prob >)))))))

(defn top-hypotheses-efficient
  "Return top-k hypotheses using efficient filtering."
  ([examples] (top-hypotheses-efficient examples 10))
  ([examples k] (top-hypotheses-efficient examples k h/prior))
  ([examples k prior]
   (->> (posterior-efficient examples prior)
        (take k)
        (mapv #(select-keys % [:hypothesis :prob])))))

;; =============================================================================
;; Learning Dynamics - Watch the pruning happen
;; =============================================================================

(defn pruning-trajectory
  "Return detailed trajectory of hypothesis pruning.
   Shows what remains and what's eliminated at each step."
  ([examples] (pruning-trajectory examples h/prior))
  ([examples prior]
   (let [steps (sequential-filter examples)
         n-examples (count examples)]

     {:examples examples
      :trajectory
      (vec
       (for [[i remaining] (map-indexed vector steps)]
         (let [prev-count (if (zero? i)
                           h/hypothesis-count
                           (count (nth steps (dec i))))
               curr-count (count remaining)
               eliminated (- prev-count curr-count)]

           (merge
            {:step i
             :remaining-count curr-count
             :eliminated eliminated
             :elimination-rate (if (zero? prev-count)
                                0.0
                                (double (/ eliminated prev-count)))}

            (when (pos? i)
              {:after-example (nth examples (dec i))})

            ;; Add top hypotheses for this step
            (when (and (pos? i) (<= i n-examples))
              (let [seen (take i examples)
                    post (posterior-efficient seen prior)]
                {:top-3 (vec (take 3 (map #(select-keys % [:hypothesis :prob])
                                          post)))}))))))

      :summary
      {:initial-hypotheses h/hypothesis-count
       :final-hypotheses (count (last steps))
       :total-eliminated (- h/hypothesis-count (count (last steps)))
       :efficiency-gain (/ (double h/hypothesis-count)
                          (max 1 (count (last steps))))}})))

(defn print-pruning
  "Print a nice visualization of the pruning process."
  ([examples] (print-pruning examples h/prior))
  ([examples prior]
   (let [{:keys [trajectory summary]} (pruning-trajectory examples prior)]

     (println "\nSequential Hypothesis Pruning")
     (println (apply str (repeat 60 "=")))
     (println)

     (doseq [{:keys [step remaining-count eliminated elimination-rate
                     after-example top-3]} trajectory]
       (if (zero? step)
         (println (format "Start:      %,d hypotheses" remaining-count))
         (do
           (println (format "After %-4s  %,d hypotheses (-%,d, %.0f%% eliminated)"
                           (str after-example ":")
                           remaining-count
                           eliminated
                           (* 100 elimination-rate)))
           (when top-3
             (doseq [{:keys [hypothesis prob]} top-3]
               (println (format "            → %-25s (%.3f)"
                               (:id hypothesis) prob)))))))

     (println)
     (println (apply str (repeat 60 "-")))
     (println (format "Efficiency: Only computed %d/%d likelihoods (%.1fx speedup)"
                     (:final-hypotheses summary)
                     (:initial-hypotheses summary)
                     (:efficiency-gain summary))))))

;; =============================================================================
;; Analysis: What gets eliminated?
;; =============================================================================

(defn eliminated-at-step
  "Return hypotheses eliminated when observing a particular example."
  [examples step-idx]
  (let [steps (vec (sequential-filter examples))
        before (if (zero? step-idx) all-indices (nth steps step-idx))
        after (nth steps (inc step-idx))]
    (set/difference before after)))

(defn sample-eliminated
  "Sample some hypotheses that were eliminated at a step (for inspection)."
  [examples step-idx n]
  (let [eliminated (eliminated-at-step examples step-idx)]
    (->> eliminated
         (take n)
         (mapv #(:id (nth h/all-hypotheses %))))))

;; =============================================================================
;; Optimal Questioning: Which example would be most informative?
;; =============================================================================

(defn expected-elimination
  "If we observed number n, how many hypotheses would be eliminated on average?
   (Assumes n is consistent with current hypothesis set.)"
  [current-indices n]
  (let [containing (hypotheses-containing n)
        would-remain (set/intersection current-indices containing)]
    (- (count current-indices) (count would-remain))))

(defn most-informative-examples
  "Given current consistent hypotheses, which numbers would eliminate the most?
   Returns numbers that ARE in at least one consistent hypothesis."
  [examples top-k]
  (let [remaining (consistent-hypotheses examples)
        ;; Only consider numbers that appear in at least one remaining hypothesis
        candidate-numbers (into #{}
                            (mapcat #(:members (nth h/all-hypotheses %)))
                            remaining)]
    (->> candidate-numbers
         (map (fn [n]
                {:number n
                 :would-eliminate (expected-elimination remaining n)}))
         (sort-by :would-eliminate >)
         (take top-k))))

;; =============================================================================
;; Comparison with brute-force
;; =============================================================================

(defn benchmark
  "Compare timing of efficient vs brute-force posterior computation."
  [examples]
  (let [;; Brute force (require inference namespace)
        t1 (System/nanoTime)
        _ (require '[numbergame.inference :as infer])
        bf-result ((resolve 'numbergame.inference/top-hypotheses) examples 5)
        t2 (System/nanoTime)
        bf-time (/ (- t2 t1) 1e6)

        ;; Efficient
        t3 (System/nanoTime)
        eff-result (top-hypotheses-efficient examples 5)
        t4 (System/nanoTime)
        eff-time (/ (- t4 t3) 1e6)]

    {:brute-force-ms bf-time
     :efficient-ms eff-time
     :speedup (/ bf-time (max 0.001 eff-time))
     :results-match? (= (map #(:id (:hypothesis %)) eff-result)
                        (map #(:id (first %)) bf-result))}))
