(ns numbergame.program-induction
  "Program induction for the Number Game.

   Instead of fixed hypotheses, we define a GRAMMAR of concepts.
   This allows compositional concepts like (union :prime (mult 10))
   that aren't in our fixed hypothesis set.

   Two approaches:
     Option A: Enumerate all programs to depth N, exact Bayes (efficient)
     Option B: Sample programs randomly, rejection sampling (inefficient)
   "
  (:require [clojure.set :as set]))

;; =============================================================================
;; Domain and Primitives
;; =============================================================================

(def domain (set (range 1 101)))

(defn prime? [n]
  (and (> n 1)
       (not-any? #(zero? (rem n %))
                 (range 2 (inc (Math/sqrt n))))))

(def primitives
  "Primitive concepts - base cases for the grammar."
  {:odd      (set (filter odd? domain))
   :even     (set (filter even? domain))
   :prime    (set (filter prime? domain))
   :square   #{1 4 9 16 25 36 49 64 81 100}
   :pow2     #{1 2 4 8 16 32 64}
   :pow3     #{1 3 9 27 81}
   :small    (set (range 1 26))      ; 1-25
   :medium   (set (range 26 76))     ; 26-75
   :large    (set (range 76 101))})  ; 76-100

(def primitive-names (vec (keys primitives)))
(def n-primitives (count primitive-names))

;; =============================================================================
;; Program Representation
;; =============================================================================

;; Programs are nested maps:
;;   {:type :primitive, :name :pow2}
;;   {:type :mult, :k 5}
;;   {:type :range, :lo 10, :hi 20}
;;   {:type :union, :left <program>, :right <program>}
;;   {:type :intersect, :left <program>, :right <program>}

(defn eval-program
  "Execute a program to get the set of numbers it defines."
  [prog]
  (case (:type prog)
    :primitive (get primitives (:name prog) #{})
    :mult      (set (filter #(zero? (rem % (:k prog))) domain))
    :range     (set (range (:lo prog) (inc (:hi prog))))
    :union     (set/union (eval-program (:left prog))
                          (eval-program (:right prog)))
    :intersect (set/intersection (eval-program (:left prog))
                                  (eval-program (:right prog)))
    ;; Default: empty set
    #{}))

(defn program->string
  "Pretty-print a program."
  [prog]
  (case (:type prog)
    :primitive (name (:name prog))
    :mult      (format "(mult %d)" (:k prog))
    :range     (format "(range %d %d)" (:lo prog) (:hi prog))
    :union     (format "(union %s %s)"
                       (program->string (:left prog))
                       (program->string (:right prog)))
    :intersect (format "(intersect %s %s)"
                       (program->string (:left prog))
                       (program->string (:right prog)))
    "?"))

;; =============================================================================
;; OPTION A: Enumerate All Programs (Exact Inference)
;; =============================================================================

(defn enumerate-terminals
  "Enumerate all terminal programs (no recursion needed)."
  []
  (concat
   ;; All primitives
   (for [name primitive-names]
     {:type :primitive :name name})

   ;; Mult with k from 2 to 12
   (for [k (range 2 13)]
     {:type :mult :k k})

   ;; Ranges - but limit to avoid explosion
   ;; Only include "reasonable" ranges (not all 5050)
   (for [lo [1 5 10 15 20 25 30 40 50 60 70 80 90]
         span [5 10 15 20 25 30 40 50]
         :let [hi (+ lo span)]
         :when (<= hi 100)]
     {:type :range :lo lo :hi hi})))

(defn enumerate-programs
  "Enumerate all programs up to given depth.

   Depth 0: terminals only
   Depth 1: terminals + (op terminal terminal)
   Depth 2: terminals + (op depth1 depth1)
   etc."
  [max-depth]
  (loop [depth 0
         programs (vec (enumerate-terminals))]
    (if (>= depth max-depth)
      programs
      ;; Add combinations at next depth
      (let [combiners
            (for [op [:union :intersect]
                  left programs
                  right programs
                  :when (not= left right)]  ; Avoid (union X X)
              {:type op :left left :right right})]
        (recur (inc depth)
               (into programs combiners))))))

(defn enumerate-programs-lazy
  "Lazily enumerate programs - useful for very large spaces.
   Returns terminals first, then depth-1 combinations, etc."
  [max-depth]
  (let [terminals (enumerate-terminals)]
    (if (zero? max-depth)
      terminals
      (concat terminals
              (for [op [:union :intersect]
                    left terminals
                    right terminals
                    :when (not= left right)]
                {:type op :left left :right right})))))

;; Pre-compute programs for quick access
(def enumerated-programs-depth-0
  "All terminal programs."
  (vec (enumerate-terminals)))

(def enumerated-programs-depth-1
  "All programs up to depth 1 (terminals + one level of combination)."
  (vec (enumerate-programs 1)))

(defn programs->hypotheses
  "Convert programs to hypothesis format compatible with our inference code."
  [programs]
  (vec
   (for [prog programs
         :let [members (eval-program prog)]
         :when (seq members)]  ; Skip empty concepts
     {:id (program->string prog)
      :members members
      :program prog})))

(def enumerated-hypotheses-depth-1
  "Programs as hypotheses for exact inference."
  (programs->hypotheses enumerated-programs-depth-1))

;; =============================================================================
;; Option A: Exact Bayesian Inference over Enumerated Programs
;; =============================================================================

(defn log-likelihood-program
  "Log likelihood under size principle: -n * log(|concept|)"
  [hypothesis examples]
  (let [members (:members hypothesis)]
    (if (every? members examples)
      (- (* (count examples) (Math/log (count members))))
      Double/NEGATIVE_INFINITY)))

(defn exact-posterior
  "Compute exact posterior over enumerated programs.
   Uses uniform prior and size principle likelihood."
  ([examples] (exact-posterior examples enumerated-hypotheses-depth-1))
  ([examples hypotheses]
   (let [n-hyp (count hypotheses)
         log-prior (Math/log (/ 1.0 n-hyp))

         ;; Compute log posterior (unnormalized)
         log-posts (mapv (fn [h]
                           (let [ll (log-likelihood-program h examples)]
                             (if (Double/isInfinite ll)
                               Double/NEGATIVE_INFINITY
                               (+ log-prior ll))))
                         hypotheses)

         ;; Normalize
         max-lp (apply max (filter #(not (Double/isInfinite %)) log-posts))
         weights (mapv (fn [lp]
                         (if (Double/isInfinite lp)
                           0.0
                           (Math/exp (- lp max-lp))))
                       log-posts)
         total (reduce + weights)]

     (if (zero? total)
       {:error "No consistent programs"
        :examples examples}
       {:examples examples
        :n-hypotheses n-hyp
        :n-consistent (count (filter pos? weights))
        :posterior (mapv (fn [h w]
                           (assoc h :prob (/ w total)))
                         hypotheses weights)}))))

(defn top-programs-exact
  "Get top-k programs by exact posterior probability."
  ([examples] (top-programs-exact examples 10))
  ([examples k] (top-programs-exact examples k enumerated-hypotheses-depth-1))
  ([examples k hypotheses]
   (let [result (exact-posterior examples hypotheses)]
     (if (:error result)
       result
       (->> (:posterior result)
            (filter #(pos? (:prob %)))
            (sort-by :prob >)
            (take k)
            (mapv #(select-keys % [:id :prob :members])))))))

;; =============================================================================
;; OPTION B: Sample Programs Randomly (Approximate Inference)
;; =============================================================================

;; NOTE: Gen.clj's recursive gen functions have issues with trace!
;; So we use pure Clojure for program sampling, which is actually cleaner.

(defn sample-program
  "Sample a program from the grammar using pure Clojure randomness.

   This is simpler and works better than fighting Gen.clj's recursion."
  ([] (sample-program 0))
  ([depth]
   (let [;; Deeper = more likely to terminate
         stop-prob (/ depth (+ depth 2.0))

         ;; Choose node type: 0-2 are terminals, 3-4 are recursive
         type-idx (if (or (>= depth 4)
                         (< (rand) stop-prob))
                    ;; Terminal: primitive, mult, or range
                    (rand-int 3)
                    ;; Recursive: union or intersect
                    (+ 3 (rand-int 2)))]

     (case type-idx
       ;; Primitive
       0 {:type :primitive
          :name (nth primitive-names (rand-int n-primitives))}

       ;; Mult
       1 {:type :mult
          :k (+ 2 (rand-int 11))}  ; 2-12

       ;; Range
       2 (let [lo (+ 1 (rand-int 100))
               hi (+ lo (rand-int (- 101 lo)))]
           {:type :range
            :lo lo
            :hi hi})

       ;; Union
       3 {:type :union
          :left (sample-program (inc depth))
          :right (sample-program (inc depth))}

       ;; Intersect
       4 {:type :intersect
          :left (sample-program (inc depth))
          :right (sample-program (inc depth))}))))

;; =============================================================================
;; Gen.clj: Full Generative Model
;; =============================================================================

(defn sample-program-and-examples
  "Sample a program and then examples from it.
   Pure Clojure - no Gen.clj needed for the basic version."
  [n-examples]
  (let [program (sample-program)
        members (eval-program program)
        members-vec (vec members)]
    (if (empty? members-vec)
      {:program program :members members :examples []}
      {:program program
       :members members
       :examples (vec (repeatedly n-examples #(rand-nth members-vec)))})))

;; =============================================================================
;; Inference: Importance Sampling (Rejection-style)
;; =============================================================================

(defn infer-programs
  "Infer programs that could have generated the observed examples.

   Uses rejection-style importance sampling:
   1. Sample many programs from the prior
   2. Keep only those consistent with examples
   3. Weight by size principle: |concept|^(-n)

   Returns top programs by posterior probability."
  [examples n-samples]
  (let [n (count examples)
        example-set (set examples)

        ;; Sample many programs and filter to consistent ones
        samples
        (loop [remaining n-samples
               results []]
          (if (zero? remaining)
            results
            (let [program (sample-program)
                  members (eval-program program)
                  consistent? (and (seq members)
                                   (every? members example-set))]
              (recur (dec remaining)
                     (if consistent?
                       (conj results {:program program
                                      :members members
                                      :size (count members)})
                       results)))))

        _ (when (empty? samples)
            (println "WARNING: No consistent programs found in" n-samples "samples"))]

    (if (empty? samples)
      {:n-samples n-samples
       :n-consistent 0
       :acceptance-rate 0.0
       :top-programs []}

      ;; Weight by size principle
      (let [log-weights (mapv (fn [{:keys [size]}]
                                (- (* n (Math/log (max 1 size)))))
                              samples)
            max-lw (apply max log-weights)
            weights (mapv #(Math/exp (- % max-lw)) log-weights)
            total (reduce + weights)
            weighted (mapv (fn [sample w]
                             (assoc sample :prob (/ w total)))
                           samples weights)]

        {:n-samples n-samples
         :n-consistent (count samples)
         :acceptance-rate (double (/ (count samples) n-samples))
         :top-programs (->> weighted
                            (sort-by :prob >)
                            (take 10)
                            (mapv #(select-keys % [:program :prob :size])))}))))

;; =============================================================================
;; Demo
;; =============================================================================

(defn show-enumeration-stats
  "Show statistics about enumerated program space."
  []
  (println "\nEnumerated Program Space:")
  (println (format "  Terminals (depth 0): %d programs" (count enumerated-programs-depth-0)))
  (println (format "  Depth 1: %d programs" (count enumerated-programs-depth-1)))
  (println (format "  As hypotheses: %d (excluding empty concepts)" (count enumerated-hypotheses-depth-1))))

(defn demo-option-a
  "Demonstrate Option A: Exact inference over enumerated programs."
  [examples]
  (println "\n" (apply str (repeat 60 "-")))
  (println "OPTION A: Exact Inference (Enumeration)")
  (println (apply str (repeat 60 "-")))

  (println "\nExamples:" examples)

  (let [t1 (System/nanoTime)
        result (exact-posterior examples)
        t2 (System/nanoTime)
        time-ms (/ (- t2 t1) 1e6)]

    (if (:error result)
      (println "Error:" (:error result))
      (do
        (println (format "\nSearched %d programs in %.2f ms" (:n-hypotheses result) time-ms))
        (println (format "Consistent: %d" (:n-consistent result)))
        (println "\nTop programs:")
        (doseq [h (->> (:posterior result)
                       (filter #(pos? (:prob %)))
                       (sort-by :prob >)
                       (take 5))]
          (println (format "  %-40s prob: %.4f  size: %3d"
                           (:id h)
                           (:prob h)
                           (count (:members h)))))))))

(defn demo-option-b
  "Demonstrate Option B: Sampling-based inference."
  [examples n-samples]
  (println "\n" (apply str (repeat 60 "-")))
  (println "OPTION B: Sampling-based Inference")
  (println (apply str (repeat 60 "-")))

  (println "\nExamples:" examples)
  (println "Sampling" n-samples "programs...")

  (let [t1 (System/nanoTime)
        result (infer-programs examples n-samples)
        t2 (System/nanoTime)
        time-ms (/ (- t2 t1) 1e6)]

    (println (format "\nSampled %d programs in %.2f ms" n-samples time-ms))
    (println (format "Consistent: %d (%.2f%% acceptance)"
                     (:n-consistent result)
                     (* 100.0 (:acceptance-rate result))))

    (if (empty? (:top-programs result))
      (println "No consistent programs found!")
      (do
        (println "\nTop programs:")
        (doseq [{:keys [program prob size]} (take 5 (:top-programs result))]
          (println (format "  %-40s prob: %.4f  size: %3d"
                           (program->string program)
                           prob
                           size)))))))

(defn demo-comparison
  "Compare Option A and Option B on the same examples."
  [examples n-samples]
  (println "\n" (apply str (repeat 60 "=")) "\n")
  (println "COMPARISON: Option A vs Option B")
  (println (apply str (repeat 60 "=")) "\n")

  (println "Examples:" examples)

  ;; Option A
  (let [t1 (System/nanoTime)
        result-a (top-programs-exact examples 5)
        t2 (System/nanoTime)
        time-a (/ (- t2 t1) 1e6)]

    (println (format "\nOption A (Enumeration): %.2f ms" time-a))
    (if (:error result-a)
      (println "  Error:" (:error result-a))
      (doseq [h result-a]
        (println (format "  %-35s prob: %.4f" (:id h) (:prob h)))))

    ;; Option B
    (let [t3 (System/nanoTime)
          result-b (infer-programs examples n-samples)
          t4 (System/nanoTime)
          time-b (/ (- t4 t3) 1e6)]

      (println (format "\nOption B (Sampling %d): %.2f ms" n-samples time-b))
      (println (format "  Acceptance: %.2f%%" (* 100 (:acceptance-rate result-b))))
      (doseq [{:keys [program prob]} (take 5 (:top-programs result-b))]
        (println (format "  %-35s prob: %.4f" (program->string program) prob)))

      (println (format "\nSpeedup: %.1fx" (/ time-b time-a))))))

(defn run-demo []
  (println "\n" (apply str (repeat 60 "*")))
  (println "  PROGRAM INDUCTION: Option A vs Option B")
  (println (apply str (repeat 60 "*")))

  ;; Show enumeration stats
  (show-enumeration-stats)

  ;; Compare on different stimuli
  (demo-comparison [16 8 2 64] 5000)
  (demo-comparison [16 17 18 19] 5000)
  (demo-comparison [7 11 13 17] 5000)

  (println "\n" (apply str (repeat 60 "=")) "\n")
  (println "SUMMARY")
  (println (apply str (repeat 60 "=")) "\n")
  (println "Option A (Enumeration):")
  (println "  + Exact inference")
  (println "  + Fast (searches all programs once)")
  (println "  + Deterministic results")
  (println "  - Limited to enumerable depth")
  (println)
  (println "Option B (Sampling):")
  (println "  + Can sample arbitrarily deep programs")
  (println "  - Approximate (misses rare programs)")
  (println "  - Slow (most samples rejected)")
  (println "  - Non-deterministic results"))
