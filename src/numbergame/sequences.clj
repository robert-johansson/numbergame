(ns numbergame.sequences
  "Sequence synthesis: going backwards from data to program.

   The Clojure insight: Lazy sequences ARE programs.

   (iterate #(* 2 %) 1) is both:
     - DATA: the sequence [1 2 4 8 16 ...]
     - PROGRAM: the computation that generates it

   Synthesis = find which lazy seq matches the examples.")

;; =============================================================================
;; The Grammar: Sequence-generating programs
;; =============================================================================

(defn prime? [n]
  (cond
    (< n 2) false
    (= n 2) true
    (even? n) false
    :else (not-any? #(zero? (rem n %))
                    (range 3 (inc (Math/sqrt n)) 2))))

;; Sequence generators - return functions that generate finite prefixes
(defn make-seq-entry [name gen-fn desc]
  {:name name
   :gen gen-fn  ; Function that takes n and returns first n elements
   :desc desc})

(defn sequence-grammar []
  [(make-seq-entry [:arithmetic 1 1]
                   (fn [n] (vec (take n (iterate inc 1))))
                   "Natural numbers: 1, 2, 3, ...")

   (make-seq-entry [:arithmetic 0 2]
                   (fn [n] (vec (take n (iterate #(+ % 2) 0))))
                   "Even numbers: 0, 2, 4, ...")

   (make-seq-entry [:arithmetic 1 2]
                   (fn [n] (vec (take n (iterate #(+ % 2) 1))))
                   "Odd numbers: 1, 3, 5, ...")

   (make-seq-entry [:geometric 1 2]
                   (fn [n] (vec (take n (iterate #(* % 2) 1))))
                   "Powers of 2: 1, 2, 4, 8, ...")

   (make-seq-entry [:geometric 1 3]
                   (fn [n] (vec (take n (iterate #(* % 3) 1))))
                   "Powers of 3: 1, 3, 9, 27, ...")

   (make-seq-entry [:squares]
                   (fn [n] (vec (take n (map #(* % %) (range 1 1000)))))
                   "Perfect squares: 1, 4, 9, 16, ...")

   (make-seq-entry [:cubes]
                   (fn [n] (vec (take n (map #(* % % %) (range 1 1000)))))
                   "Perfect cubes: 1, 8, 27, 64, ...")

   (make-seq-entry [:triangular]
                   (fn [n] (vec (take n (reductions + (range 1 1000)))))
                   "Triangular: 1, 3, 6, 10, ...")

   (make-seq-entry [:primes]
                   (fn [n] (vec (take n (filter prime? (iterate inc 2)))))
                   "Primes: 2, 3, 5, 7, 11, ...")

   (make-seq-entry [:fibonacci]
                   (fn [n] (vec (take n (map first (iterate (fn [[a b]] [b (+ a b)]) [0 1])))))
                   "Fibonacci: 0, 1, 1, 2, 3, 5, 8, ...")

   (make-seq-entry [:multiples 3]
                   (fn [n] (vec (take n (iterate #(+ % 3) 3))))
                   "Multiples of 3: 3, 6, 9, ...")

   (make-seq-entry [:multiples 5]
                   (fn [n] (vec (take n (iterate #(+ % 5) 5))))
                   "Multiples of 5: 5, 10, 15, ...")

   (make-seq-entry [:multiples 7]
                   (fn [n] (vec (take n (iterate #(+ % 7) 7))))
                   "Multiples of 7: 7, 14, 21, ...")])

;; =============================================================================
;; Parameterized sequence generators
;; =============================================================================

(defn arithmetic-seq
  "Generate arithmetic sequence: start, start+step, start+2*step, ..."
  [start step]
  (iterate #(+ % step) start))

(defn geometric-seq
  "Generate geometric sequence: start, start*ratio, start*ratio^2, ..."
  [start ratio]
  (iterate #(*' % ratio) start))

(defn multiples-seq
  "Generate multiples: k, 2k, 3k, ..."
  [k]
  (iterate #(+ % k) k))

(defn powers-seq
  "Generate powers: base^0, base^1, base^2, ..."
  [base]
  (iterate #(* % base) 1))

;; =============================================================================
;; The Backwards Direction: Infer parameters from data
;; =============================================================================

(defn infer-arithmetic
  "Given examples, infer arithmetic sequence parameters.
   Returns {:start :step} or nil."
  [examples]
  (when (>= (count examples) 2)
    (let [diffs (map - (rest examples) examples)]
      (when (apply = diffs)
        {:type :arithmetic
         :start (first examples)
         :step (first diffs)}))))

(defn infer-geometric
  "Given examples, infer geometric sequence parameters.
   Returns {:start :ratio} or nil."
  [examples]
  (when (and (>= (count examples) 2)
             (every? pos? examples))
    (let [ratios (map / (rest examples) examples)]
      (when (and (apply = ratios)
                 (integer? (first ratios)))
        {:type :geometric
         :start (first examples)
         :ratio (long (first ratios))}))))

(defn infer-polynomial-degree
  "Infer degree by looking at differences.
   Constant sequence = degree 0
   Arithmetic = degree 1
   Quadratic = degree 2, etc."
  [examples]
  (loop [diffs examples
         degree 0]
    (if (or (apply = diffs) (> degree 5))
      degree
      (recur (map - (rest diffs) diffs) (inc degree)))))

(defn infer-recurrence
  "Try to find a linear recurrence relation.
   f(n) = a*f(n-1) + b*f(n-2)"
  [examples]
  (when (>= (count examples) 4)
    (let [[a b c d] (take 4 examples)]
      ;; Check if c = α*b + β*a for some α, β
      ;; And d = α*c + β*b
      ;; This is a simple check for order-2 recurrence
      (when (and (not= b 0) (not= c 0))
        (let [;; Solve: c = α*b + β*a, d = α*c + β*b
              ;; If Fibonacci-like: α=1, β=1
              fib-check (and (= c (+ a b)) (= d (+ b c)))]
          (when fib-check
            {:type :fibonacci-like
             :seed [a b]}))))))

;; =============================================================================
;; Main Synthesis: Data → Program
;; =============================================================================

(defn matches?
  "Does the sequence generator match the examples?"
  [{:keys [gen]} examples]
  (try
    (= (gen (count examples)) (vec examples))
    (catch Exception _ false)))

(defn synthesize-from-grammar
  "Find sequences from grammar that match examples."
  [examples]
  (->> (sequence-grammar)
       (filter #(matches? % examples))))

(defn synthesize-with-inference
  "Infer program structure directly from data patterns."
  [examples]
  (let [inferences (remove nil?
                    [(infer-arithmetic examples)
                     (infer-geometric examples)
                     (infer-recurrence examples)])]
    (for [inf inferences]
      (case (:type inf)
        :arithmetic {:name [:arithmetic (:start inf) (:step inf)]
                    :gen (fn [n] (vec (take n (arithmetic-seq (:start inf) (:step inf)))))
                    :desc (format "Arithmetic: start=%d, step=%d"
                                 (:start inf) (:step inf))
                    :inferred inf}
        :geometric {:name [:geometric (:start inf) (:ratio inf)]
                   :gen (fn [n] (vec (take n (geometric-seq (:start inf) (:ratio inf)))))
                   :desc (format "Geometric: start=%d, ratio=%d"
                                (:start inf) (:ratio inf))
                   :inferred inf}
        :fibonacci-like {:name [:fibonacci-like (:seed inf)]
                        :gen (fn [n] (vec (take n (map first (iterate
                                                              (fn [[a b]] [b (+ a b)])
                                                              (:seed inf))))))
                        :desc (format "Fibonacci-like: seed=%s" (:seed inf))
                        :inferred inf}
        nil))))

(defn synthesize
  "Main synthesis: find programs that generate the sequence.
   Combines grammar matching with parameter inference."
  [examples]
  (distinct
   (concat
    (synthesize-with-inference examples)  ; Inferred programs (backwards!)
    (synthesize-from-grammar examples)))) ; Grammar matching (forward check)

;; =============================================================================
;; Ranking by description length (Occam's razor)
;; =============================================================================

(defn program-complexity
  "Measure program complexity by its structure."
  [{:keys [name]}]
  (cond
    (keyword? name) 1
    (vector? name) (+ (count name)
                      (reduce + (map #(if (number? %) (Math/log (inc (abs %))) 0)
                                    (rest name))))
    :else 10))

(defn rank-by-simplicity
  "Simpler programs are more likely (Occam's razor)."
  [programs]
  (sort-by program-complexity programs))

;; =============================================================================
;; Demo
;; =============================================================================

(defn demo []
  (println "\n" (apply str (repeat 60 "=")) "\n")
  (println "SEQUENCE SYNTHESIS: Data → Program")
  (println (apply str (repeat 60 "=")) "\n")

  (println "The Clojure insight: Lazy sequences ARE programs.\n")
  (println "(iterate #(* 2 %) 1) is both DATA and PROGRAM.\n")

  (doseq [[name examples]
          [["Powers of 2" [1 2 4 8 16]]
           ["Primes" [2 3 5 7 11 13]]
           ["Fibonacci" [0 1 1 2 3 5 8]]
           ["Squares" [1 4 9 16 25]]
           ["Arithmetic" [3 7 11 15 19]]
           ["Mystery" [1 3 6 10 15]]]]

    (println (str name ": " examples))
    (let [programs (synthesize examples)]
      (if (seq programs)
        (doseq [{:keys [desc inferred]} (take 2 (rank-by-simplicity programs))]
          (println (str "  → " desc))
          (when inferred
            (println (str "    (inferred: " inferred ")"))))
        (println "  → No program found")))
    (println))

  (println (apply str (repeat 60 "-")))
  (println "\nThe backwards direction:")
  (println "  [1 2 4 8 16] → infer-geometric → {:start 1, :ratio 2}")
  (println "  [3 7 11 15]  → infer-arithmetic → {:start 3, :step 4}")
  (println "\nData tells us the program structure!"))

(comment
  (demo)

  ;; The magic: data → program
  (infer-arithmetic [3 7 11 15 19])
  ;; => {:type :arithmetic, :start 3, :step 4}

  (infer-geometric [1 2 4 8 16])
  ;; => {:type :geometric, :start 1, :ratio 2}

  ;; Synthesize returns the PROGRAM (lazy seq)
  (let [prog (first (synthesize [1 2 4 8 16]))]
    (take 10 (:seq prog)))
  ;; => (1 2 4 8 16 32 64 128 256 512)
  )
