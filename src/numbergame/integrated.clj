(ns numbergame.integrated
  "Integrated program synthesis combining:
   - clojure.spec: Grammar definition and validation
   - core.logic: Constraint-based program finding
   - Gen.clj: Probabilistic inference over programs

   This demonstrates a uniquely Clojure approach where each tool
   handles what it does best."
  (:require [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as sgen]
            [clojure.core.logic :as l]
            [clojure.core.logic.fd :as fd]
            [gen.dynamic :as g :refer [gen]]
            [gen.distribution.commons-math :as dist]
            [gen.generative-function :as gf]
            [gen.trace :as trace]
            [gen.choicemap :as choicemap]
            [numbergame.logic :as logic]))

;; =============================================================================
;; LAYER 1: clojure.spec - Define the Grammar
;; =============================================================================
;; spec answers: "What is a valid program?"

(s/def ::base (s/int-in 2 11))           ; 2-10
(s/def ::mult-k (s/int-in 2 21))         ; 2-20
(s/def ::range-lo (s/int-in 1 101))      ; 1-100
(s/def ::range-hi (s/int-in 1 101))      ; 1-100

;; Primitive program types
(s/def ::power-of (s/tuple #{:power-of} ::base))
(s/def ::mult (s/tuple #{:mult} ::mult-k))
(s/def ::range (s/and (s/tuple #{:range} ::range-lo ::range-hi)
                      (fn [[_ lo hi]] (<= lo hi))))
(s/def ::prime (s/tuple #{:prime}))
(s/def ::square (s/tuple #{:square}))
(s/def ::even (s/tuple #{:even}))
(s/def ::odd (s/tuple #{:odd}))

;; A program is one of the primitive types
(s/def ::primitive-program
  (s/or :power-of ::power-of
        :mult ::mult
        :range ::range
        :prime ::prime
        :square ::square
        :even ::even
        :odd ::odd))

;; Composite programs (recursive)
(s/def ::program
  (s/or :primitive ::primitive-program
        :union (s/tuple #{:union} ::program ::program)
        :intersect (s/tuple #{:intersect} ::program ::program)))

;; Validation functions
(defn valid-program? [prog]
  (s/valid? ::primitive-program prog))

(defn explain-program [prog]
  (s/explain-str ::primitive-program prog))

;; =============================================================================
;; LAYER 2: spec generators - Random Program Generation
;; =============================================================================
;; spec's generative testing gives us random valid programs

(defn generate-random-programs
  "Generate n random valid programs using spec generators."
  [n]
  (take n (repeatedly #(sgen/generate (s/gen ::primitive-program)))))

;; Custom generator with controlled distribution
(def program-type-gen
  "Generator that samples program types with custom weights."
  (sgen/frequency
   [[10 (sgen/return [:prime])]
    [10 (sgen/return [:square])]
    [5 (sgen/return [:even])]
    [5 (sgen/return [:odd])]
    [20 (sgen/fmap (fn [k] [:power-of k]) (s/gen ::base))]
    [30 (sgen/fmap (fn [k] [:mult k]) (s/gen ::mult-k))]
    [20 (sgen/fmap (fn [[lo hi]] [:range (min lo hi) (max lo hi)])
                   (sgen/tuple (s/gen ::range-lo) (s/gen ::range-hi)))]]))

;; =============================================================================
;; LAYER 3: core.logic - Find Consistent Programs
;; =============================================================================
;; core.logic answers: "Which programs contain all examples?"
;; (Reuses logic from numbergame.logic)

(defn find-consistent-programs
  "Use core.logic to find programs consistent with examples."
  [examples max-programs]
  (logic/find-programs-primitives examples max-programs))

;; =============================================================================
;; LAYER 4: Gen.clj - Probabilistic Modeling
;; =============================================================================
;; Gen.clj answers: "What's the probability distribution over programs?"

(defn program->index
  "Convert program to index in a program list."
  [programs prog]
  (.indexOf (vec programs) prog))

(defn programs->categorical-params
  "Convert programs + examples to categorical distribution parameters.
   Uses size principle for likelihood, rule-bias for prior."
  [programs examples prior]
  (let [n (count examples)
        log-weights (mapv (fn [prog]
                           (let [size (logic/concept-size prog)
                                 log-like (- (* n (Math/log (max 1 size))))
                                 log-pr (logic/log-prior prior prog)]
                             (+ log-pr log-like)))
                         programs)
        ;; Normalize to probabilities
        max-lw (apply max log-weights)
        weights (mapv #(Math/exp (- % max-lw)) log-weights)
        total (reduce + weights)]
    (mapv #(/ % total) weights)))

;; Gen.clj generative model for concept learning
(def concept-model
  "Generative model: sample a program, then sample examples from it.

   This is the FORWARD model - how data is generated."
  (gen [n-examples prior programs]
    ;; Sample a program from the prior (weighted by type)
    (let [probs (programs->categorical-params programs [] prior)
          prog-idx (g/trace! :program dist/categorical probs)
          prog (nth programs prog-idx)
          ;; Get the extension of this program
          extension (vec (logic/concept-extension prog))
          ext-size (count extension)
          ;; Sample n examples uniformly from the extension
          examples (vec (for [i (range n-examples)]
                          (let [idx (g/trace! (keyword (str "example-" i))
                                             dist/categorical
                                             (vec (repeat ext-size (/ 1.0 ext-size))))]
                            (nth extension idx))))]
      {:program prog
       :examples examples})))

;; Gen.clj model for INFERENCE - given examples, infer program
(def inference-model
  "Model for inference: given candidate programs and examples,
   sample a program weighted by posterior probability."
  (gen [examples prior]
    (let [;; Use core.logic to find consistent programs
          programs (vec (find-consistent-programs examples 50))
          n-programs (count programs)]
      (if (zero? n-programs)
        {:error "No consistent programs found"}
        (let [;; Compute posterior probabilities
              probs (programs->categorical-params programs examples prior)
              ;; Sample from posterior
              prog-idx (g/trace! :program dist/categorical probs)
              prog (nth programs prog-idx)]
          {:program prog
           :all-programs programs
           :probabilities (zipmap programs probs)})))))

;; =============================================================================
;; LAYER 5: Gen.clj Inference Algorithms
;; =============================================================================

(defn importance-sample
  "Run importance sampling to estimate posterior over programs.

   Returns empirical distribution from n-samples."
  [examples n-samples prior]
  (let [programs (vec (find-consistent-programs examples 50))
        probs (programs->categorical-params programs examples prior)]
    ;; Sample programs according to posterior
    (frequencies
     (for [_ (range n-samples)]
       (let [tr (gf/simulate inference-model [examples prior])
             choices (trace/get-choices tr)
             prog-idx (choicemap/get-value choices :program)]
         (nth programs prog-idx))))))

(defn posterior-summary
  "Compute full posterior summary using Gen.clj traces."
  [examples prior]
  (let [programs (vec (find-consistent-programs examples 50))
        probs (programs->categorical-params programs examples prior)
        prog-probs (zipmap programs probs)]
    {:examples examples
     :n-consistent (count programs)
     :top-5 (->> prog-probs
                 (sort-by val >)
                 (take 5)
                 (mapv (fn [[prog prob]]
                         {:program prog
                          :probability prob
                          :size (logic/concept-size prog)})))
     :entropy (- (reduce + (map #(if (pos? %) (* % (Math/log %)) 0) probs)))}))

;; =============================================================================
;; LAYER 6: Full Pipeline - Combining All Three
;; =============================================================================

(defn synthesize-with-trace
  "Full synthesis pipeline with Gen.clj tracing.

   1. spec validates input
   2. core.logic finds consistent programs
   3. Gen.clj computes posterior and samples

   Returns trace for inspection."
  [examples & {:keys [prior n-samples]
               :or {prior logic/rule-biased-prior
                    n-samples 1000}}]
  (let [;; Validate examples are in domain
        _ (assert (every? #(<= 1 % 100) examples) "Examples must be in [1,100]")

        ;; core.logic: find consistent programs
        programs (find-consistent-programs examples 50)
        _ (assert (seq programs) "No consistent programs found")

        ;; Gen.clj: run inference model
        tr (gf/simulate inference-model [examples prior])
        choices (trace/get-choices tr)
        result (trace/get-retval tr)
        score (trace/get-score tr)

        ;; Importance sampling for uncertainty
        samples (importance-sample examples n-samples prior)]

    {:examples examples
     :n-consistent (count programs)
     :map-program (:program result)  ; Maximum a posteriori
     :trace-score score
     :posterior-samples samples
     :top-by-samples (->> samples
                          (sort-by val >)
                          (take 5))}))

;; =============================================================================
;; Demo
;; =============================================================================

(defn demo
  "Demonstrate the integrated approach."
  []
  (println "\n" (apply str (repeat 70 "=")) "\n")
  (println "INTEGRATED SYNTHESIS: spec + core.logic + Gen.clj")
  (println (apply str (repeat 70 "=")) "\n")

  (println "LAYER 1: clojure.spec - Grammar Definition")
  (println "  Valid program? [:power-of 2] =>" (valid-program? [:power-of 2]))
  (println "  Valid program? [:power-of 99] =>" (valid-program? [:power-of 99]))
  (println "  Random programs from spec:" (take 3 (generate-random-programs 10)))
  (println)

  (println "LAYER 2: core.logic - Constraint Satisfaction")
  (let [examples [16 8 2 64]
        programs (find-consistent-programs examples 10)]
    (println "  Examples:" examples)
    (println "  Consistent programs:" (take 5 programs)))
  (println)

  (println "LAYER 3: Gen.clj - Probabilistic Inference")
  (let [examples [7 11 13 17]
        summary (posterior-summary examples logic/rule-biased-prior)]
    (println "  Examples:" examples)
    (println "  Posterior entropy:" (format "%.3f" (:entropy summary)))
    (println "  Top 3 by posterior:")
    (doseq [{:keys [program probability size]} (take 3 (:top-5 summary))]
      (println (format "    %-20s prob: %.4f  size: %d"
                      (pr-str program) probability size))))
  (println)

  (println "LAYER 4: Full Pipeline with Tracing")
  (let [result (synthesize-with-trace [16 8 2 64] :n-samples 500)]
    (println "  MAP program:" (:map-program result))
    (println "  Trace score:" (format "%.3f" (:trace-score result)))
    (println "  Top by importance sampling:")
    (doseq [[prog count] (take 3 (:top-by-samples result))]
      (println (format "    %-20s samples: %d" (pr-str prog) count))))

  (println "\n" (apply str (repeat 70 "-")) "\n")
  (println "Each tool handles what it does best:")
  (println "  spec      → Grammar, validation, random generation")
  (println "  core.logic → Constraint satisfaction, parameter discovery")
  (println "  Gen.clj   → Probabilistic modeling, inference, tracing"))

(comment
  ;; REPL exploration

  ;; spec: validate and generate
  (s/valid? ::primitive-program [:power-of 2])
  (s/valid? ::primitive-program [:power-of 99])  ; false - out of range
  (generate-random-programs 5)

  ;; core.logic: find consistent programs
  (find-consistent-programs [16 8 2 64] 10)

  ;; Gen.clj: posterior inference
  (posterior-summary [7 11 13 17] logic/rule-biased-prior)

  ;; Full pipeline
  (synthesize-with-trace [16 8 2 64])

  ;; Simulate from generative model
  (let [programs (find-consistent-programs [16 8 2 64] 20)
        tr (gf/simulate concept-model [4 logic/uniform-prior programs])]
    (trace/get-retval tr))

  ;; Run demo
  (demo)
  )
