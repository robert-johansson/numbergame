(ns ex03-choicemaps
  "Example 3: Choice Maps and Addressing

   Choice maps are the data structure used to represent the random
   choices made during execution of a generative function.

   Key concepts:
   - Addresses can be any Clojure value (keywords, vectors, etc.)
   - Choice maps are hierarchical (can nest)
   - Used both for reading traces AND for constraints in inference"
  (:require [gen.distribution.commons-math :as dist]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.generative-function :as gf]
            [gen.trace :as trace]
            [gen.choicemap :as choicemap]))

;; =============================================================================
;; PART 1: Addresses
;; =============================================================================

;; Addresses can be any Clojure value

(def addressing-demo
  (gen []
    ;; Keyword address (most common)
    (dynamic/trace! :my-choice dist/normal 0 1)

    ;; Vector address (common for indexed data)
    (dynamic/trace! [:data 0] dist/normal 0 1)
    (dynamic/trace! [:data 1] dist/normal 0 1)

    ;; String address
    (dynamic/trace! "string-addr" dist/normal 0 1)

    ;; Integer address
    (dynamic/trace! 42 dist/normal 0 1)

    ;; Nested vector address
    (dynamic/trace! [:level1 :level2 :level3] dist/normal 0 1)

    :done))

(defn demo-addresses []
  (println "\n=== Different Address Types ===\n")
  (let [tr (gf/simulate addressing-demo [])
        choices (trace/get-choices tr)]
    (println "All choices:")
    (doseq [[addr value] choices]
      (println "  " addr "->" value))))

;; =============================================================================
;; PART 2: Accessing Choice Map Values
;; =============================================================================

(def sample-model
  (gen [n]
    (let [mean (dynamic/trace! :mean dist/normal 0 10)]
      (dotimes [i n]
        (dynamic/trace! [:obs i] dist/normal mean 1))
      mean)))

(defn demo-access []
  (println "\n=== Accessing Choice Map Values ===\n")

  (let [tr (gf/simulate sample-model [3])
        choices (trace/get-choices tr)]

    (println "Trace choices:" choices)

    ;; Method 1: Use choicemap as a function
    (println "\nMethod 1 - Call as function:")
    (println "  (choices :mean) =" (choices :mean))

    ;; Method 2: Use get
    (println "\nMethod 2 - Use get:")
    (println "  (get choices :mean) =" (get choices :mean))
    (println "  (get choices [:obs 0]) =" (get choices [:obs 0]))

    ;; Method 3: Use choicemap/get-value
    (println "\nMethod 3 - choicemap/get-value:")
    (println "  " (choicemap/get-value choices :mean))

    ;; Method 4: Keyword lookup (for keyword addresses)
    (println "\nMethod 4 - Keyword lookup:")
    (println "  (:mean choices) =" (:mean choices))))

;; =============================================================================
;; PART 3: Hierarchical Choice Maps
;; =============================================================================

(def inner-gen
  (gen []
    {:a (dynamic/trace! :a dist/normal 0 1)
     :b (dynamic/trace! :b dist/normal 0 1)}))

(def outer-gen
  (gen []
    {:first (dynamic/trace! :first inner-gen)
     :second (dynamic/trace! :second inner-gen)}))

(defn demo-hierarchical []
  (println "\n=== Hierarchical Choice Maps ===\n")

  (let [tr (gf/simulate outer-gen [])
        choices (trace/get-choices tr)]

    (println "Full choice map:" choices)

    ;; Access nested values
    (println "\nAccessing nested values:")
    (println "  [:first :a] =" (get-in choices [:first :a]))
    (println "  [:second :b] =" (get-in choices [:second :b]))

    ;; Get submaps
    (println "\nSubmaps:")
    (println "  :first submap =" (choicemap/get-submap choices :first))

    ;; Check for submaps
    (println "\nChecking structure:")
    (println "  has-submap? :first =" (choicemap/has-submap? choices :first))
    (println "  has-value? :first =" (choicemap/has-value? choices :first))))

;; =============================================================================
;; PART 4: Creating Choice Maps (for Constraints)
;; =============================================================================

(defn demo-creating []
  (println "\n=== Creating Choice Maps ===\n")

  ;; Method 1: From a regular map (auto-converts)
  (let [cm1 {:x 1.0 :y 2.0}]
    (println "From map:" cm1))

  ;; Method 2: Using choicemap function
  (let [cm2 (choicemap/choicemap {:x 1.0 [:obs 0] 3.0})]
    (println "Using choicemap/choicemap:" cm2))

  ;; Method 3: Empty then assoc
  (let [cm3 (-> (choicemap/choicemap)
                (assoc :a 1.0)
                (assoc :b 2.0))]
    (println "Built with assoc:" cm3))

  ;; Method 4: Nested structure
  (let [cm4 {:outer {:inner 42}}]
    (println "Nested:" cm4)))

;; =============================================================================
;; PART 5: Using Choice Maps as Constraints
;; =============================================================================

(def constrained-model
  (gen []
    (let [x (dynamic/trace! :x dist/normal 0 1)
          y (dynamic/trace! :y dist/normal x 1)]
      {:x x :y y})))

(defn demo-constraints []
  (println "\n=== Using Constraints (gf/generate) ===\n")

  ;; Unconstrained simulation
  (println "Unconstrained simulation:")
  (let [tr (gf/simulate constrained-model [])]
    (println "  Choices:" (trace/get-choices tr)))

  ;; Constrain x to be 2.0
  (println "\nConstrained (x=2.0):")
  (let [constraints {:x 2.0}
        result (gf/generate constrained-model [] constraints)
        tr (:trace result)]
    (println "  Constraints:" constraints)
    (println "  Choices:" (trace/get-choices tr))
    (println "  Weight:" (:weight result)))

  ;; Constrain both
  (println "\nConstrained (x=2.0, y=3.0):")
  (let [constraints {:x 2.0 :y 3.0}
        result (gf/generate constrained-model [] constraints)
        tr (:trace result)]
    (println "  Constraints:" constraints)
    (println "  Choices:" (trace/get-choices tr))
    (println "  Weight:" (:weight result))))

;; =============================================================================
;; PART 6: Choice Map Utilities
;; =============================================================================

(defn demo-utilities []
  (println "\n=== Choice Map Utilities ===\n")

  (let [cm (choicemap/choicemap {:a 1 :b 2 [:nested :x] 3})]
    (println "Choice map:" cm)

    ;; Check emptiness
    (println "\nchoicemap/empty?:" (choicemap/empty? cm))
    (println "choicemap/empty? on empty:" (choicemap/empty? (choicemap/choicemap)))

    ;; Get all shallow values
    (println "\nget-values-shallow:" (choicemap/get-values-shallow cm))

    ;; Get all shallow submaps
    (println "get-submaps-shallow:" (choicemap/get-submaps-shallow cm))

    ;; Convert to regular map
    (println "\n->map:" (choicemap/->map cm))))

;; =============================================================================
;; PART 7: Common Patterns
;; =============================================================================

(defn demo-patterns []
  (println "\n=== Common Addressing Patterns ===\n")

  ;; Pattern 1: Indexed observations
  (println "Pattern 1 - Indexed observations [:y i]:")
  (let [model (gen [n]
                (dotimes [i n]
                  (dynamic/trace! [:y i] dist/normal 0 1)))
        tr (gf/simulate model [3])]
    (println "  " (trace/get-choices tr)))

  ;; Pattern 2: Named groups
  (println "\nPattern 2 - Named groups [:group :param]:")
  (let [model (gen []
                (dynamic/trace! [:prior :mean] dist/normal 0 10)
                (dynamic/trace! [:prior :std] dist/gamma 1 1)
                (dynamic/trace! [:likelihood :obs] dist/normal 0 1))
        tr (gf/simulate model [])]
    (println "  " (trace/get-choices tr)))

  ;; Pattern 3: Building constraints for observations
  (println "\nPattern 3 - Building observation constraints:")
  (let [observed-ys [1.5 2.3 1.8 2.1]
        constraints (reduce (fn [cm [i y]]
                              (assoc cm [:y i] y))
                            (choicemap/choicemap)
                            (map-indexed vector observed-ys))]
    (println "  Observed:" observed-ys)
    (println "  Constraints:" constraints)))

;; =============================================================================
;; Main
;; =============================================================================

(defn -main [& _]
  (demo-addresses)
  (demo-access)
  (demo-hierarchical)
  (demo-creating)
  (demo-constraints)
  (demo-utilities)
  (demo-patterns)
  (println "\nDone!"))

(comment
  (-main)

  ;; Explore interactively:
  (choicemap/choicemap {:x 1 :y 2})
  (choicemap/choicemap [1 2 3])  ; vector choice map
  )
