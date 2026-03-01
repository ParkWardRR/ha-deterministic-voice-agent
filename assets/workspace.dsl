workspace "Deterministic HA Voice Agent" "Deterministic-first voice control for Home Assistant" {

    model {
        user = person "Smart Home User" "User speaking voice commands"

        ha = softwareSystem "Home Assistant" "Smart Home Hub" {
            assist = container "Assist Pipeline" "Speech-to-text processing"
            agent = container "Deterministic Agent Custom Component" "Routes text to Orchestrator and executes final plan"
            devices = container "Smart Home Devices" "Lights, Thermostats, Locks, etc."
        }

        db = softwareSystem "PostgreSQL + pgvector (NVMe Storage)" "Vector Database storing entity embeddings" "Database"
        onnx = softwareSystem "Local Intent LLM (ONNX + RTX 3060 CUDA)" "Qwen 2.5 1.5B via Direct CUDA & TensorRT hooks" "AI"
        glm = softwareSystem "Fallback Chat LLM" "General assistant chat model" "AI"

        orch = softwareSystem "Rust Orchestrator Daemon (<50MB RAM)" "Core execution engine for entity resolution and safety" {
            api = container "Core API (:5000)" "HTTP routing layer (Axum + Tokio async)" "Rust"
            resolver = container "Resolver Engine (AVX-512 SIMD)" "Lexical + Vector Entity Matching utilizing explicit CPU SIMD AVX-512 dot-product computations" "Rust" {
                lexical = component "Lexical Search" "In-memory substring matching cache"
                simd = component "SIMD AVX-512 Scorer" "AVX-512 and AVX2 hardware accelerated dot-product vector ranking"
                merge = component "Deduplication & Ranker" "Merges lists securely, favoring deterministic matches"
            }
            plan = container "Intent Execution Planner" "Coordinates ML outputs and dynamically adjusts PostgreSQL connection pools (DB_MAX_CONNS)" "Rust"
            safety = container "Safety Gate" "Evaluates domain rules and requires strict user WebSockets confirmations" "Rust"
        }

        user -> assist "Speaks (e.g. 'turn on living room')"
        assist -> agent "Passes text command"
        agent -> api "POST /v1/ha/process"
        
        api -> resolver "Passes input text"
        resolver -> db "Queries lexical hashmap and SIMD vector embeddings"
        
        api -> plan "Delegates execution plan"
        plan -> onnx "Requests JSON action generation on the RTX 3060"
        plan -> glm "Requests answer if no entities match"
        
        plan -> safety "Passes actions for validation"
        safety -> agent "Returns allowed intent plan"
        agent -> devices "Executes service calls securely"
        
        # Link explicit internal components
        api -> lexical "Queries"
        api -> simd "Queries"
        lexical -> merge "Updates candidates"
        simd -> merge "Updates candidates"
    }

    views {
        systemLandscape "architecture" "High level cloud and local architecture" {
            include *
            autoLayout
        }

        dynamic orch "request_flow" "Request processing workflow" {
            user -> assist "1. Speak command"
            assist -> agent "2. Process to text"
            agent -> api "3. Forward Request"
            api -> resolver "4. Resolve Entities (AVX-512)"
            resolver -> db "5. Lookup Vectors"
            api -> plan "6. Build Action Plan"
            plan -> onnx "7. Parse strict JSON Intent (RTX 3060 CUDA)"
            plan -> safety "8. Verify Domain Security"
            safety -> agent "9. Return Verified Actions"
            agent -> devices "10. Mutate State via HA Core"
            autoLayout tb
        }

        component resolver "entity_resolution" "Entity Resolution Components" {
            include api resolver db
            autoLayout lr
        }

        container orch "safety_gate" "Safety Gate Flow" {
            include plan safety agent devices
            autoLayout lr
        }

        theme default
        
        styles {
            element "Software System" {
                background #1168bd
                color #ffffff
                shape RoundedBox
            }
            element "Person" {
                background #08427b
                color #ffffff
                shape Person
            }
            element "Container" {
                background #438dd5
                color #ffffff
                shape RoundedBox
            }
            element "Database" {
                shape Cylinder
                background #6a1b9a
            }
            element "AI" {
                background #00695c
                shape Robot
            }
        }
    }
}
