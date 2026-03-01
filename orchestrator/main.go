package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"
)

// ProcessRequest is the input from the HA custom integration.
type ProcessRequest struct {
	Text           string `json:"text"`
	ConversationID string `json:"conversation_id,omitempty"`
}

// Action represents a single HA service call.
type Action struct {
	EntityID    string                 `json:"entity_id"`
	Domain      string                 `json:"domain"`
	Service     string                 `json:"service"`
	ServiceData map[string]interface{} `json:"service_data,omitempty"`
}

// ProcessResponse is returned to the HA custom integration.
type ProcessResponse struct {
	Actions            []Action `json:"actions,omitempty"`
	Speech             string   `json:"speech"`
	NeedsConfirmation  bool     `json:"needs_confirmation,omitempty"`
	NeedsClarification bool     `json:"needs_clarification,omitempty"`
	NonHAResponse      string   `json:"non_ha_response,omitempty"`
}

// HealthResponse reports dependency status.
type HealthResponse struct {
	Status string            `json:"status"`
	Uptime string            `json:"uptime"`
	Deps   map[string]string `json:"deps"`
}

var (
	resolver  *Resolver
	intentLLM *IntentClient
	glmClient *GLMClient
	safety    *SafetyGate
	startTime time.Time
)

func main() {
	startTime = time.Now()

	pgDSN := envOr("PG_DSN", "host=localhost port=5432 dbname=agent user=agent password=change-me sslmode=disable")
	embedderURL := os.Getenv("EMBEDDER_URL")
	intentURL := envOr("INTENT_URL", "http://localhost:8081")
	glmURL := envOr("GLM_URL", "http://localhost:8080")
	listenAddr := envOr("LISTEN_ADDR", "0.0.0.0:5000")

	var err error
	resolver, err = NewResolver(pgDSN, NewEmbedderClient(embedderURL))
	if err != nil {
		log.Fatalf("Failed to connect to pgvector: %v", err)
	}
	defer resolver.Close()

	intentLLM = NewIntentClient(intentURL)
	glmClient = NewGLMClient(glmURL)
	safety = NewSafetyGate()

	http.HandleFunc("/v1/ha/process", handleProcess)
	http.HandleFunc("/healthz", handleHealth)

	log.Printf("Deterministic agent orchestrator listening on %s", listenAddr)
	log.Fatal(http.ListenAndServe(listenAddr, nil))
}

func handleProcess(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req ProcessRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, ProcessResponse{Speech: "I didn't understand that."})
		return
	}

	if req.Text == "" {
		writeJSON(w, http.StatusBadRequest, ProcessResponse{Speech: "No text provided."})
		return
	}

	start := time.Now()
	log.Printf("Processing: %q", req.Text)

	// Step 1: Resolve entities via lexical + vector search
	candidates, err := resolver.Resolve(req.Text)
	if err != nil {
		log.Printf("Resolver error: %v", err)
		writeJSON(w, http.StatusOK, ProcessResponse{Speech: "Voice control is temporarily degraded. Please try again."})
		return
	}

	// Step 2: If no candidates found, try GLM fallback (non-HA query)
	if len(candidates) == 0 {
		if shouldRouteToGLM(req.Text) {
			log.Printf("No entity match for question-like input, routing to GLM fallback")
			glmResp, err := glmClient.Ask(req.Text)
			if err != nil {
				log.Printf("GLM fallback error: %v", err)
				writeJSON(w, http.StatusOK, ProcessResponse{Speech: "I couldn't find any matching devices or answer your question."})
				return
			}
			writeJSON(w, http.StatusOK, ProcessResponse{
				Speech:        glmResp,
				NonHAResponse: glmResp,
			})
			log.Printf("GLM fallback responded in %v", time.Since(start))
			return
		}
		writeJSON(w, http.StatusOK, ProcessResponse{
			Speech:             "I couldn't find a matching device. Please include the room or exact device name.",
			NeedsClarification: true,
		})
		return
	}

	// Step 3: Confidence gate
	if candidates[0].Score < 0.70 {
		names := extractTopNames(candidates, 3)
		writeJSON(w, http.StatusOK, ProcessResponse{
			Speech:             fmt.Sprintf("I'm not fully sure which device you meant. Did you mean: %s?", joinNames(names)),
			NeedsClarification: true,
		})
		return
	}

	// Step 4: Tie detection - if top-2 are too close, clarify
	if len(candidates) >= 2 && (candidates[0].Score-candidates[1].Score) < 0.05 {
		names := make([]string, 0, min(len(candidates), 5))
		for _, c := range candidates[:min(len(candidates), 5)] {
			names = append(names, c.Name)
		}
		speech := fmt.Sprintf("Which one did you mean? I found: %s", joinNames(names))
		writeJSON(w, http.StatusOK, ProcessResponse{
			Speech:             speech,
			NeedsClarification: true,
		})
		log.Printf("Clarification needed: %v", names)
		return
	}

	// Step 5: Send to intent LLM for action parsing
	plan, err := intentLLM.Parse(req.Text, candidates)
	if err != nil {
		log.Printf("Intent LLM error: %v, falling back to best-guess single action", err)
		// Fallback: single action based on top candidate
		plan = &IntentPlan{
			Actions: []IntentAction{{
				EntityID: candidates[0].EntityID,
				Service:  guessService(req.Text, candidates[0].Domain),
			}},
			Speech: fmt.Sprintf("OK, %s %s.", guessServiceVerb(req.Text), candidates[0].Name),
		}
	}

	// Step 6: Safety gate
	candidateNames := make(map[string]string, len(candidates))
	for _, c := range candidates {
		candidateNames[c.EntityID] = c.Name
	}
	var actions []Action
	for _, a := range plan.Actions {
		domain := domainOf(a.EntityID)
		if !safety.IsAllowed(domain) {
			log.Printf("BLOCKED: domain %q for %s", domain, a.EntityID)
			continue
		}
		if safety.NeedsConfirmation(domain) || safety.NeedsEntityConfirmation(a.EntityID) {
			target := candidateNames[a.EntityID]
			if target == "" {
				target = a.EntityID
			}
			writeJSON(w, http.StatusOK, ProcessResponse{
				Speech:            fmt.Sprintf("Are you sure you want to %s %s?", a.Service, target),
				NeedsConfirmation: true,
				Actions: []Action{{
					EntityID:    a.EntityID,
					Domain:      domain,
					Service:     a.Service,
					ServiceData: a.ServiceData,
				}},
			})
			return
		}
		actions = append(actions, Action{
			EntityID:    a.EntityID,
			Domain:      domain,
			Service:     a.Service,
			ServiceData: a.ServiceData,
		})
	}

	if len(actions) == 0 {
		writeJSON(w, http.StatusOK, ProcessResponse{Speech: "I can't do that for safety reasons."})
		return
	}

	// Cap at 5 actions per request
	if len(actions) > 5 {
		actions = actions[:5]
	}

	speech := plan.Speech
	if speech == "" {
		speech = "Done."
	}

	writeJSON(w, http.StatusOK, ProcessResponse{
		Actions: actions,
		Speech:  speech,
	})
	log.Printf("Planned %d actions in %v", len(actions), time.Since(start))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	deps := map[string]string{
		"pgvector":   "unknown",
		"intent_llm": "unknown",
		"glm":        "unknown",
	}

	if err := resolver.Ping(); err == nil {
		deps["pgvector"] = "ok"
	} else {
		deps["pgvector"] = err.Error()
	}

	resp, err := http.Get(envOr("INTENT_URL", "http://localhost:8081") + "/health")
	if err == nil && resp.StatusCode == 200 {
		deps["intent_llm"] = "ok"
		resp.Body.Close()
	} else if err == nil {
		deps["intent_llm"] = fmt.Sprintf("status %d", resp.StatusCode)
		resp.Body.Close()
	} else if err != nil {
		deps["intent_llm"] = err.Error()
	}

	resp, err = http.Get(envOr("GLM_URL", "http://localhost:8080") + "/health")
	if err == nil && resp.StatusCode == 200 {
		deps["glm"] = "ok"
		resp.Body.Close()
	} else if err == nil {
		deps["glm"] = fmt.Sprintf("status %d", resp.StatusCode)
		resp.Body.Close()
	} else if err != nil {
		deps["glm"] = err.Error()
	}

	status := "ok"
	for _, v := range deps {
		if v != "ok" {
			status = "degraded"
			break
		}
	}

	writeJSON(w, http.StatusOK, HealthResponse{
		Status: status,
		Uptime: time.Since(startTime).Round(time.Second).String(),
		Deps:   deps,
	})
}

func writeJSON(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func joinNames(names []string) string {
	if len(names) == 0 {
		return ""
	}
	if len(names) == 1 {
		return names[0]
	}
	result := ""
	for i, n := range names {
		if i > 0 {
			if i == len(names)-1 {
				result += ", or "
			} else {
				result += ", "
			}
		}
		result += n
	}
	return result
}

func domainOf(entityID string) string {
	for i, c := range entityID {
		if c == '.' {
			return entityID[:i]
		}
	}
	return entityID
}

func guessService(text, domain string) string {
	lower := toLower(text)
	switch {
	case contains(lower, "turn off") || contains(lower, "shut off") || contains(lower, "disable"):
		return "turn_off"
	case contains(lower, "turn on") || contains(lower, "enable") || contains(lower, "start"):
		return "turn_on"
	case contains(lower, "toggle"):
		return "toggle"
	case contains(lower, "dim") || contains(lower, "brightness"):
		return "turn_on"
	case contains(lower, "play") || contains(lower, "resume"):
		return "media_play"
	case contains(lower, "pause") || contains(lower, "stop"):
		return "media_pause"
	case contains(lower, "volume"):
		return "volume_set"
	default:
		switch domain {
		case "light", "switch", "fan", "input_boolean":
			return "toggle"
		case "media_player":
			return "media_play"
		case "scene":
			return "turn_on"
		case "automation":
			return "trigger"
		default:
			return "toggle"
		}
	}
}

func guessServiceVerb(text string) string {
	lower := toLower(text)
	switch {
	case contains(lower, "turn off") || contains(lower, "shut off"):
		return "turning off"
	case contains(lower, "turn on"):
		return "turning on"
	case contains(lower, "toggle"):
		return "toggling"
	default:
		return "controlling"
	}
}

func toLower(s string) string {
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		b[i] = c
	}
	return string(b)
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(s) > 0 && containsImpl(s, sub))
}

func containsImpl(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func shouldRouteToGLM(text string) bool {
	lower := toLower(text)
	if contains(lower, "?") {
		return true
	}
	questionStarts := []string{
		"what ", "who ", "when ", "where ", "why ", "how ",
		"tell me ", "explain ", "define ",
	}
	for _, q := range questionStarts {
		if hasPrefix(lower, q) {
			return true
		}
	}
	return false
}

func hasPrefix(s, prefix string) bool {
	if len(prefix) > len(s) {
		return false
	}
	return s[:len(prefix)] == prefix
}

func extractTopNames(candidates []Candidate, max int) []string {
	if max <= 0 {
		return nil
	}
	names := make([]string, 0, min(len(candidates), max))
	seen := make(map[string]bool, max)
	for _, c := range candidates {
		if c.Name == "" || seen[c.Name] {
			continue
		}
		seen[c.Name] = true
		names = append(names, c.Name)
		if len(names) == max {
			break
		}
	}
	if len(names) == 0 {
		return []string{"that device"}
	}
	return names
}
