package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// IntentPlan is the parsed plan from the intent LLM.
type IntentPlan struct {
	Actions []IntentAction `json:"actions"`
	Speech  string         `json:"speech"`
}

// IntentAction is a single action within a plan.
type IntentAction struct {
	EntityID    string                 `json:"entity_id"`
	Service     string                 `json:"service"`
	ServiceData map[string]interface{} `json:"service_data,omitempty"`
}

// IntentClient communicates with the Qwen2.5-1.5B intent parser.
type IntentClient struct {
	baseURL string
	client  *http.Client
}

// NewIntentClient creates a new intent LLM client.
func NewIntentClient(baseURL string) *IntentClient {
	return &IntentClient{
		baseURL: baseURL,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

// Parse sends user text and candidates to the intent LLM, returns a structured plan.
func (ic *IntentClient) Parse(userText string, candidates []Candidate) (*IntentPlan, error) {
	// Build candidate list for the prompt
	var candidateLines []string
	for _, c := range candidates {
		line := fmt.Sprintf("- entity_id: %s, name: %q, domain: %s", c.EntityID, c.Name, c.Domain)
		if c.Area != "" {
			line += fmt.Sprintf(", area: %s", c.Area)
		}
		candidateLines = append(candidateLines, line)
	}

	systemPrompt := `You are an intent parser for Home Assistant voice commands. Given user text and candidate entities, output ONLY valid JSON matching this schema:
{"actions": [{"entity_id": "...", "service": "...", "service_data": {}}], "speech": "..."}

Rules:
- entity_id MUST be one of the provided candidates. Never invent entity IDs.
- service must be a valid HA service for the domain (e.g., turn_on, turn_off, toggle for lights/switches; media_play, media_pause for media_player).
- service_data is optional (e.g., {"brightness": 128} for lights).
- speech is a short confirmation message (e.g., "Turning on the basement stairs light").
- For compound commands, include multiple actions.
- Output ONLY the JSON object. No markdown, no explanation.`

	userPrompt := fmt.Sprintf("User said: %q\n\nCandidate entities:\n%s\n\nOutput the JSON plan:",
		userText, strings.Join(candidateLines, "\n"))

	reqBody := map[string]interface{}{
		"model": "qwen2.5-1.5b-instruct",
		"messages": []map[string]string{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": userPrompt},
		},
		"max_tokens":  256,
		"temperature": 0.1,
		"top_p":       1.0,
	}

	bodyBytes, _ := json.Marshal(reqBody)
	resp, err := ic.client.Post(ic.baseURL+"/v1/chat/completions", "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("intent LLM request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("intent LLM status %d: %s", resp.StatusCode, string(body))
	}

	var llmResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&llmResp); err != nil {
		return nil, fmt.Errorf("intent LLM decode: %w", err)
	}

	if len(llmResp.Choices) == 0 {
		return nil, fmt.Errorf("intent LLM returned no choices")
	}

	content := strings.TrimSpace(llmResp.Choices[0].Message.Content)
	// Strip markdown code fences if present
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)

	var plan IntentPlan
	if err := json.Unmarshal([]byte(content), &plan); err != nil {
		log.Printf("Intent LLM returned non-JSON: %q", content)
		return nil, fmt.Errorf("intent LLM JSON parse: %w (raw: %s)", err, content)
	}

	// Validate: all entity_ids must be from candidates
	validIDs := make(map[string]bool)
	for _, c := range candidates {
		validIDs[c.EntityID] = true
	}
	var validActions []IntentAction
	for _, a := range plan.Actions {
		if validIDs[a.EntityID] {
			validActions = append(validActions, a)
		} else {
			log.Printf("Intent LLM invented entity_id %q — dropping", a.EntityID)
		}
	}
	plan.Actions = validActions

	return &plan, nil
}
