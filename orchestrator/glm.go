package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// GLMClient communicates with the GLM-4.7-Flash fallback for non-HA queries.
type GLMClient struct {
	baseURL string
	client  *http.Client
}

// NewGLMClient creates a new GLM fallback client.
func NewGLMClient(baseURL string) *GLMClient {
	return &GLMClient{
		baseURL: baseURL,
		client:  &http.Client{Timeout: 60 * time.Second},
	}
}

// Ask sends a non-HA question to GLM and returns the response text.
func (gc *GLMClient) Ask(text string) (string, error) {
	reqBody := map[string]interface{}{
		"model": "glm-4.7-flash",
		"messages": []map[string]string{
			{
				"role":    "system",
				"content": "You are a helpful assistant in a smart home. Reply with only a short final answer (1-2 sentences). Do not include reasoning steps. Do not attempt to control any devices.",
			},
			{"role": "user", "content": text},
		},
		"max_tokens":  200,
		"temperature": 0.3,
	}

	bodyBytes, _ := json.Marshal(reqBody)
	resp, err := gc.client.Post(gc.baseURL+"/v1/chat/completions", "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("GLM request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("GLM status %d: %s", resp.StatusCode, string(body))
	}

	var llmResp struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&llmResp); err != nil {
		return "", fmt.Errorf("GLM decode: %w", err)
	}

	if len(llmResp.Choices) == 0 {
		return "", fmt.Errorf("GLM returned no choices")
	}

	content := strings.TrimSpace(llmResp.Choices[0].Message.Content)
	if content == "" {
		// GLM-4.7 sometimes puts response in reasoning_content only
		content = strings.TrimSpace(llmResp.Choices[0].Message.ReasoningContent)
	}

	if content == "" {
		return "I'm not sure how to answer that.", nil
	}
	return sanitizeGLMContent(text, content), nil
}

func sanitizeGLMContent(userText, content string) string {
	lower := strings.ToLower(content)
	// Some model variants leak chain-of-thought style traces. Replace with concise final speech.
	if strings.Contains(lower, "the user is asking") || strings.Contains(lower, "analyze the user's request") {
		userLower := strings.ToLower(userText)
		if strings.Contains(userLower, "weather") {
			return "I can help with weather. Tell me your city or zip code."
		}
		return "I can help with that. Could you rephrase it briefly?"
	}

	// Keep only the first non-empty line to avoid long multi-step dumps.
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			return line
		}
	}
	return content
}
