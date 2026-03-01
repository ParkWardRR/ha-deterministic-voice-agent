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

// EmbedderClient generates text embeddings used for pgvector retrieval.
type EmbedderClient struct {
	baseURL string
	client  *http.Client
}

// NewEmbedderClient creates a new embedder client; empty baseURL disables embedding lookup.
func NewEmbedderClient(baseURL string) *EmbedderClient {
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" {
		return nil
	}
	return &EmbedderClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		client:  &http.Client{Timeout: 20 * time.Second},
	}
}

// Embed returns an embedding vector for input text.
// Supported responses:
// - OpenAI style: {"data":[{"embedding":[...]}]}
// - Simple style: {"embedding":[...]}
func (ec *EmbedderClient) Embed(text string) ([]float64, error) {
	reqBody := map[string]interface{}{
		"input": text,
	}
	body, _ := json.Marshal(reqBody)

	resp, err := ec.client.Post(ec.baseURL+"/v1/embeddings", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("embedder request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedder status %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("embedder read: %w", err)
	}

	var openAIStyle struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &openAIStyle); err == nil && len(openAIStyle.Data) > 0 {
		if len(openAIStyle.Data[0].Embedding) > 0 {
			return openAIStyle.Data[0].Embedding, nil
		}
	}

	var simple struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.Unmarshal(raw, &simple); err == nil && len(simple.Embedding) > 0 {
		return simple.Embedding, nil
	}

	return nil, fmt.Errorf("embedder response missing embedding")
}
