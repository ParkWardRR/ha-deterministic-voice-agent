package main

import (
	"database/sql"
	"fmt"
	"log"
	"strconv"
	"strings"

	_ "github.com/lib/pq"
)

// Candidate represents an entity match from the catalog.
type Candidate struct {
	ItemID   int64
	Kind     string
	Domain   string
	EntityID string
	Name     string
	Area     string
	Score    float64 // 1.0 = perfect match, 0.0 = no match
}

// Resolver handles entity matching against the pgvector catalog.
type Resolver struct {
	db       *sql.DB
	embedder *EmbedderClient
}

// NewResolver connects to PostgreSQL.
func NewResolver(dsn string, embedder *EmbedderClient) (*Resolver, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("pgvector connect: %w", err)
	}
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("pgvector ping: %w", err)
	}
	db.SetMaxOpenConns(5)
	db.SetMaxIdleConns(2)
	log.Println("Connected to pgvector catalog")
	return &Resolver{db: db, embedder: embedder}, nil
}

// Close closes the DB connection.
func (r *Resolver) Close() {
	r.db.Close()
}

// Ping checks the DB connection.
func (r *Resolver) Ping() error {
	return r.db.Ping()
}

// Resolve finds matching entities for a user utterance.
// It runs lexical match first, then vector search, and merges results.
func (r *Resolver) Resolve(text string) ([]Candidate, error) {
	lower := strings.ToLower(strings.TrimSpace(text))

	// Pass A: Lexical matching
	lexical, err := r.lexicalMatch(lower)
	if err != nil {
		log.Printf("Lexical match error (non-fatal): %v", err)
	}
	lexical = dedupeCandidates(lexical)

	// If we got a strong lexical match (exact entity_id or exact name), return it
	for _, c := range lexical {
		if c.Score >= 0.95 {
			log.Printf("Strong lexical match: %s (%.2f)", c.EntityID, c.Score)
			return []Candidate{c}, nil
		}
	}

	// Pass B: Vector search via embedder + pgvector; if unavailable, fall back to text search.
	var vector []Candidate
	if r.embedder != nil {
		emb, embErr := r.embedder.Embed(lower)
		if embErr != nil {
			log.Printf("Embedder error (falling back to text search): %v", embErr)
		} else {
			vector, err = r.vectorSearch(emb)
			if err != nil {
				log.Printf("Vector search error (falling back to text search): %v", err)
				vector = nil
			}
		}
	}
	if len(vector) == 0 {
		vector, err = r.textSearch(lower)
		if err != nil {
			log.Printf("Text search error (non-fatal): %v", err)
		}
	}
	vector = dedupeCandidates(vector)

	// Merge results, lexical first (higher priority) and keep highest score per entity.
	byID := make(map[string]Candidate, len(lexical)+len(vector))
	order := make([]string, 0, len(lexical)+len(vector))
	for _, c := range lexical {
		if _, ok := byID[c.EntityID]; !ok {
			order = append(order, c.EntityID)
		}
		byID[c.EntityID] = c
	}
	for _, c := range vector {
		if existing, ok := byID[c.EntityID]; ok {
			if c.Score > existing.Score {
				byID[c.EntityID] = c
			}
		} else {
			order = append(order, c.EntityID)
			byID[c.EntityID] = c
		}
	}
	merged := make([]Candidate, 0, len(byID))
	for _, id := range order {
		if c, ok := byID[id]; ok {
			merged = append(merged, c)
		}
	}

	// Sort by score descending
	for i := 0; i < len(merged); i++ {
		for j := i + 1; j < len(merged); j++ {
			if merged[j].Score > merged[i].Score {
				merged[i], merged[j] = merged[j], merged[i]
			}
		}
	}

	// Limit to top 8
	if len(merged) > 8 {
		merged = merged[:8]
	}

	return merged, nil
}

func (r *Resolver) vectorSearch(embedding []float64) ([]Candidate, error) {
	if len(embedding) == 0 {
		return nil, nil
	}

	vectorLiteral := formatVector(embedding)
	rows, err := r.db.Query(`
		SELECT item_id, kind, domain, entity_id, name, COALESCE(area, ''),
		       1 - (embedding <=> $1::vector) AS score
		FROM catalog_items
		WHERE enabled = true
		ORDER BY embedding <=> $1::vector
		LIMIT 8
	`, vectorLiteral)
	if err != nil {
		return nil, fmt.Errorf("vector query: %w", err)
	}
	defer rows.Close()

	var candidates []Candidate
	for rows.Next() {
		var c Candidate
		if err := rows.Scan(&c.ItemID, &c.Kind, &c.Domain, &c.EntityID, &c.Name, &c.Area, &c.Score); err != nil {
			return nil, fmt.Errorf("vector scan: %w", err)
		}
		candidates = append(candidates, c)
	}
	return candidates, nil
}

func formatVector(v []float64) string {
	var b strings.Builder
	b.Grow(len(v) * 8)
	b.WriteByte('[')
	for i, n := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(n, 'f', -1, 64))
	}
	b.WriteByte(']')
	return b.String()
}

// lexicalMatch performs exact and fuzzy text matching against the catalog.
func (r *Resolver) lexicalMatch(text string) ([]Candidate, error) {
	var candidates []Candidate

	// Exact entity_id match
	rows, err := r.db.Query(`
		SELECT item_id, kind, domain, entity_id, name, COALESCE(area, '')
		FROM catalog_items
		WHERE enabled = true AND entity_id = $1
	`, text)
	if err == nil {
		defer rows.Close()
		for rows.Next() {
			var c Candidate
			rows.Scan(&c.ItemID, &c.Kind, &c.Domain, &c.EntityID, &c.Name, &c.Area)
			c.Score = 1.0
			candidates = append(candidates, c)
		}
	}

	// Name match (case-insensitive, contains)
	words := extractKeywords(text)
	if len(words) > 0 {
		// Build a LIKE pattern from important words
		for _, word := range words {
			if len(word) < 3 {
				continue
			}
			rows, err := r.db.Query(`
				SELECT item_id, kind, domain, entity_id, name, COALESCE(area, '')
				FROM catalog_items
				WHERE enabled = true AND lower(name) LIKE '%' || $1 || '%'
				LIMIT 10
			`, word)
			if err != nil {
				continue
			}
			for rows.Next() {
				var c Candidate
				rows.Scan(&c.ItemID, &c.Kind, &c.Domain, &c.EntityID, &c.Name, &c.Area)
				// Score based on how well the name matches
				nameLower := strings.ToLower(c.Name)
				if nameLower == text {
					c.Score = 0.95
				} else if strings.Contains(nameLower, text) {
					c.Score = 0.85
				} else {
					c.Score = 0.6 + 0.1*float64(len(word))/float64(len(nameLower))
				}
				candidates = append(candidates, c)
			}
			rows.Close()
		}
	}

	return candidates, nil
}

// textSearch does a broader text-based search for entities.
func (r *Resolver) textSearch(text string) ([]Candidate, error) {
	var candidates []Candidate

	// Search by matching any of the significant words against name, domain, area
	words := extractKeywords(text)
	if len(words) == 0 {
		return nil, nil
	}

	// Build OR conditions for each word
	conditions := make([]string, 0, len(words))
	args := make([]interface{}, 0, len(words))
	for i, w := range words {
		if len(w) < 3 {
			continue
		}
		conditions = append(conditions, fmt.Sprintf(
			"(lower(name) LIKE '%%' || $%d || '%%' OR lower(domain) = $%d OR lower(COALESCE(area,'')) LIKE '%%' || $%d || '%%')",
			i+1, i+1, i+1,
		))
		args = append(args, w)
	}

	if len(conditions) == 0 {
		return nil, nil
	}

	query := fmt.Sprintf(`
		SELECT item_id, kind, domain, entity_id, name, COALESCE(area, '')
		FROM catalog_items
		WHERE enabled = true AND (%s)
		LIMIT 20
	`, strings.Join(conditions, " OR "))

	rows, err := r.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("text search: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var c Candidate
		rows.Scan(&c.ItemID, &c.Kind, &c.Domain, &c.EntityID, &c.Name, &c.Area)
		// Score based on word overlap
		nameLower := strings.ToLower(c.Name)
		matchCount := 0
		for _, w := range words {
			if strings.Contains(nameLower, w) || strings.ToLower(c.Domain) == w {
				matchCount++
			}
		}
		c.Score = 0.4 + 0.15*float64(matchCount)
		candidates = append(candidates, c)
	}

	return candidates, nil
}

// extractKeywords pulls meaningful words from user text.
func extractKeywords(text string) []string {
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "to": true, "in": true,
		"on": true, "off": true, "turn": true, "set": true, "make": true,
		"please": true, "can": true, "you": true, "my": true, "me": true,
		"and": true, "or": true, "but": true, "is": true, "it": true,
		"of": true, "for": true, "with": true, "at": true, "from": true,
		"up": true, "down": true, "do": true, "this": true, "that": true,
		"what": true, "how": true, "all": true, "i": true, "hey": true,
	}

	words := strings.Fields(strings.ToLower(text))
	var keywords []string
	for _, w := range words {
		// Strip punctuation
		w = strings.TrimFunc(w, func(r rune) bool {
			return r < 'a' || r > 'z'
		})
		if len(w) > 0 && !stopWords[w] {
			keywords = append(keywords, w)
		}
	}
	return keywords
}

func dedupeCandidates(candidates []Candidate) []Candidate {
	if len(candidates) < 2 {
		return candidates
	}
	seen := make(map[string]Candidate, len(candidates))
	order := make([]string, 0, len(candidates))
	for _, c := range candidates {
		if current, ok := seen[c.EntityID]; ok {
			if c.Score > current.Score {
				seen[c.EntityID] = c
			}
			continue
		}
		order = append(order, c.EntityID)
		seen[c.EntityID] = c
	}

	out := make([]Candidate, 0, len(seen))
	for _, id := range order {
		if c, ok := seen[id]; ok {
			out = append(out, c)
		}
	}
	return out
}
