package main

import "strings"

// SafetyGate controls which domains are allowed and which need confirmation.
type SafetyGate struct {
	allowed               map[string]bool
	blocked               map[string]bool
	confirmOnly           map[string]bool
	confirmEntityKeywords []string
}

// NewSafetyGate creates a new safety gate with default policies.
func NewSafetyGate() *SafetyGate {
	return &SafetyGate{
		allowed: map[string]bool{
			"light":         true,
			"switch":        true,
			"cover":         true,
			"climate":       true,
			"media_player":  true,
			"fan":           true,
			"scene":         true,
			"script":        true,
			"automation":    true,
			"input_boolean": true,
			"input_select":  true,
			"input_number":  true,
			"input_text":    true,
			"vacuum":        true,
			"humidifier":    true,
			"water_heater":  true,
			"siren":         true,
			"remote":        true,
			"button":        true,
			"number":        true,
			"select":        true,
		},
		blocked: map[string]bool{
			"shell_command":           true,
			"rest_command":            true,
			"notify":                  true,
			"persistent_notification": true,
			"homeassistant":           true,
		},
		confirmOnly: map[string]bool{
			"lock":                true,
			"alarm_control_panel": true,
			"valve":               true,
		},
		confirmEntityKeywords: []string{
			"garage",
		},
	}
}

// IsAllowed returns true if the domain can be acted on.
func (sg *SafetyGate) IsAllowed(domain string) bool {
	if sg.blocked[domain] {
		return false
	}
	return sg.allowed[domain]
}

// NeedsConfirmation returns true if the domain requires explicit confirmation.
func (sg *SafetyGate) NeedsConfirmation(domain string) bool {
	return sg.confirmOnly[domain]
}

// NeedsEntityConfirmation enforces confirmation for risky targets identified by name/entity_id.
func (sg *SafetyGate) NeedsEntityConfirmation(entityID string) bool {
	lower := strings.ToLower(entityID)
	for _, keyword := range sg.confirmEntityKeywords {
		if strings.Contains(lower, keyword) {
			return true
		}
	}
	return false
}
