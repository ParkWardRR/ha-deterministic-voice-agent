#[cfg(test)]
mod tests {
    use crate::safety::SafetyGate;

    #[test]
    fn test_safety_gate_allowed_domains() {
        let gate = SafetyGate::new();
        assert!(gate.is_allowed("light"));
        assert!(gate.is_allowed("switch"));
        assert!(gate.is_allowed("climate"));
        assert!(gate.is_allowed("media_player"));
        assert!(gate.is_allowed("vacuum"));

        // Unknown domains mapped strictly
        assert!(!gate.is_allowed("unknown_domain"));
    }

    #[test]
    fn test_safety_gate_blocked_domains() {
        let gate = SafetyGate::new();
        assert!(!gate.is_allowed("shell_command"));
        assert!(!gate.is_allowed("rest_command"));
        assert!(!gate.is_allowed("homeassistant"));
    }

    #[test]
    fn test_safety_gate_confirmation() {
        let gate = SafetyGate::new();
        assert!(gate.needs_confirmation("lock"));
        assert!(gate.needs_confirmation("alarm_control_panel"));
        assert!(gate.needs_confirmation("valve"));
        
        assert!(!gate.needs_confirmation("light"));
        assert!(!gate.needs_confirmation("switch"));
    }

    #[test]
    fn test_safety_gate_entity_keywords() {
        let gate = SafetyGate::new();
        
        // Allowed domain, but has risky keyword -> needs entity conf
        assert!(gate.needs_entity_confirmation("cover.main_garage_door"));
        assert!(gate.needs_entity_confirmation("switch.garage_lights"));
        
        // Safe entities
        assert!(!gate.needs_entity_confirmation("light.living_room"));
        assert!(!gate.needs_entity_confirmation("switch.tv_plug"));
    }
}
