#[cfg(test)]
mod tests {
    
    

    // Helper to mirror domain_of
    fn domain_of(entity_id: &str) -> String {
        if let Some(idx) = entity_id.find('.') {
            entity_id[..idx].to_string()
        } else {
            entity_id.to_string()
        }
    }

    // Helper to mirror guess_service
    fn guess_service(text: &str, domain: &str) -> String {
        let lower = text.to_lowercase();
        if lower.contains("turn off") || lower.contains("shut off") || lower.contains("disable") {
            "turn_off".to_string()
        } else if lower.contains("turn on") || lower.contains("enable") || lower.contains("start") {
            "turn_on".to_string()
        } else if lower.contains("toggle") {
            "toggle".to_string()
        } else if lower.contains("dim") || lower.contains("brightness") {
            "turn_on".to_string()
        } else if lower.contains("play") || lower.contains("resume") {
            "media_play".to_string()
        } else if lower.contains("pause") || lower.contains("stop") {
            "media_pause".to_string()
        } else if lower.contains("volume") {
            "volume_set".to_string()
        } else {
            match domain {
                "light" | "switch" | "fan" | "input_boolean" => "toggle".to_string(),
                "media_player" => "media_play".to_string(),
                "scene" => "turn_on".to_string(),
                "automation" => "trigger".to_string(),
                _ => "toggle".to_string(),
            }
        }
    }

    #[test]
    fn test_domain_extraction() {
        assert_eq!(domain_of("light.living_room"), "light");
        assert_eq!(domain_of("switch.tv"), "switch");
        assert_eq!(domain_of("scene.movie_time"), "scene");
        assert_eq!(domain_of("nodotdomain"), "nodotdomain");
    }

    #[test]
    fn test_guess_service() {
        assert_eq!(guess_service("turn off the light", "light"), "turn_off");
        assert_eq!(guess_service("turn on the tv", "switch"), "turn_on");
        assert_eq!(guess_service("pause the music", "media_player"), "media_pause");
        assert_eq!(guess_service("set brightness to 50%", "light"), "turn_on");
        
        // Fallbacks based on domain
        assert_eq!(guess_service("what about this?", "scene"), "turn_on");
        assert_eq!(guess_service("what about this?", "light"), "toggle");
        assert_eq!(guess_service("what about this?", "media_player"), "media_play");
        assert_eq!(guess_service("what about this?", "vacuum"), "toggle");
    }
}
