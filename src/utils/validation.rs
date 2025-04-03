/// Validator trait that defines methods any validator must implement
pub trait Validator<T> {
    /// Validate if a value is valid
    ///
    /// Returns a tuple (bool, String), where the first element indicates validity and the second is a message
    fn validate(&self, value: &T) -> (bool, String);
}

/// Project name validator, used to validate if a project name meets requirements
#[derive(Default)]
pub struct ProjectNameValidator;

impl Validator<String> for ProjectNameValidator {
    fn validate(&self, name: &String) -> (bool, String) {
        // Check if name contains spaces
        if name.contains(' ') {
            return (
                false,
                "Project name cannot contain spaces. Spaces will cause environment creation to fail.".to_string()
            );
        }

        // Check for other invalid characters
        let has_invalid_chars = name
            .chars()
            .any(|c| !c.is_alphanumeric() && c != '_' && c != '-');
        if has_invalid_chars {
            return (
                false,
                "Project name contains invalid characters. Only alphanumeric characters, underscore, and hyphen are allowed.".to_string()
            );
        }

        // Check if name starts with alphanumeric character
        if !name.is_empty() && !name.chars().next().unwrap().is_alphanumeric() {
            return (
                false,
                "Project name must start with an alphanumeric character.".to_string(),
            );
        }

        // Check name length
        if name.len() < 2 {
            return (
                false,
                "Project name must be at least 2 characters long.".to_string(),
            );
        }

        (true, "Project name is valid.".to_string())
    }
}

/// Conda environment name validator, extends basic project name validation with Conda-specific rules
#[derive(Default)]
pub struct CondaEnvNameValidator;

impl Validator<String> for CondaEnvNameValidator {
    fn validate(&self, name: &String) -> (bool, String) {
        // First run the basic project name validation
        let project_validator = ProjectNameValidator;
        let (is_valid, message) = project_validator.validate(name);
        if !is_valid {
            return (false, message);
        }

        // Check if it meets Conda environment name specific rules
        // (In the current implementation, the basic rules are already strict enough)

        (
            true,
            "Project name is valid for Conda environment.".to_string(),
        )
    }
}

/// Helper function: Convert a string to a valid conda environment name
#[allow(dead_code)]
pub fn sanitize_for_conda_env(name: &str) -> String {
    // Replace spaces and invalid characters with underscores
    let sanitized = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>();

    // If name begins with a non-alphanumeric character, prefix with "env_"
    if !sanitized.is_empty() && !sanitized.chars().next().unwrap().is_alphanumeric() {
        format!("env_{}", sanitized)
    } else {
        sanitized
    }
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_name_validator() {
        let validator = ProjectNameValidator;

        // Valid names
        assert!(validator.validate(&"valid-name".to_string()).0);
        assert!(validator.validate(&"valid_name123".to_string()).0);

        // Invalid names
        assert!(!validator.validate(&"invalid name".to_string()).0); // Contains spaces
        assert!(!validator.validate(&"invalid@name".to_string()).0); // Contains special chars
        assert!(!validator.validate(&"_invalidname".to_string()).0); // Non-alphanumeric start
        assert!(!validator.validate(&"a".to_string()).0); // Too short
    }

    #[test]
    fn test_conda_env_name_validator() {
        let validator = CondaEnvNameValidator;

        // Valid names
        assert!(validator.validate(&"valid-name".to_string()).0);
        assert!(validator.validate(&"valid_name123".to_string()).0);

        // Invalid names
        assert!(!validator.validate(&"invalid name".to_string()).0); // Contains spaces
    }

    #[test]
    fn test_sanitize_for_conda_env() {
        assert_eq!(sanitize_for_conda_env("valid-name"), "valid-name");
        assert_eq!(sanitize_for_conda_env("invalid name"), "invalid_name");
        assert_eq!(sanitize_for_conda_env("@invalid"), "env__invalid");
    }
}

// Create a public exports module
pub mod exports {
    pub use super::sanitize_for_conda_env;
}
