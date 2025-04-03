/// 验证器特征，定义任何验证器必须实现的方法
pub trait Validator<T> {
    /// 验证值是否合法
    ///
    /// 返回一个元组 (bool, String)，第一个元素表示是否有效，第二个元素为消息
    fn validate(&self, value: &T) -> (bool, String);
}

/// 项目名称验证器，用于验证项目名称是否符合要求
#[derive(Default)]
pub struct ProjectNameValidator;

impl Validator<String> for ProjectNameValidator {
    fn validate(&self, name: &String) -> (bool, String) {
        // 检查名称中是否包含空格
        if name.contains(' ') {
            return (
                false,
                "Project name cannot contain spaces. Spaces will cause environment creation to fail.".to_string()
            );
        }

        // 检查其他非法字符
        let has_invalid_chars = name
            .chars()
            .any(|c| !c.is_alphanumeric() && c != '_' && c != '-');
        if has_invalid_chars {
            return (
                false,
                "Project name contains invalid characters. Only alphanumeric characters, underscore, and hyphen are allowed.".to_string()
            );
        }

        // 检查名称是否以字母或数字开头
        if !name.is_empty() && !name.chars().next().unwrap().is_alphanumeric() {
            return (
                false,
                "Project name must start with an alphanumeric character.".to_string(),
            );
        }

        // 检查名称长度
        if name.len() < 2 {
            return (
                false,
                "Project name must be at least 2 characters long.".to_string(),
            );
        }

        (true, "Project name is valid.".to_string())
    }
}

/// Conda环境名称验证器，扩展了普通项目名称验证，添加了Conda环境特定的规则
#[derive(Default)]
pub struct CondaEnvNameValidator;

impl Validator<String> for CondaEnvNameValidator {
    fn validate(&self, name: &String) -> (bool, String) {
        // 首先运行基本的项目名称验证
        let project_validator = ProjectNameValidator;
        let (is_valid, message) = project_validator.validate(name);
        if !is_valid {
            return (false, message);
        }

        // 检查是否符合Conda环境名称的特定规则
        // (在当前实现中，基本规则已经足够严格，可以不添加额外规则)

        (
            true,
            "Project name is valid for Conda environment.".to_string(),
        )
    }
}

/// 帮助函数：将字符串转换为有效的conda环境名称
#[allow(dead_code)]
pub fn sanitize_for_conda_env(name: &str) -> String {
    // 替换空格和无效字符为下划线
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

    // 如果名称以非字母数字字符开头，添加前缀"env_"
    if !sanitized.is_empty() && !sanitized.chars().next().unwrap().is_alphanumeric() {
        format!("env_{}", sanitized)
    } else {
        sanitized
    }
}

// 单元测试
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_name_validator() {
        let validator = ProjectNameValidator;

        // 有效的名称
        assert!(validator.validate(&"valid-name".to_string()).0);
        assert!(validator.validate(&"valid_name123".to_string()).0);

        // 无效的名称
        assert!(!validator.validate(&"invalid name".to_string()).0); // 含空格
        assert!(!validator.validate(&"invalid@name".to_string()).0); // 含特殊字符
        assert!(!validator.validate(&"_invalidname".to_string()).0); // 非字母数字开头
        assert!(!validator.validate(&"a".to_string()).0); // 太短
    }

    #[test]
    fn test_conda_env_name_validator() {
        let validator = CondaEnvNameValidator;

        // 有效的名称
        assert!(validator.validate(&"valid-name".to_string()).0);
        assert!(validator.validate(&"valid_name123".to_string()).0);

        // 无效的名称
        assert!(!validator.validate(&"invalid name".to_string()).0); // 含空格
    }

    #[test]
    fn test_sanitize_for_conda_env() {
        assert_eq!(sanitize_for_conda_env("valid-name"), "valid-name");
        assert_eq!(sanitize_for_conda_env("invalid name"), "invalid_name");
        assert_eq!(sanitize_for_conda_env("@invalid"), "env__invalid");
    }
}

// 创建一个公共导出模块
pub mod exports {
    pub use super::sanitize_for_conda_env;
}
