use clap::Parser;
use log::info;
use std::path::PathBuf;
use toml::Value;

use crate::error::Result;

#[derive(Parser, Debug)]
pub struct DiffCommand {
    /// First CRESP configuration file
    #[arg(short, long)]
    path1: PathBuf,

    /// Second CRESP configuration file
    #[arg(short, long)]
    path2: PathBuf,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    format: String,

    /// Ignore whitespace differences
    #[arg(short, long)]
    ignore_whitespace: bool,
}

impl DiffCommand {
    pub async fn execute(&self) -> Result<()> {
        info!("🔍 Comparing CRESP configurations...");

        // Read and parse TOML files
        let contents1 = std::fs::read_to_string(&self.path1)?;
        let contents2 = std::fs::read_to_string(&self.path2)?;

        let config1: Value = toml::from_str(&contents1)?;
        let config2: Value = toml::from_str(&contents2)?;

        // Compare configurations
        let differences = self.compare_configurations(&config1, &config2)?;

        // Output differences based on format
        match self.format.to_lowercase().as_str() {
            "text" => self.output_text_diff(&differences)?,
            "json" => self.output_json_diff(&differences)?,
            _ => {
                return Err(crate::error::Error::Config(format!(
                    "Unsupported output format: {}",
                    self.format
                )))
            }
        }

        Ok(())
    }

    fn compare_configurations(&self, config1: &Value, config2: &Value) -> Result<Vec<Difference>> {
        let mut differences = Vec::new();

        // Compare version
        if let Some(version1) = config1.get("cresp_version").and_then(|v| v.as_str()) {
            if let Some(version2) = config2.get("cresp_version").and_then(|v| v.as_str()) {
                if version1 != version2 {
                    differences.push(Difference {
                        path: "cresp_version".to_string(),
                        type_: DifferenceType::Value,
                        old_value: Some(version1.to_string()),
                        new_value: Some(version2.to_string()),
                    });
                }
            }
        }

        // Compare experiment section
        if let Some(exp1) = config1.get("experiment") {
            if let Some(exp2) = config2.get("experiment") {
                self.compare_section(exp1, exp2, "experiment", &mut differences)?;
            }
        }

        Ok(differences)
    }

    fn compare_section(
        &self,
        value1: &Value,
        value2: &Value,
        path: &str,
        differences: &mut Vec<Difference>,
    ) -> Result<()> {
        match (value1, value2) {
            (Value::Table(table1), Value::Table(table2)) => {
                for (key, value1) in table1 {
                    if let Some(value2) = table2.get(key) {
                        let new_path = format!("{}.{}", path, key);
                        self.compare_values(value1, value2, &new_path, differences)?;
                    } else {
                        let new_path = format!("{}.{}", path, key);
                        differences.push(Difference {
                            path: new_path,
                            type_: DifferenceType::Missing,
                            old_value: Some(value1.to_string()),
                            new_value: None,
                        });
                    }
                }

                for (key, value2) in table2 {
                    if !table1.contains_key(key) {
                        let new_path = format!("{}.{}", path, key);
                        differences.push(Difference {
                            path: new_path,
                            type_: DifferenceType::Added,
                            old_value: None,
                            new_value: Some(value2.to_string()),
                        });
                    }
                }
            }
            (Value::Array(arr1), Value::Array(arr2)) => {
                for (i, (value1, value2)) in arr1.iter().zip(arr2.iter()).enumerate() {
                    let new_path = format!("{}[{}]", path, i);
                    self.compare_values(value1, value2, &new_path, differences)?;
                }

                if arr1.len() != arr2.len() {
                    differences.push(Difference {
                        path: path.to_string(),
                        type_: DifferenceType::Length,
                        old_value: Some(arr1.len().to_string()),
                        new_value: Some(arr2.len().to_string()),
                    });
                }
            }
            (value1, value2) => {
                let str1 = value1.to_string();
                let str2 = value2.to_string();
                if str1 != str2 {
                    differences.push(Difference {
                        path: path.to_string(),
                        type_: DifferenceType::Value,
                        old_value: Some(str1),
                        new_value: Some(str2),
                    });
                }
            }
        }

        Ok(())
    }

    fn compare_values(
        &self,
        value1: &Value,
        value2: &Value,
        path: &str,
        differences: &mut Vec<Difference>,
    ) -> Result<()> {
        match (value1, value2) {
            (Value::Table(_table1), Value::Table(_table2)) => {
                self.compare_section(value1, value2, path, differences)?;
            }
            (Value::Array(_arr1), Value::Array(_arr2)) => {
                self.compare_section(value1, value2, path, differences)?;
            }
            (value1, value2) => {
                let str1 = value1.to_string();
                let str2 = value2.to_string();
                if str1 != str2 {
                    differences.push(Difference {
                        path: path.to_string(),
                        type_: DifferenceType::Value,
                        old_value: Some(str1),
                        new_value: Some(str2),
                    });
                }
            }
        }

        Ok(())
    }

    fn output_text_diff(&self, differences: &[Difference]) -> Result<()> {
        if differences.is_empty() {
            info!("✅ Configurations are identical");
            return Ok(());
        }

        info!("📝 Found {} differences:", differences.len());
        for diff in differences {
            match diff.type_ {
                DifferenceType::Value => {
                    println!(
                        "  {}: {} -> {}",
                        diff.path,
                        diff.old_value.as_deref().unwrap_or(""),
                        diff.new_value.as_deref().unwrap_or("")
                    );
                }
                DifferenceType::Missing => {
                    println!(
                        "  - {}: {}",
                        diff.path,
                        diff.old_value.as_deref().unwrap_or("")
                    );
                }
                DifferenceType::Added => {
                    println!(
                        "  + {}: {}",
                        diff.path,
                        diff.new_value.as_deref().unwrap_or("")
                    );
                }
                DifferenceType::Length => {
                    println!(
                        "  {}: Array length changed from {} to {}",
                        diff.path,
                        diff.old_value.as_deref().unwrap_or(""),
                        diff.new_value.as_deref().unwrap_or("")
                    );
                }
            }
        }

        Ok(())
    }

    fn output_json_diff(&self, differences: &[Difference]) -> Result<()> {
        let json = serde_json::to_string_pretty(&differences)?;
        println!("{}", json);
        Ok(())
    }
}

#[derive(Debug, serde::Serialize)]
struct Difference {
    path: String,
    #[serde(rename = "type")]
    type_: DifferenceType,
    old_value: Option<String>,
    new_value: Option<String>,
}

#[derive(Debug, serde::Serialize)]
enum DifferenceType {
    Value,
    Missing,
    Added,
    Length,
}
