use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

#[derive(Default)]
pub struct SystemInfo {
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub gpu: GpuInfo,
    pub storage: StorageInfo,
    pub network: NetworkInfo,
    pub os: OsInfo,
    pub limits: SystemLimits,
    pub packages: Vec<String>,
    pub software: HashMap<String, String>,
    pub cuda: CudaInfo,
}

#[derive(Default)]
pub struct CpuInfo {
    pub model: String,
    pub architecture: String,
    pub cores: u32,
    pub threads: u32,
    pub frequency: String,
}

#[derive(Default)]
pub struct MemoryInfo {
    pub size: String,
    pub memory_type: String,
}

#[derive(Default)]
pub struct GpuInfo {
    pub model: String,
    pub memory: String,
    pub compute_capability: String,
    pub driver_version: String,
}

#[derive(Default)]
pub struct StorageInfo {
    pub storage_type: String,
}

#[derive(Default)]
pub struct NetworkInfo {
    pub network_type: String,
    pub bandwidth: String,
}

#[derive(Default)]
pub struct OsInfo {
    pub name: String,
    pub version: String,
    pub kernel: String,
    pub architecture: String,
    pub locale: String,
    pub timezone: String,
}

#[derive(Default)]
pub struct SystemLimits {
    pub max_open_files: u64,
    pub max_processes: u64,
    pub stack_size: String,
    pub virtual_memory: String,
}

#[derive(Default)]
pub struct CudaInfo {
    pub cuda_home: String,
    pub ld_library_path: Vec<String>,
    pub cupti_path: String,
}

/// 收集系统信息
pub fn collect_system_info() -> Result<SystemInfo> {
    let mut info = SystemInfo::default();

    // 获取 CPU 信息
    collect_cpu_info(&mut info)?;

    // 获取内存信息
    collect_memory_info(&mut info)?;

    // 获取 GPU 信息
    collect_gpu_info(&mut info)?;

    // 获取存储信息
    collect_storage_info(&mut info)?;

    // 获取网络信息
    collect_network_info(&mut info)?;

    // 获取操作系统信息
    collect_os_info(&mut info)?;

    // 获取系统限制
    collect_system_limits(&mut info)?;

    // 获取已安装的包
    collect_installed_packages(&mut info)?;

    // 获取软件版本
    collect_software_versions(&mut info)?;

    // 获取 CUDA 路径
    collect_cuda_paths(&mut info)?;

    Ok(info)
}

fn collect_cpu_info(info: &mut SystemInfo) -> Result<()> {
    if cfg!(target_os = "linux") {
        let cpu_info = Command::new("lscpu").output()?.stdout;
        let cpu_info = String::from_utf8_lossy(&cpu_info);

        info.cpu.model = cpu_info
            .lines()
            .find(|line| line.starts_with("Model name:"))
            .map(|line| line.split(":").nth(1).unwrap_or("").trim().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        info.cpu.architecture = cpu_info
            .lines()
            .find(|line| line.starts_with("Architecture:"))
            .map(|line| line.split(":").nth(1).unwrap_or("").trim().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        info.cpu.cores = cpu_info
            .lines()
            .find(|line| line.starts_with("CPU(s):"))
            .and_then(|line| line.split(":").nth(1).unwrap_or("").trim().parse().ok())
            .unwrap_or(1);

        info.cpu.threads = cpu_info
            .lines()
            .find(|line| line.starts_with("Thread(s) per core:"))
            .and_then(|line| line.split(":").nth(1).unwrap_or("").trim().parse().ok())
            .unwrap_or(1)
            * info.cpu.cores;

        info.cpu.frequency = cpu_info
            .lines()
            .find(|line| line.starts_with("CPU MHz:"))
            .map(|line| format!("{}MHz", line.split(":").nth(1).unwrap_or("").trim()))
            .unwrap_or_else(|| "Unknown".to_string());
    } else if cfg!(target_os = "macos") {
        let cpu_info = Command::new("sysctl").arg("machdep.cpu").output()?.stdout;
        let cpu_info = String::from_utf8_lossy(&cpu_info);

        info.cpu.model = cpu_info
            .lines()
            .find(|line| line.starts_with("machdep.cpu.brand_string:"))
            .map(|line| line.split(":").nth(1).unwrap_or("").trim().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        info.cpu.architecture = "x86_64".to_string();

        info.cpu.cores = cpu_info
            .lines()
            .find(|line| line.starts_with("machdep.cpu.core_count:"))
            .and_then(|line| line.split(":").nth(1).unwrap_or("").trim().parse().ok())
            .unwrap_or(1);

        info.cpu.threads = cpu_info
            .lines()
            .find(|line| line.starts_with("machdep.cpu.thread_count:"))
            .and_then(|line| line.split(":").nth(1).unwrap_or("").trim().parse().ok())
            .unwrap_or(1);

        info.cpu.frequency = cpu_info
            .lines()
            .find(|line| line.starts_with("machdep.cpu.maxspeed:"))
            .map(|line| format!("{}MHz", line.split(":").nth(1).unwrap_or("").trim()))
            .unwrap_or_else(|| "Unknown".to_string());
    }

    Ok(())
}

fn collect_memory_info(info: &mut SystemInfo) -> Result<()> {
    if cfg!(target_os = "linux") {
        let mem_info = Command::new("free").arg("-h").output()?.stdout;
        let mem_info = String::from_utf8_lossy(&mem_info);

        info.memory.size = mem_info
            .lines()
            .nth(1)
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("Unknown")
            .to_string();

        info.memory.memory_type = "DDR4/DDR5".to_string(); // This is hard to detect
    } else if cfg!(target_os = "macos") {
        let mem_info = Command::new("sysctl").arg("hw.memsize").output()?.stdout;
        info.memory.size = String::from_utf8_lossy(&mem_info)
            .split(":")
            .nth(1)
            .map(|s| {
                let bytes = s.trim().parse::<u64>().unwrap_or(0);
                format!("{} GB", bytes / 1_073_741_824) // Convert bytes to GB
            })
            .unwrap_or_else(|| "Unknown".to_string());

        info.memory.memory_type = "DDR4/DDR5".to_string(); // This is hard to detect
    }

    Ok(())
}

fn collect_gpu_info(info: &mut SystemInfo) -> Result<()> {
    if cfg!(target_os = "linux") {
        if let Ok(nvidia_smi) = Command::new("nvidia-smi")
            .arg("--query-gpu=gpu_name,memory.total,compute_cap,driver_version")
            .arg("--format=csv,noheader")
            .output()
        {
            let gpu_info = String::from_utf8_lossy(&nvidia_smi.stdout);
            if let Some(line) = gpu_info.lines().next() {
                let parts: Vec<&str> = line.split(", ").collect();
                if parts.len() >= 4 {
                    info.gpu.model = parts[0].to_string();
                    info.gpu.memory = parts[1].to_string();
                    info.gpu.compute_capability = parts[2].to_string();
                    info.gpu.driver_version = parts[3].to_string();
                }
            }
        }
    } else if cfg!(target_os = "macos") {
        if let Ok(system_profiler) = Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
        {
            let gpu_info = String::from_utf8_lossy(&system_profiler.stdout);
            info.gpu.model = gpu_info
                .lines()
                .find(|line| line.contains("Chipset Model:"))
                .map(|line| line.split(":").nth(1).unwrap_or("").trim().to_string())
                .unwrap_or_else(|| "Unknown".to_string());
            info.gpu.memory = "Integrated".to_string();
            info.gpu.compute_capability = "Unknown".to_string();
            info.gpu.driver_version = "Unknown".to_string();
        }
    }

    Ok(())
}

fn collect_storage_info(info: &mut SystemInfo) -> Result<()> {
    if cfg!(target_os = "linux") {
        let storage_info = Command::new("lsblk")
            .arg("-d")
            .arg("-o")
            .arg("NAME,SIZE,MODEL")
            .output()?
            .stdout;
        let storage_info = String::from_utf8_lossy(&storage_info);

        info.storage.storage_type = storage_info
            .lines()
            .nth(1)
            .and_then(|line| line.split_whitespace().nth(2))
            .unwrap_or("Unknown")
            .to_string();
    } else if cfg!(target_os = "macos") {
        let storage_info = Command::new("diskutil")
            .arg("info")
            .arg("/")
            .output()?
            .stdout;
        let storage_info = String::from_utf8_lossy(&storage_info);

        info.storage.storage_type = storage_info
            .lines()
            .find(|line| line.contains("Media Type:"))
            .map(|line| line.split(":").nth(1).unwrap_or("").trim().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
    }

    Ok(())
}

fn collect_network_info(info: &mut SystemInfo) -> Result<()> {
    if cfg!(target_os = "linux") {
        let network_info = Command::new("ethtool")
            .arg("eth0")
            .output()
            .ok()
            .map(|output| String::from_utf8_lossy(&output.stdout).to_string());

        info.network.network_type = network_info
            .as_ref()
            .and_then(|info| {
                info.lines()
                    .find(|line| line.contains("Supported link modes"))
            })
            .map(|line| line.split(":").nth(1).unwrap_or("").trim().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        info.network.bandwidth = network_info
            .as_ref()
            .and_then(|info| info.lines().find(|line| line.contains("Speed:")))
            .map(|line| line.split(":").nth(1).unwrap_or("").trim().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
    } else if cfg!(target_os = "macos") {
        let network_info = Command::new("networksetup")
            .arg("-getinfo")
            .arg("Wi-Fi")
            .output()
            .ok()
            .map(|output| String::from_utf8_lossy(&output.stdout).to_string());

        info.network.network_type = "Wi-Fi".to_string();
        info.network.bandwidth = network_info
            .as_ref()
            .and_then(|info| info.lines().find(|line| line.contains("Speed:")))
            .map(|line| line.split(":").nth(1).unwrap_or("").trim().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
    }

    Ok(())
}

fn collect_os_info(info: &mut SystemInfo) -> Result<()> {
    if cfg!(target_os = "linux") {
        let os_info = Command::new("cat").arg("/etc/os-release").output()?.stdout;
        let os_info = String::from_utf8_lossy(&os_info);

        info.os.name = os_info
            .lines()
            .find(|line| line.starts_with("NAME="))
            .map(|line| {
                line.split("=")
                    .nth(1)
                    .unwrap_or("")
                    .trim_matches('"')
                    .to_string()
            })
            .unwrap_or_else(|| "Unknown".to_string());

        info.os.version = os_info
            .lines()
            .find(|line| line.starts_with("VERSION="))
            .map(|line| {
                line.split("=")
                    .nth(1)
                    .unwrap_or("")
                    .trim_matches('"')
                    .to_string()
            })
            .unwrap_or_else(|| "Unknown".to_string());

        info.os.kernel = Command::new("uname")
            .arg("-r")
            .output()?
            .stdout
            .iter()
            .map(|&b| b as char)
            .collect::<String>()
            .trim()
            .to_string();

        info.os.architecture = Command::new("uname")
            .arg("-m")
            .output()?
            .stdout
            .iter()
            .map(|&b| b as char)
            .collect::<String>()
            .trim()
            .to_string();
    } else if cfg!(target_os = "macos") {
        let os_info = Command::new("sw_vers").output()?.stdout;
        let os_info = String::from_utf8_lossy(&os_info);

        info.os.name = os_info
            .lines()
            .find(|line| line.starts_with("ProductName:"))
            .map(|line| line.split(":").nth(1).unwrap_or("").trim().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        info.os.version = os_info
            .lines()
            .find(|line| line.starts_with("ProductVersion:"))
            .map(|line| line.split(":").nth(1).unwrap_or("").trim().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        info.os.kernel = Command::new("uname")
            .arg("-r")
            .output()?
            .stdout
            .iter()
            .map(|&b| b as char)
            .collect::<String>()
            .trim()
            .to_string();

        info.os.architecture = Command::new("uname")
            .arg("-m")
            .output()?
            .stdout
            .iter()
            .map(|&b| b as char)
            .collect::<String>()
            .trim()
            .to_string();
    }

    // 获取 locale 和 timezone
    info.os.locale = Command::new("locale")
        .arg("LANG")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .unwrap_or_else(|| "en_US.UTF-8".to_string())
        .trim()
        .to_string();

    info.os.timezone = Command::new("date")
        .arg("+%Z")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .unwrap_or_else(|| "UTC".to_string())
        .trim()
        .to_string();

    Ok(())
}

fn collect_system_limits(info: &mut SystemInfo) -> Result<()> {
    if cfg!(target_os = "linux") {
        let limits = Command::new("ulimit").arg("-n").output()?.stdout;
        info.limits.max_open_files = String::from_utf8_lossy(&limits)
            .split_whitespace()
            .next()
            .unwrap_or("65535")
            .parse()
            .unwrap_or(65535);

        let limits = Command::new("ulimit").arg("-u").output()?.stdout;
        info.limits.max_processes = String::from_utf8_lossy(&limits)
            .split_whitespace()
            .next()
            .unwrap_or("32768")
            .parse()
            .unwrap_or(32768);

        let limits = Command::new("ulimit").arg("-s").output()?.stdout;
        info.limits.stack_size = format!(
            "{}K",
            String::from_utf8_lossy(&limits)
                .split_whitespace()
                .next()
                .unwrap_or("8192")
                .parse::<u64>()
                .unwrap_or(8192)
        );

        info.limits.virtual_memory = "unlimited".to_string();
    } else {
        // macOS default values
        info.limits.max_open_files = 12288;
        info.limits.max_processes = 2500;
        info.limits.stack_size = "8192K".to_string();
        info.limits.virtual_memory = "unlimited".to_string();
    }

    Ok(())
}

fn collect_installed_packages(info: &mut SystemInfo) -> Result<()> {
    if cfg!(target_os = "linux") {
        if let Ok(output) = Command::new("dpkg").arg("-l").output() {
            let packages = String::from_utf8_lossy(&output.stdout);
            info.packages = packages
                .lines()
                .filter(|line| line.starts_with("ii"))
                .take(50) // 限制包数量
                .map(|line| {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 {
                        format!("{{ name = \"{}\", version = \"{}\" }}", parts[1], parts[2])
                    } else {
                        "{ name = \"unknown\", version = \"unknown\" }".to_string()
                    }
                })
                .collect();
        }
    } else if cfg!(target_os = "macos") {
        if let Ok(output) = Command::new("brew").arg("list").arg("--versions").output() {
            let packages = String::from_utf8_lossy(&output.stdout);
            info.packages = packages
                .lines()
                .take(50) // 限制包数量
                .map(|line| {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        format!("{{ name = \"{}\", version = \"{}\" }}", parts[0], parts[1])
                    } else if !parts.is_empty() {
                        format!("{{ name = \"{}\", version = \"latest\" }}", parts[0])
                    } else {
                        "{ name = \"unknown\", version = \"unknown\" }".to_string()
                    }
                })
                .collect();
        }
    }

    Ok(())
}

fn collect_software_versions(info: &mut SystemInfo) -> Result<()> {
    // Python
    if let Ok(output) = Command::new("python3").arg("--version").output() {
        let version = String::from_utf8_lossy(&output.stdout).to_string();
        if version.is_empty() {
            // 有时候 Python 版本会输出到 stderr
            let version = String::from_utf8_lossy(&output.stderr).to_string();
            if let Some(ver) = version.split_whitespace().nth(1) {
                info.software.insert("python".to_string(), ver.to_string());
            }
        } else if let Some(ver) = version.split_whitespace().nth(1) {
            info.software.insert("python".to_string(), ver.to_string());
        }
    }

    // R
    if let Ok(output) = Command::new("R").arg("--version").output() {
        let version = String::from_utf8_lossy(&output.stdout);
        if let Some(line) = version.lines().next() {
            if let Some(ver) = line.split_whitespace().nth(2) {
                info.software.insert("r".to_string(), ver.to_string());
            }
        }
    }

    // MATLAB
    if let Ok(output) = Command::new("matlab").arg("-batch").arg("version").output() {
        let version = String::from_utf8_lossy(&output.stdout);
        if let Some(ver) = version.split_whitespace().last() {
            info.software.insert("matlab".to_string(), ver.to_string());
        }
    }

    // Conda
    if let Ok(output) = Command::new("conda").arg("--version").output() {
        let version = String::from_utf8_lossy(&output.stdout);
        if let Some(ver) = version.split_whitespace().last() {
            info.software.insert("conda".to_string(), ver.to_string());
        }
    }

    // CUDA
    if let Ok(output) = Command::new("nvcc").arg("--version").output() {
        let version = String::from_utf8_lossy(&output.stdout);
        for line in version.lines() {
            if line.contains("release") {
                let words: Vec<&str> = line.split_whitespace().collect();
                if words.len() > 5 {
                    info.software
                        .insert("cuda".to_string(), words[4].to_string());
                    break;
                }
            }
        }
    }

    Ok(())
}

fn collect_cuda_paths(info: &mut SystemInfo) -> Result<()> {
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        let path = String::from_utf8_lossy(&output.stdout);
        let nvcc_path = path.trim();

        if !nvcc_path.is_empty() {
            if let Some(cuda_home) = Path::new(nvcc_path).parent().and_then(|p| p.parent()) {
                let cuda_home_str = cuda_home.to_string_lossy().to_string();
                info.cuda.cuda_home = cuda_home_str.clone();
                info.cuda
                    .ld_library_path
                    .push(format!("{}/lib64", cuda_home_str));
                info.cuda.cupti_path = format!("{}/extras/CUPTI/lib64", cuda_home_str);
            }
        }
    }

    Ok(())
}
