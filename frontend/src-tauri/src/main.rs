#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::process::Stdio;
use std::sync::Mutex;
use image as image_rs;
use sysinfo::System;
use tauri::image::Image;
use tauri::{Manager, RunEvent};

struct BackendProcessState(Mutex<Option<Child>>);

fn append_log(log_file: &Path, msg: &str) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log_file) {
        let _ = writeln!(f, "{msg}");
    }
}

fn set_main_window_icon(app: &tauri::App) {
    // Force a high-resolution icon for dev mode (and keep consistent with build mode).
    let icon_bytes = include_bytes!("../icons/icon.png");
    if let Ok(decoded) = image_rs::load_from_memory(icon_bytes) {
        let rgba = decoded.to_rgba8();
        let (width, height) = rgba.dimensions();
        let icon = Image::new_owned(rgba.into_raw(), width, height);
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.set_icon(icon);
        }
    }
}

#[cfg(target_os = "macos")]
fn configure_main_window_platform(app: &tauri::App) {
    // On macOS, use the native traffic-light buttons via titleBarStyle: Overlay,
    // so re-enable decorations (the JSON config sets decorations=false for Windows/Linux).
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.set_decorations(true);
    }
}

#[cfg(not(target_os = "macos"))]
fn configure_main_window_platform(_app: &tauri::App) {
    // Windows/Linux: keep frameless window with custom titlebar (decorations=false from config).
}

fn resolve_log_file(app_handle: &tauri::AppHandle) -> PathBuf {
    if let Ok(app_data_dir) = app_handle.path().app_data_dir() {
        let log_dir = app_data_dir.join("runtime").join("logs");
        let _ = fs::create_dir_all(&log_dir);
        return log_dir.join("backend-sidecar.log");
    }
    std::env::temp_dir().join("mcube-backend-sidecar.log")
}

fn kill_sidecar_descendants_and_leftovers(root_pid: u32, process_name_marker: &str, log_file: &Path) {
    let mut sys = System::new_all();
    sys.refresh_all();

    let self_pid = std::process::id().to_string();
    let root_pid_str = root_pid.to_string();
    let marker = process_name_marker.to_lowercase();

    // 1) Recursively locate descendants of the known sidecar root pid.
    let mut frontier: Vec<String> = vec![root_pid_str.clone()];
    let mut descendants: Vec<String> = Vec::new();
    loop {
        let mut discovered: Vec<String> = Vec::new();
        for (pid, process) in sys.processes() {
            let parent_str = process.parent().map(|p| p.to_string());
            if let Some(parent) = parent_str {
                let pid_str = pid.to_string();
                if frontier.iter().any(|f| f == &parent)
                    && pid_str != self_pid
                    && !descendants.iter().any(|d| d == &pid_str)
                {
                    discovered.push(pid_str);
                }
            }
        }
        if discovered.is_empty() {
            break;
        }
        frontier = discovered.clone();
        descendants.extend(discovered);
    }

    // Kill children first (reverse BFS order).
    for pid_str in descendants.iter().rev() {
        for (pid, process) in sys.processes() {
            if pid.to_string() == *pid_str {
                let name = process.name().to_string();
                let killed = process.kill();
                append_log(
                    log_file,
                    &format!("Kill descendant pid={pid_str} name={name} result={killed}"),
                );
            }
        }
    }

    // 2) Name-based fallback sweep for stubborn onefile leftovers.
    for (pid, process) in sys.processes() {
        let pid_str = pid.to_string();
        if pid_str == self_pid {
            continue;
        }
        let name = process.name().to_lowercase();
        if name.contains(&marker) {
            let killed = process.kill();
            append_log(
                log_file,
                &format!("Kill fallback pid={pid_str} name={name} result={killed}"),
            );
        }
    }
}

fn kill_stale_sidecars_by_name(process_name_marker: &str, log_file: &Path) {
    let mut sys = System::new_all();
    sys.refresh_all();

    let self_pid = std::process::id().to_string();
    let marker = process_name_marker.to_lowercase();

    for (pid, process) in sys.processes() {
        let pid_str = pid.to_string();
        if pid_str == self_pid {
            continue;
        }
        let name = process.name().to_lowercase();
        if name.contains(&marker) {
            let killed = process.kill();
            append_log(
                log_file,
                &format!("Startup sweep kill pid={pid_str} name={name} result={killed}"),
            );
        }
    }
}

fn main() {
    let app = tauri::Builder::default()
        .manage(BackendProcessState(Mutex::new(None)))
        .setup(|app| {
            set_main_window_icon(app);
            configure_main_window_platform(app);

            if cfg!(debug_assertions) {
                // In dev mode, backend is expected to run separately (uvicorn).
                return Ok(());
            }

            let app_handle = app.handle().clone();

            let app_data_dir = app_handle
                .path()
                .app_data_dir()
                .map_err(|e| format!("Failed to resolve app data directory: {e}"))?;
            std::fs::create_dir_all(&app_data_dir)
                .map_err(|e| format!("Failed to create app data directory: {e}"))?;
            let upload_root = app_data_dir.join("runtime").join("uploads");
            std::fs::create_dir_all(&upload_root)
                .map_err(|e| format!("Failed to create upload root directory: {e}"))?;
            let sidecar_tmp = std::env::temp_dir().join("mcube_sidecar_tmp");
            std::fs::create_dir_all(&sidecar_tmp)
                .map_err(|e| format!("Failed to create sidecar temp directory: {e}"))?;
            let sidecar_tmp_str = sidecar_tmp.to_string_lossy().to_string();
            let log_dir = app_data_dir.join("runtime").join("logs");
            fs::create_dir_all(&log_dir).map_err(|e| format!("Failed to create log directory: {e}"))?;
            let log_file = log_dir.join("backend-sidecar.log");
            append_log(
                &log_file,
                &format!("=== Launch M-Cube backend sidecar at {:?} ===", std::time::SystemTime::now()),
            );
            kill_stale_sidecars_by_name("mcube-backend", &log_file);

            let resource_dir = app_handle
                .path()
                .resource_dir()
                .map_err(|e| format!("Failed to resolve resource directory: {e}"))?;
            let binaries_dir = resource_dir.join("binaries");
            let sidecar_bin_name = if cfg!(target_os = "windows") {
                format!("mcube-backend-{}.exe", env!("TAURI_ENV_TARGET_TRIPLE"))
            } else {
                format!("mcube-backend-{}", env!("TAURI_ENV_TARGET_TRIPLE"))
            };
            let sidecar_path = binaries_dir.join(sidecar_bin_name);
            if !sidecar_path.exists() {
                append_log(
                    &log_file,
                    &format!("Backend binary not found: {}", sidecar_path.display()),
                );
                return Ok(());
            }

            let stdout_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_file)
                .map_err(|e| format!("Failed to open sidecar stdout log: {e}"))?;
            let stderr_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_file)
                .map_err(|e| format!("Failed to open sidecar stderr log: {e}"))?;

            let mut command = Command::new(&sidecar_path);
            command
                .current_dir(&binaries_dir)
                .env("UPLOAD_ROOT_DIR", upload_root.to_string_lossy().to_string())
                .env("MCUBE_BACKEND_HOST", "127.0.0.1")
                .env("MCUBE_BACKEND_PORT", "8000")
                .env("TMP", sidecar_tmp_str.clone())
                .env("TEMP", sidecar_tmp_str)
                .stdout(Stdio::from(stdout_file))
                .stderr(Stdio::from(stderr_file));

            #[cfg(target_os = "windows")]
            {
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x08000000;
                command.creation_flags(CREATE_NO_WINDOW);
            }

            let child = match command.spawn() {
                Ok(v) => v,
                Err(e) => {
                    append_log(
                        &log_file,
                        &format!(
                            "Failed to spawn backend sidecar: {e} ; bin={} ; cwd={}",
                            sidecar_path.display(),
                            binaries_dir.display()
                        ),
                    );
                    return Ok(());
                }
            };

            let state = app_handle.state::<BackendProcessState>();
            let mut guard = state
                .0
                .lock()
                .map_err(|_| String::from("Failed to lock backend process state"))?;
            *guard = Some(child);
            append_log(&log_file, "Backend sidecar spawned.");
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if matches!(event, RunEvent::Exit | RunEvent::ExitRequested { .. }) {
            let log_file = resolve_log_file(&app_handle);
            let state = app_handle.state::<BackendProcessState>();
            let child_to_kill = match state.0.lock() {
                Ok(mut guard) => guard.take(),
                Err(_) => None,
            };
            if let Some(mut child) = child_to_kill {
                let root_pid = child.id();
                let _ = child.kill();
                append_log(
                    &log_file,
                    &format!("Primary sidecar kill sent. root_pid={root_pid}"),
                );
                kill_sidecar_descendants_and_leftovers(root_pid, "mcube-backend", &log_file);
                append_log(&log_file, "Sidecar cleanup completed.");
            }
        }
    });
}
