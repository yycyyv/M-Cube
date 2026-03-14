#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::process::{Child, Command};
use std::process::Stdio;
use std::sync::Mutex;
use tauri::{Manager, RunEvent};

struct BackendProcessState(Mutex<Option<Child>>);

fn append_log(log_file: &std::path::Path, msg: &str) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log_file) {
        let _ = writeln!(f, "{msg}");
    }
}

fn main() {
    let app = tauri::Builder::default()
        .manage(BackendProcessState(Mutex::new(None)))
        .setup(|app| {
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
            let state = app_handle.state::<BackendProcessState>();
            let child_to_kill = match state.0.lock() {
                Ok(mut guard) => guard.take(),
                Err(_) => None,
            };
            if let Some(mut child) = child_to_kill {
                let _ = child.kill();
            }
        }
    });
}
