"""Validation utilities for workflow outputs."""

import os
from pathlib import Path
from typing import Any, Tuple, Dict, List, Optional
from rich.console import Console

# Import from other CRESP modules
from ..validation import validate_artifact
from ..utils import calculate_artifact_hash
from .visualization import RICH_AVAILABLE, DummyConsole

# Import visualization utilities
try:
    if RICH_AVAILABLE:
        console: Console = Console()
    else:
        console: Console = DummyConsole()  # type: ignore
except ImportError:
    RICH_AVAILABLE = False
    console: Console = DummyConsole()  # type: ignore


def check_outputs_unchanged(
    stage_id: str,
    registered_stage,
    config,
    active_output_dir: Path,
    shared_data_dir: Path,
) -> bool:
    """Check if stage outputs exist and match stored hashes.

    Args:
        stage_id: The ID of the stage
        registered_stage: The StageFunction object for the stage
        config: CrespConfig object
        active_output_dir: Path to the active output directory
        shared_data_dir: Path to the shared data directory

    Returns:
        Boolean indicating if all outputs are unchanged
    """
    stage_config = config.get_stage(stage_id)

    if not registered_stage or not stage_config or "outputs" not in stage_config:
        # Cannot determine if unchanged if config or stage definition is missing
        return False

    declared_outputs = registered_stage.outputs  # From @workflow.stage(...)
    if not declared_outputs:
        # If no outputs are declared in the stage definition,
        # we cannot skip based on "unchanged outputs". Always run.
        return False

    config_outputs = stage_config.get("outputs", [])
    if not config_outputs:
        return False  # Outputs declared in code, but none found in config (e.g., first run)

    # Create a lookup for config outputs by path
    expected_files_config: dict[str, dict[str, Any]] = {
        cfg_out["path"]: cfg_out
        for cfg_out in config_outputs
        if "path" in cfg_out and "hash" in cfg_out  # Only consider items with path and hash
    }

    if not expected_files_config:
        # We have outputs in config, but none have hashes yet
        return False

    all_match = True
    files_checked_count = 0

    # Iterate through outputs declared in the @workflow.stage decorator
    for output_decl in declared_outputs:
        # --- Determine reproduction settings from declaration and stage defaults ---
        default_repro_mode = registered_stage.reproduction_mode
        default_tol_abs = registered_stage.tolerance_absolute
        default_tol_rel = registered_stage.tolerance_relative
        default_sim_thresh = registered_stage.similarity_threshold

        output_specific_repro_config = {}
        if isinstance(output_decl, str):
            path_str = output_decl
            is_shared: bool = False  # Default to non-shared for strings
        elif isinstance(output_decl, dict) and "path" in output_decl:
            path_str = output_decl["path"]
            output_specific_repro_config = output_decl.get("reproduction", {})
            # Check shared flag EXPLICITLY
            is_shared = bool(output_decl.get("shared", False))
        else:
            continue  # Skip invalid declaration

        # --- Resolve path for validation ---
        try:
            if is_shared:
                resolved_path = shared_data_dir / path_str
            else:
                resolved_path = active_output_dir / path_str

            # Ensure parent directory exists
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If we can't resolve the path, consider it changed
            return False

        # --- Find corresponding files in config based on path_str ---
        files_to_validate_for_this_decl: dict[str, dict[str, Any]] = {}

        # --- Determine the base directory for comparison ---
        base_dir = shared_data_dir if is_shared else active_output_dir

        # Calculate the relative path key this declaration would correspond to
        try:
            relative_key = resolved_path.relative_to(base_dir)
            relative_key_str = str(relative_key)
        except ValueError:
            # Path is not relative to the expected base, something is wrong
            return False  # Cannot determine, assume changed

        # Case 1: Declared path is a file
        if resolved_path.exists() and resolved_path.is_file():
            if relative_key_str in expected_files_config:
                config_entry = expected_files_config[relative_key_str]
                # Check if scope matches (config should store 'shared' flag)
                config_is_shared = bool(config_entry.get("shared", False))
                if config_is_shared == is_shared:
                    files_to_validate_for_this_decl[relative_key_str] = config_entry
                else:
                    # Mismatch in scope between declaration and config, assume changed
                    return False
            else:
                # Declared file exists but not found in config (with hash)
                return False

        # Case 2: Declared path is a directory
        elif resolved_path.exists() and resolved_path.is_dir():
            found_match_in_dir = False
            # Iterate through config entries and see if they fall under this directory
            for config_key, config_entry in expected_files_config.items():
                config_is_shared = bool(config_entry.get("shared", False))
                config_base_dir = shared_data_dir if config_is_shared else active_output_dir
                try:
                    # Construct full path from config key
                    config_full_path = (config_base_dir / config_key).resolve()

                    # Check if config file path is within the declared directory path
                    if config_full_path.is_relative_to(resolved_path.resolve()):
                        # Check if file exists on disk
                        if not config_full_path.exists():
                            return False  # File listed in config (under dir) is missing

                        # Check scope consistency (config file must have same scope as declared dir)
                        if config_is_shared != is_shared:
                            return False  # Scope mismatch

                        files_to_validate_for_this_decl[config_key] = config_entry
                        found_match_in_dir = True
                except ValueError:
                    # Config path not relative to declared directory path
                    continue
                except Exception:
                    # Error resolving or checking path
                    return False  # Error, assume changed

            if not found_match_in_dir:
                # Directory declared, but no known/hashed files within found in config
                return False
        else:
            # Declared path does not exist (as file or dir)
            return False

        # --- Validate the collected files for this declaration ---
        if not files_to_validate_for_this_decl:
            # This case might mean the declaration exists but has no hashed files associated yet.
            return False

        for file_path_str, file_cfg in files_to_validate_for_this_decl.items():
            files_checked_count += 1

            # Determine final validation parameters (File > Declaration > Stage)
            file_specific_repro_config = file_cfg.get("reproduction", {})

            final_repro_mode = file_specific_repro_config.get("mode", output_specific_repro_config.get("mode", default_repro_mode))
            final_tol_abs = file_specific_repro_config.get(
                "tolerance_absolute",
                output_specific_repro_config.get("tolerance_absolute", default_tol_abs),
            )
            final_tol_rel = file_specific_repro_config.get(
                "tolerance_relative",
                output_specific_repro_config.get("tolerance_relative", default_tol_rel),
            )
            final_sim_thresh = file_specific_repro_config.get(
                "similarity_threshold",
                output_specific_repro_config.get("similarity_threshold", default_sim_thresh),
            )

            # Resolve file path based on shared flag
            try:
                file_is_shared = bool(file_cfg.get("shared", is_shared))
                if file_is_shared:
                    actual_file_path = shared_data_dir / file_path_str
                else:
                    actual_file_path = active_output_dir / file_path_str
            except Exception:
                all_match = False
                break

            # Perform validation using the hash from config
            success, _ = validate_artifact(
                actual_file_path,
                file_cfg["hash"],  # Hash must exist from earlier check
                validation_type=final_repro_mode,
                tolerance_absolute=final_tol_abs,
                tolerance_relative=final_tol_rel,
                similarity_threshold=final_sim_thresh,
            )

            if not success:
                all_match = False
                break  # Stop checking this declaration if one file fails

        if not all_match:
            break  # Stop checking other declarations if one failed

    # Return True only if all declarations were processed,
    # at least one file was checked, and all matched.
    return all_match and files_checked_count > 0


def update_output_hashes(
    stage_id: str,
    declared_outputs: list[str | dict[str, Any]],
    config,
    active_output_dir: Path,
    shared_data_dir: Path,
    use_rich: bool = True,
) -> list[tuple[str, str]]:
    """Calculate and store hashes for stage outputs in experiment mode.

    Args:
        stage_id: The ID of the stage
        declared_outputs: List of output declarations
        config: CrespConfig object
        active_output_dir: Path to the active output directory
        shared_data_dir: Path to the shared data directory
        use_rich: Whether to use rich output

    Returns:
        List of (path, hash) tuples
    """
    hashes_calculated = []
    # Use batch update for efficiency when hashing multiple files/directories
    with config.batch_update():
        for output_decl in declared_outputs:
            # --- Determine path and hash method from declaration ---
            path_str: str | None = None
            hash_method = "sha256"  # Default hash method
            is_shared = False  # Default scope is mode-specific

            if isinstance(output_decl, str):
                path_str = output_decl
                # Apply heuristic: path without separator is likely shared
                if "/" not in path_str and "\\" not in path_str:
                    is_shared = True
            elif isinstance(output_decl, dict):
                path_str = output_decl.get("path")
                # Allow overriding hash method per output declaration
                hash_method = output_decl.get("hash_method", hash_method)
                # Check for shared flag with fallback to heuristic
                if "shared" in output_decl:
                    is_shared = bool(output_decl.get("shared", False))
                elif path_str and "/" not in path_str and "\\" not in path_str:
                    is_shared = True

            if not path_str:
                if use_rich and RICH_AVAILABLE:
                    console.print(f"[yellow]Warning: Invalid output declaration in stage '{stage_id}': {output_decl}[/yellow]")
                continue

            # --- Resolve path based on scope ---
            try:
                if is_shared:
                    resolved_path = shared_data_dir / path_str
                else:
                    resolved_path = active_output_dir / path_str

                # Ensure parent directory exists
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as path_resolve_e:
                if use_rich and RICH_AVAILABLE:
                    console.print(f"[yellow]Warning: Could not resolve path '{path_str}' for stage '{stage_id}': {path_resolve_e}[/yellow]")
                continue

            # --- Check existence using the RESOLVED path ---
            try:
                if not resolved_path.exists():
                    if use_rich and RICH_AVAILABLE:
                        msg = f"Warning: Output path does not exist after running stage '{stage_id}': {resolved_path} (declared as '{path_str}')"
                        console.print(f"[yellow]{msg}[/yellow]")
                    continue

                # --- Hash File or Directory Contents using RESOLVED path ---
                if resolved_path.is_file():
                    hash_value = calculate_artifact_hash(resolved_path, method=hash_method)
                    # Update config using the RELATIVE path as the key
                    base_dir = shared_data_dir if is_shared else active_output_dir
                    try:
                        relative_key = str(resolved_path.relative_to(base_dir))
                    except ValueError:
                        relative_key = str(resolved_path)
                    config_entry = {
                        "path": relative_key,
                        "hash": hash_value,
                        "hash_method": hash_method,
                        "shared": is_shared,
                    }
                    if hasattr(config, "update_artifact"):
                        config.update_artifact(stage_id, relative_key, config_entry)
                    else:
                        config.update_hash(stage_id, relative_key, hash_value, hash_method)
                    hashes_calculated.append((relative_key, hash_value))
                elif resolved_path.is_dir():
                    if use_rich and RICH_AVAILABLE:
                        console.print(f"[dim]Hashing contents of directory [bold]{resolved_path}[/bold] (declared as '{path_str}')...[/dim]")
                    files_hashed_count = 0
                    base_dir = shared_data_dir if is_shared else active_output_dir
                    for file_path in resolved_path.rglob("*"):
                        if file_path.is_file():
                            try:
                                file_hash_value = calculate_artifact_hash(file_path, method=hash_method)
                                try:
                                    relative_key_path_str = str(file_path.relative_to(base_dir))
                                except ValueError:
                                    relative_key_path_str = str(file_path.resolve())
                                config_entry = {
                                    "path": relative_key_path_str,
                                    "hash": file_hash_value,
                                    "hash_method": hash_method,
                                    "shared": is_shared,
                                }
                                if hasattr(config, "update_artifact"):
                                    config.update_artifact(stage_id, relative_key_path_str, config_entry)
                                else:
                                    config.update_hash(
                                        stage_id,
                                        relative_key_path_str,
                                        file_hash_value,
                                        hash_method,
                                    )
                                hashes_calculated.append((relative_key_path_str, file_hash_value))
                                files_hashed_count += 1
                            except Exception as file_e:
                                if use_rich and RICH_AVAILABLE:
                                    console.print(f"[yellow]  Warning: Failed to hash file {file_path}: {str(file_e)}[/yellow]")
                    if use_rich and RICH_AVAILABLE and files_hashed_count > 0:
                        console.print(f"[dim]Hashed {files_hashed_count} files in [bold]{resolved_path}[/bold].[/dim]")
                else:
                    if use_rich and RICH_AVAILABLE:
                        console.print(f"[yellow]Warning: Resolved output path is not a file or directory: {resolved_path}[/yellow]")

            except Exception as e:
                if use_rich and RICH_AVAILABLE:
                    msg = f"Warning: Failed to process output {resolved_path} (declared as '{path_str}') for stage '{stage_id}': {str(e)}"
                    console.print(f"[yellow]{msg}[/yellow]")

    return hashes_calculated


def validate_outputs(
    stage_id: str,
    registered_stage,
    config,
    active_output_dir: Path,
    shared_data_dir: Path,
    validation_results: list,
    use_rich: bool = True,
) -> bool:
    """Validate stage outputs against stored hashes in reproduction mode.

    Args:
        stage_id: The ID of the stage
        registered_stage: The StageFunction object for the stage
        config: CrespConfig object
        active_output_dir: Path to the active output directory
        shared_data_dir: Path to the shared data directory
        validation_results: List to append validation results to
        use_rich: Whether to use rich output

    Returns:
        Boolean indicating if all non-ignored validations passed
    """
    stage_config = config.get_stage(stage_id)

    if not registered_stage:
        return True  # Should not happen, but safer

    declared_outputs = registered_stage.outputs

    if not stage_config or not stage_config.get("outputs"):
        # If outputs declared in code but none in config (maybe first run?), treat as warning/pass?
        # If no outputs declared in code, it's fine.
        if declared_outputs and use_rich and RICH_AVAILABLE:
            console.print(f"[yellow]Warning: No reference outputs/hashes found in config for stage '{stage_id}' to validate against.[/yellow]")
        return True  # Consider valid if nothing to compare against

    if not declared_outputs:
        return True  # Nothing declared in code, nothing to validate here.

    # Create a lookup for config outputs by path (must have hash)
    expected_files_config: dict[str, dict[str, Any]] = {
        cfg_out["path"]: cfg_out for cfg_out in stage_config.get("outputs", []) if cfg_out.get("path") and cfg_out.get("hash")
    }

    if not expected_files_config:
        if use_rich and RICH_AVAILABLE:
            console.print(f"[yellow]Warning: Outputs found in config for stage '{stage_id}', but none have recorded hashes.[/yellow]")
        # Let's treat it as pass for now, as we can't validate.
        return True

    stage_validation_passed = True  # Assume success until proven otherwise
    validation_performed = False  # Track if any actual validation occurred

    # Iterate through outputs declared in the @workflow.stage decorator
    for output_decl in declared_outputs:
        # --- Determine reproduction settings from declaration and stage defaults ---
        default_repro_mode = registered_stage.reproduction_mode
        default_tol_abs = registered_stage.tolerance_absolute
        default_tol_rel = registered_stage.tolerance_relative
        default_sim_thresh = registered_stage.similarity_threshold

        output_specific_repro_config = {}
        path_str: str | None = None
        is_shared = False  # Default to mode-specific scope

        if isinstance(output_decl, str):
            path_str = output_decl
            # Apply heuristic for backward compatibility
            if "/" not in path_str and "\\" not in path_str:
                is_shared = True
        elif isinstance(output_decl, dict) and "path" in output_decl:
            path_str = output_decl["path"]
            output_specific_repro_config = output_decl.get("reproduction", {})
            # Check shared flag with fallback to heuristic
            if "shared" in output_decl:
                is_shared = bool(output_decl.get("shared", False))
            # Apply heuristic as fallback
            elif path_str and "/" not in path_str and "\\" not in path_str:
                is_shared = True
        else:
            continue  # Skip invalid declaration

        # --- Resolve path for validation ---
        try:
            if path_str is None:
                # Skip if path is None
                continue

            if is_shared:
                resolved_path = shared_data_dir / path_str
            else:
                resolved_path = active_output_dir / path_str

            # Ensure parent directory exists
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as path_resolve_e:
            if use_rich and RICH_AVAILABLE:
                console.print(f"[yellow]Warning: Could not resolve path '{path_str}' for validation in stage '{stage_id}': {path_resolve_e}[/yellow]")
            continue

        # --- Find corresponding files in config based on ORIGINAL path_str ---
        # We still use path_str to lookup in expected_files_config which uses declared paths as keys
        files_to_validate_for_this_decl: dict[str, dict[str, Any]] = {}
        found_config_match = False

        # Case 1: Exact file path match in config
        if path_str in expected_files_config:
            # Override is_shared based on what's stored in config if available
            if "shared" in expected_files_config[path_str]:
                is_shared = bool(expected_files_config[path_str].get("shared", False))
            files_to_validate_for_this_decl[path_str] = expected_files_config[path_str]
            found_config_match = True
        # Case 2: Directory path - find config entries within
        elif resolved_path.exists() and resolved_path.is_dir():
            # Check config paths starting with the declared path_str or within the resolved path
            config_paths_in_dir = []
            for expected_path, expected_cfg in expected_files_config.items():
                # Normalize paths for comparison
                # Skip None values to avoid errors in normpath
                if expected_path is None or path_str is None:
                    continue

                norm_expected = os.path.normpath(expected_path)
                norm_declared = os.path.normpath(path_str)

                # Is the config file directly within this directory?
                is_in_dir = False

                # Method 1: Check if expected path starts with declared path + separator
                if norm_expected.startswith(norm_declared + os.sep):
                    is_in_dir = True
                # Method 2: Physically check if the file is in the resolved directory
                else:
                    # Try to resolve as shared or output path
                    try:
                        expected_is_shared = bool(expected_cfg.get("shared", False))
                        if expected_is_shared:
                            expected_resolved = shared_data_dir / expected_path
                        else:
                            expected_resolved = active_output_dir / expected_path
                        # Is this file physically within the declared directory?
                        try:
                            is_in_dir = expected_resolved.is_relative_to(resolved_path)
                        except ValueError:
                            # Not within the directory
                            pass
                    except Exception:
                        # Couldn't resolve path
                        pass

                if is_in_dir:
                    config_paths_in_dir.append(expected_path)
                    # Override is_shared based on what's stored in config if available
                    file_is_shared = is_shared
                    if "shared" in expected_cfg:
                        file_is_shared = bool(expected_cfg.get("shared", False))
                    # Store with is_shared flag explicitly known from config
                    expected_cfg_with_scope = expected_cfg.copy()
                    expected_cfg_with_scope["_is_shared"] = file_is_shared
                    # The key remains the declared path from config
                    files_to_validate_for_this_decl[expected_path] = expected_cfg_with_scope
                    found_config_match = True

        if not found_config_match:
            # Declared output has no corresponding hashed entry in config
            if use_rich and RICH_AVAILABLE:
                console.print(
                    f"[yellow]Warning: No reference hash found in config matching output declaration '{path_str}' for stage '{stage_id}'. Cannot validate.[/yellow]"
                )

                # Debug - For directories, list all config paths to help diagnose matching issues
                if resolved_path.exists() and resolved_path.is_dir():
                    console.print(f"[dim]Directory exists at [bold]{resolved_path}[/bold]. Config has the following entries:[/dim]")
                    for i, (config_path, _) in enumerate(expected_files_config.items()):
                        if i < 10:  # Limit to first 10 for readability
                            console.print(f"[dim]  - {config_path}[/dim]")
                        elif i == 10:
                            console.print(f"[dim]  - ... and {len(expected_files_config) - 10} more[/dim]")
            continue

        # --- Validate the collected files for this declaration ---

        # Note: The loop below iterates through paths found in the config (expected_path)
        # We need to map these back to resolved paths on the filesystem for validation.
        for expected_path_key, file_cfg in files_to_validate_for_this_decl.items():
            validation_performed = True  # Mark that we attempted validation/ignore
            # Double check hash exists (should from earlier filter)
            if "hash" not in file_cfg:
                continue

            # --- Resolve the *expected* path from config to its actual filesystem location ---
            # Use _is_shared flag set above if available, otherwise fall back to global is_shared
            expected_is_shared = file_cfg.pop("_is_shared", is_shared)
            # Alternatively, check if shared is explicitly set in config
            if "shared" in file_cfg:
                expected_is_shared = bool(file_cfg.get("shared", False))

            # --- Determine final validation parameters (File > Declaration > Stage) ---
            file_specific_repro_config = file_cfg.get("reproduction", {})
            final_repro_mode = file_specific_repro_config.get("mode", output_specific_repro_config.get("mode", default_repro_mode))
            final_tol_abs = file_specific_repro_config.get(
                "tolerance_absolute",
                output_specific_repro_config.get("tolerance_absolute", default_tol_abs),
            )
            final_tol_rel = file_specific_repro_config.get(
                "tolerance_relative",
                output_specific_repro_config.get("tolerance_relative", default_tol_rel),
            )
            final_sim_thresh = file_specific_repro_config.get(
                "similarity_threshold",
                output_specific_repro_config.get("similarity_threshold", default_sim_thresh),
            )

            # --- Get actual file path for display/validation ---
            actual_file_path_display: str
            try:
                if expected_is_shared:
                    actual_file_path_to_check = shared_data_dir / expected_path_key
                else:
                    actual_file_path_to_check = active_output_dir / expected_path_key
                actual_file_path_display = str(actual_file_path_to_check)  # For logging
            except Exception:
                # If we can't even resolve the path from config, treat as validation failure
                message = f"Could not resolve expected path from config: {expected_path_key}"
                success = False
                final_repro_mode = "N/A"  # Mode doesn't apply if path resolution failed
                actual_file_path_display = expected_path_key  # Display the problematic key
                status_str = "Failed"
                status_symbol = "❌"
                log_style = "red"

                if use_rich and RICH_AVAILABLE:
                    console.print(f"[{log_style}]  {status_symbol} {actual_file_path_display}: {message}[/{log_style}]")
                stage_validation_passed = False  # Mark stage as failed

                # Record result even on resolution failure
                validation_results.append(
                    {
                        "stage": stage_id,
                        "file": actual_file_path_display,
                        "status": status_str,
                        "mode": final_repro_mode,
                        "message": message,
                    }
                )
                continue  # Skip to next file

            # --- Handle 'ignore' mode ---
            if final_repro_mode == "ignore":
                success = True  # Treat ignore as success for stage status
                message = "Validation skipped (ignore mode)"
                status_str = "Ignored"
                status_symbol = "⚪"
                log_style = "yellow"
            else:
                # --- Perform actual validation ---
                # Check if the *actual* file exists on disk before attempting validation
                if not actual_file_path_to_check.exists():
                    message = f"Output file not found: {actual_file_path_to_check}"
                    success = False
                    # Mode doesn't apply if file is missing, keep final_repro_mode determined above
                    status_str = "Failed"
                    status_symbol = "❌"
                    log_style = "red"
                else:
                    # Perform validation using the ACTUAL file path
                    success, message = validate_artifact(
                        actual_file_path_to_check,
                        file_cfg["hash"],
                        validation_type=final_repro_mode,
                        tolerance_absolute=final_tol_abs,
                        tolerance_relative=final_tol_rel,
                        similarity_threshold=final_sim_thresh,
                    )
                    if success:
                        status_str = "Passed"
                        status_symbol = "✅"
                        log_style = "green"
                    else:
                        status_str = "Failed"
                        status_symbol = "❌"
                        log_style = "red"

            # --- Record validation result ---
            validation_results.append(
                {
                    "stage": stage_id,
                    "file": actual_file_path_display,  # Report the actual path checked
                    "status": status_str,
                    "mode": final_repro_mode,
                    "message": message,
                }
            )

            # --- Update overall stage status and print result ---
            if use_rich and RICH_AVAILABLE:
                console.print(f"[{log_style}]  {status_symbol} {actual_file_path_display}: {message}[/{log_style}]")

            if not success and final_repro_mode != "ignore":
                stage_validation_passed = False  # Mark stage as failed if any non-ignored file fails

    # Return True only if validation was performed AND no failures occurred.
    # If validation_performed is False, it means no outputs could be checked (e.g., missing config/hashes)
    # In that case, we return True based on the initial checks.
    if not validation_performed:
        return True  # Return based on initial checks if no specific file validation happened

    return stage_validation_passed  # Return overall status for the stage
