from pathlib import Path

def clean_file(filename="temp_output.log", start_line=306, base_dir=None):
    """
    Trim the first `start_line-1` lines from a file.

    - filename: absolute path or relative to `base_dir`
    - base_dir: directory to resolve relative filenames against
                (defaults to the folder containing this file)
    """
    # Resolve base directory
    base = Path(base_dir) if base_dir else Path(__file__).resolve().parent

    # Resolve the target path (absolute stays as-is; relative joins to base)
    p = Path(filename)
    if not p.is_absolute():
        p = base / p

    # Read, slice, write back
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    start = max(1, int(start_line))
    p.write_text("".join(lines[start-1:]), encoding="utf-8", errors="ignore")

def trim_mcmc_log(input_file, output_file):
    # Keywords to keep (all must be stripped properly)
    keep_prefixes = [
        "ITERATION",
        "X_NODE_I",
        "X_NODE_II",
        "Z_NODE_II",
        "X_NODE_III",
        "Z_NODE_III",
        "X_NODE_IV",
        "Z_NODE_IV",
        "X_NODE_V",
        "CURRENT MISFIT:",
        "NEW MISFIT:",
        "ACCEPTANCE RATIO:",
        "RANDOM NUMBER:",
        "TEMPERATURE:",
        "ACCEPTANCE RATE:",
        "ACCEPTED",
        "ADAPTATIVE M-H STARTED",
        "Skipping iteration ",
        "MAX ITERATION:",
        "LOOP ITERATION IS OVER!!",
        "LOOP ITERATION IS",
        "TOTAL SAMPLING TIME",
        "NUMBER OF ACCEPTED SAMPLES", 
        "=====",
        "Peak Total",
        "-- Peaks by Phase --",
        "[total]",
        "[setup_specfem]",
        "[mesh_and_PML]",
        "[forward_sims]",
        "[1d_data_add_noise_save]",
        "[finalize]"        "ITERATION",
        "X_NODE_I",
        "X_NODE_II",
        "Z_NODE_II",
        "X_NODE_III",
        "Z_NODE_III",
        "X_NODE_IV",
        "Z_NODE_IV",
        "X_NODE_V",
        "CURRENT MISFIT:",
        "NEW MISFIT:",
        "ACCEPTANCE RATIO:",
        "RANDOM NUMBER:",
        "TEMPERATURE:",
        "ACCEPTANCE RATE:",
        "ACCEPTED",
        "ADAPTATIVE M-H STARTED",
        "Skipping iteration ",
        "MAX ITERATION:",
        "LOOP ITERATION IS OVER!!",
        "LOOP ITERATION IS",
        "TOTAL SAMPLING TIME",
        "NUMBER OF ACCEPTED SAMPLES", 
        "=====",
        "Peak Total",
        "-- Peaks by Phase --",
        "[total]",
        "[setup_specfem]",
        "[mesh_and_PML]",
        "[forward_sims]",
        "[1d_data_add_noise_save]",
        "[finalize]"
    ]
    
    # Read and filter lines
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            stripped = line.lstrip()  # remove leading spaces/tabs
            if any(stripped.startswith(prefix) for prefix in keep_prefixes):
                outfile.write(line)  # keep original formatting

# Example usage:
# trim_simulation_file("simulation_log.txt", "trimmed_log.txt")