import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # Console handler (stderr so stdout stays clean for JSON)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict) -> None:
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def write_error(output_path: str, version: str, message: str) -> None:
    write_metrics(output_path, {
        "version": version,
        "status": "error",
        "error_message": message,
    })


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file is empty or not a valid YAML mapping.")
    required = {"seed", "window", "version"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"Config 'seed' must be an integer, got: {type(cfg['seed'])}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"Config 'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str):
        raise ValueError(f"Config 'version' must be a string, got: {type(cfg['version'])}")
    return cfg


def load_dataset(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if df.empty:
        raise ValueError("Input CSV is empty.")
    if "close" not in df.columns:
        raise ValueError(f"Required column 'close' not found. Columns present: {list(df.columns)}")
    if df["close"].isnull().all():
        raise ValueError("Column 'close' contains only null values.")
    return df


def compute_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling mean with min_periods=window so the first (window-1) rows are NaN.
    These rows are excluded from signal computation.
    """
    return series.rolling(window=window, min_periods=window).mean()


def compute_signal(close: pd.Series, rolling_mean: pd.Series) -> pd.Series:
    """
    signal = 1 if close > rolling_mean, else 0.
    Rows where rolling_mean is NaN are excluded (set to NaN, then cast to int after dropping).
    """
    valid_mask = rolling_mean.notna()
    signal = pd.Series(np.nan, index=close.index)
    signal[valid_mask] = (close[valid_mask] > rolling_mean[valid_mask]).astype(int)
    return signal


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str, config_path: str, output_path: str, log_file: str) -> int:
    start_time = time.time()
    logger = setup_logging(log_file)

    logger.info("=== Job started ===")
    logger.info(f"Input: {input_path} | Config: {config_path} | Output: {output_path}")

    version = "unknown"

    try:
        # --- Config ---
        logger.info("Loading and validating config...")
        cfg = load_config(config_path)
        version = cfg["version"]
        seed = cfg["seed"]
        window = cfg["window"]
        logger.info(f"Config loaded — seed={seed}, window={window}, version={version}")

        # --- Seed ---
        np.random.seed(seed)
        logger.info(f"Random seed set: {seed}")

        # --- Dataset ---
        logger.info(f"Loading dataset from {input_path}...")
        df = load_dataset(input_path)
        total_rows = len(df)
        logger.info(f"Dataset loaded: {total_rows} rows, columns: {list(df.columns)}")

        # --- Rolling mean ---
        logger.info(f"Computing rolling mean on 'close' with window={window}...")
        df["rolling_mean"] = compute_rolling_mean(df["close"], window)
        nan_rows = df["rolling_mean"].isna().sum()
        logger.info(f"Rolling mean computed. First {nan_rows} rows excluded (NaN) due to window warm-up.")

        # --- Signal ---
        logger.info("Generating binary signal (close > rolling_mean)...")
        df["signal"] = compute_signal(df["close"], df["rolling_mean"])
        valid_signals = df["signal"].dropna()
        rows_processed = len(valid_signals)
        signal_rate = float(valid_signals.mean())
        logger.info(f"Signal generated. rows_processed={rows_processed}, signal_rate={signal_rate:.4f}")

        # --- Metrics ---
        latency_ms = int((time.time() - start_time) * 1000)
        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success",
        }
        write_metrics(output_path, metrics)
        logger.info(f"Metrics written to {output_path}: {metrics}")
        logger.info(f"=== Job completed successfully in {latency_ms}ms ===")

        # Print final metrics to stdout (required by Docker spec)
        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"Pipeline failed: {error_msg}", exc_info=True)
        write_error(output_path, version, error_msg)
        logger.info("Error metrics written. Exiting with code 1.")
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps OHLCV signal pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV")
    parser.add_argument("--config",   required=True, help="Path to YAML config")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file", help="Path for log file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = run_pipeline(
        input_path=args.input,
        config_path=args.config,
        output_path=args.output,
        log_file=args.log_file,
    )
    sys.exit(exit_code)
