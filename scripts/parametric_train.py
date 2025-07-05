#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple
import pandas as pd
from mostlyai import qa, engine


def main(data_path: str) -> None:
    """
    Main function that reads data from a CSV file, detects its type (sequential or flat),
    performs training and resampling to generate synthetic data, and saves the best result.
    """
    engine.init_logging()
    df = pd.read_csv(data_path)
    is_sequential = "group_id" in df.columns
    n = 5 if is_sequential else 2
    synth, best_acc = train_and_resample(df, n=n, m=10)
    filename = "seq" if is_sequential else "flat"
    Path("output").mkdir(exist_ok=True)
    synth[df.columns.tolist()].to_csv(f"output/{filename}_{best_acc:.6f}.csv", index=False)


def train_and_resample(
    real_data: pd.DataFrame,
    n: int = 1,
    m: int = 1,
    max_seconds: int = 5 * 60 * 60,
) -> Tuple[pd.DataFrame, float]:
    """
    Trains a model `n` times and resamples `m` times to find the best synthetic dataset based on accuracy.

    Args:
        real_data: Original real dataset.
        n: Number of initial training attempts.
        m: Number of resampling attempts with the best model.
        max_seconds: Maximum allowed runtime in seconds.

    Returns:
        Tuple of the best synthetic dataset and its validation accuracy.
    """
    start_time = time.time()
    best_acc = 0.0
    best_dataset = None
    best_ws = None

    for _ in range(n):
        if time.time() - start_time > max_seconds * 0.9:
            print("Timeout reached during training loop.")
            break
        synth, acc, ws = train_model(real_data)
        if acc > best_acc:
            best_acc = acc
            best_dataset = synth.copy(deep=True)
            best_ws = ws

    print(f"Best model acc: {best_acc:.6f}")

    for _ in range(m):
        if time.time() - start_time > max_seconds:
            print("Timeout reached during training loop.")
            break
        synth, acc = sampler(real_data, best_ws)
        if acc > best_acc:
            best_acc = acc
            best_dataset = synth.copy(deep=True)

    print(f"Best final acc: {best_acc:.6f}")
    print(f"Best final model path: {best_ws}\n")
    return best_dataset, best_acc


def sampler(real_data: pd.DataFrame, model_dir: Path) -> Tuple[pd.DataFrame, float]:
    """
    Generates synthetic data using a pre-trained model and validates it.

    Args:
        real_data: The original dataset.
        model_dir: Path to the pre-trained model workspace.

    Returns:
        Tuple of synthetic dataset and its validation accuracy.
    """
    engine.generate(workspace_dir=model_dir, sample_size=len(real_data), batch_size=8192)
    synth = pd.read_parquet(model_dir / "SyntheticData")
    return synth, validate(synth, real_data, model_dir)


def train_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, Path]:
    """
    Trains a MOSTLY AI model on the provided dataset and returns synthetic output and accuracy.

    Args:
        df: DataFrame to train on.

    Returns:
        Tuple of synthetic dataset, validation accuracy, and workspace path.
    """
    df_copy = df.copy(deep=True)
    df_ordered = reorder_cols(df_copy)
    is_seq = "group_id" in df_ordered.columns
    ws_name = "ws-tabular-sequential" if is_seq else "ws-tabular-flat"
    ws = Path(ws_name) / datetime.now().strftime("%d-%H:%M")

    ctx_df = df_ordered[['group_id']].drop_duplicates() if is_seq else None

    engine.split(
        workspace_dir=ws,
        tgt_data=df_ordered,
        ctx_data=ctx_df,
        tgt_context_key="group_id" if is_seq else None,
        ctx_primary_key="group_id" if is_seq else None,
        model_type="TABULAR",
        trn_val_split=0.85,
        tgt_encoding_types=None,
    )

    engine.analyze(workspace_dir=ws, value_protection=False)
    engine.encode(workspace_dir=ws)

    engine.train(
        model="MOSTLY_AI/Medium" if is_seq else "MOSTLY_AI/Large",
        workspace_dir=ws,
        max_training_time=120,
        enable_flexible_generation=False,
        max_epochs=200,
        max_sequence_window=10,
        batch_size=8192 if is_seq else 256,
    )

    engine.generate(workspace_dir=ws, sample_size=len(df_ordered), batch_size=8192 if is_seq else 256)
    synth = pd.read_parquet(ws / "SyntheticData")
    acc = validate(synth[df.columns], df, ws)
    return synth, acc, ws


def reorder_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorders columns in the DataFrame based on uniqueness and frequency.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns reordered.
    """
    ordered_cols = sort_columns_by_mode_frequency(df)
    print(f"Reordering columns: {ordered_cols}")
    return df[ordered_cols]


def sort_columns_by_mode_frequency(df: pd.DataFrame) -> list[str]:
    """
    Sorts columns by number of unique values and mode frequency.

    Args:
        df: Input DataFrame.

    Returns:
        List of column names sorted based on custom logic.
    """
    unique_scores = {col: -len(df[col].unique()) for col in df.columns}
    frequencies = {col: df[col].value_counts(normalize=True, dropna=False).iloc[0] for col in df.columns}

    if "group_id" not in df.columns:
        return sorted(df.columns, key=lambda x: (unique_scores[x], frequencies[x]), reverse=True)
    else:
        return sorted(df.columns, key=lambda x: frequencies[x])


def validate(final_data: pd.DataFrame, real_data_original: pd.DataFrame, ws: Path) -> float:
    """
    Validates synthetic data against the real data using MOSTLY AI's QA report.

    Args:
        final_data: Synthetic dataset.
        real_data_original: Original real dataset.
        ws: Workspace path to store validation report.

    Returns:
        Accuracy score of the synthetic data.
    """
    syn = final_data.copy()
    numeric_cols = real_data_original.select_dtypes(include='number').columns
    syn[numeric_cols] = syn[numeric_cols].apply(pd.to_numeric, errors='coerce')

    challenge_type = "sequential" if "group_id" in final_data.columns else "flat"
    _, metrics = qa.report(
        syn_tgt_data=syn,
        trn_tgt_data=real_data_original.copy(),
        tgt_context_key="group_id" if "group_id" in final_data.columns else None,
        report_path=ws / f"output/model-report-{challenge_type}.html",
    )

    acc = metrics.accuracy.overall
    print(f"Overall Accuracy of batch: {acc:.6f}")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sequential or flat models.")
    parser.add_argument("data_path", help="Path to the input CSV file.")
    args = parser.parse_args()
    main(args.data_path)
