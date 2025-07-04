import argparse
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from mostlyai import qa, engine


def main(data_path: str):

    engine.init_logging()
    df = pd.read_csv(data_path)
    is_sequential = "group_id" in df.columns
    synth, best_acc = train_and_resample(df, n=5 if is_sequential else 2, m=10)
    filename = "seq" if is_sequential else "flat"
    Path("output").mkdir(exist_ok=True)
    synth[list(df.columns)].to_csv(f"output/{filename}_{best_acc:.6f}.csv", index=False)


def train_and_resample(real_data, n=1, m=1, max_seconds=5*60*60):
    start_time = time.time()
    best_acc = 0
    best_dataset = None
    best_ws = None
    for _ in range(n):
        if time.time() - start_time > max_seconds*0.9:
            print("Timeout reached during training loop.")
            break
        y, acc, ws = train_model(real_data)
        if acc > best_acc:
            best_acc = acc
            best_dataset = y.copy(deep=True)
            best_ws = ws

    print(f"Best model acc: {best_acc:.6f}")

    for _ in range(m):
        if time.time() - start_time > max_seconds:
            print("Timeout reached during training loop.")
            break
        y, acc = sampler(real_data, best_ws)
        if acc > best_acc:
            best_acc = acc
            best_dataset = y.copy(deep=True)

    print(f"Best final acc: {best_acc:.6f}")
    print(f"Best final model path: {best_ws}\n")
    return best_dataset, best_acc


def sampler(real_data, model_dir):
    engine.generate(workspace_dir=model_dir, sample_size=len(real_data), batch_size=8192)
    y = pd.read_parquet(model_dir / "SyntheticData") #TBD

    return y, validate(y, real_data, model_dir)

def train_model(df):
    trn_to_reorder_tgt_df = df.copy(deep=True)
    trn_tgt_df = reorder_cols(trn_to_reorder_tgt_df)
    is_sequential = "group_id" in trn_tgt_df.columns
    if is_sequential:
        trn_ctx_df = trn_tgt_df[['group_id']].drop_duplicates()
        ws_path = "ws-tabular-sequential"
    else:
        trn_ctx_df = None
        ws_path = "ws-tabular-flat"
    ws = Path(ws_path) / datetime.now().strftime("%d-%H:%M")
    engine.split(
        workspace_dir=ws,
        tgt_data=trn_tgt_df,
        ctx_data=trn_ctx_df,
        tgt_context_key="group_id" if is_sequential else None,
        ctx_primary_key="group_id" if is_sequential else None,
        model_type="TABULAR",
        trn_val_split=0.85 if is_sequential else 0.85,
        tgt_encoding_types=None if is_sequential else None,
    )
    engine.analyze(workspace_dir=ws, value_protection=False)
    engine.encode(workspace_dir=ws)
    engine.train(
        model="MOSTLY_AI/Medium" if is_sequential else "MOSTLY_AI/Large",
        workspace_dir=ws,
        max_training_time=120,
        enable_flexible_generation=False,
        max_epochs=200,
        max_sequence_window=10,
        batch_size=8192 if is_sequential else 256,
    )
    engine.generate(workspace_dir=ws, sample_size=len(trn_tgt_df), batch_size=8192 if is_sequential else 256)
    synth = pd.read_parquet(ws / "SyntheticData")
    return synth, validate(synth[list(df.columns)], df, ws), ws


def reorder_cols(df):
    ordered_cols = list(sort_columns_by_mode_frequency(df))
    print(f"Reordering columns: {ordered_cols}")
    return df[ordered_cols]


def sort_columns_by_mode_frequency(df: pd.DataFrame) -> list[str]:
    unique_scores = {}
    frequencies = {}

    for col in df.columns:
        unique_scores[col] = -len(df[col].unique())
        value_counts = df[col].value_counts(normalize=True, dropna=False)
        mode_freq = value_counts.iloc[0]  # frequency of most common value
        frequencies[col] = mode_freq
    if "group_id" not in df.columns:
        sorted_cols = sorted(unique_scores, key=lambda x: (unique_scores[x], frequencies[x]), reverse=True)
    else:
        sorted_cols = sorted(unique_scores, key=lambda x: (frequencies[x]), reverse=False)
    return sorted_cols


def validate(final_data, real_data_original, ws):

    syn = final_data.copy()
    cols = real_data_original.select_dtypes(include='number').columns
    syn[cols] = syn[cols].apply(pd.to_numeric, errors='coerce')
    challenge_type = "sequential" if "group_id" in final_data.columns else "flat"
    _, metrics = qa.report(
        syn_tgt_data=syn,
        trn_tgt_data=real_data_original.copy(),
        tgt_context_key="group_id" if "group_id" in final_data.columns else None,
        report_path=ws / f"output/model-report-{challenge_type}.html",
    )

    print(f"Overall Accuracy of batch: {metrics.accuracy.overall:.6f}")
    return metrics.accuracy.overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sequential or flat models.")
    parser.add_argument("data_path", help="The path of the input csv file")

    args = parser.parse_args()
    main(args.data_path)
