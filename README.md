# 💎 Mostly AI Prize – Tecnarca’s Take

This repository is a customized fork of the [Mostly AI Engine](https://github.com/mostly-ai/mostlyai-engine), developed specifically for participation in the [2025 Mostly AI Prize](https://www.mostlyaiprize.com/).

It enables direct modification of the engine to tailor it for high-performance synthetic data generation in both the **flat** and **sequential** challenges.

---

## 🛠️ Setup

Follow these steps to get the environment ready:

### 1. Clone the Repository
```bash
git clone https://github.com/Tecnarca/mostlyai-engine-prize.git
cd mostlyai-engine-prize
```

### 2. Install `uv` (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
For alternative installation methods, refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### 3. Create Virtual Environment and Install Dependencies
If you're using a GPU:
```bash
uv sync --frozen --extra gpu --python=3.10
source .venv/bin/activate
```

---

## 🚀 Running a Training Job

Use the script at `scripts/parametric_train.py` to train a model for either challenge type.

### Script Usage
```bash
python scripts/parametric_train.py <path_to_training_dataset.csv>
```

- The script auto-detects the dataset type:
  - If the CSV has a `group_id` column → **Sequential Challenge**
  - Otherwise → **Flat Challenge**

> **Note:** GPU is required for training. Run this on an AWS EC2 instance such as `g5.2xlarge`.

### Example Commands
#### 🔹 Flat Training
```bash
python scripts/parametric_train.py flat-training.csv
```
#### 🔹 Sequential Training
```bash
python scripts/parametric_train.py sequential-training.csv
```

---

## 📂 Output Format

Once training completes:

- A CSV will be saved in an automatically created `output/` folder.
- Output file naming convention:  
  ```
  [challenge_type]_[estimated_accuracy].csv
  ```
  where:
  - `[challenge_type]` is either `flat` or `seq`
  - `[estimated_accuracy]` is a 6-digit float (e.g., `0.941238`)

### Examples
1. Flat:  
   Input → `flat-training.csv`  
   Output → `output/flat_0.941238.csv`

2. Sequential:  
   Input → `sequential-training.csv`  
   Output → `output/seq_0.928417.csv`

> 📌 The output folder is always created in the directory where you run the training script, regardless of input file location.

---

## 🧪 Test Datasets

Stage 1 sample datasets are available at:
```
scripts/stage_1_datasets/
```
Use these to test your setup and verify correct output.

---

## 📄 License & Attribution

- **Base Engine**: Cloned from [Mostly AI Engine](https://github.com/mostly-ai/mostlyai-engine)  
  → See `ORIGINAL_README.md` for details on the original package.

- **License**: Modifications are released under the Apache 2.0 license (see `LICENSE`).

- **Engine Modifications**:
  - Key files altered:  
    - `mostlyai/engine/_tabular/training.py`  
    - `mostlyai/engine/_tabular/argn.py`  
  - Purpose: Performance tuning for the Mostly AI Prize  
  - Review the commit history for details on the changes.