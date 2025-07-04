# Mostly AI Prize 💎 - Tecnarca's Take

This codebase is a clone of the Mostly AI Engine. 

It was created to directly modify the engine in order to compete in 2025's [Mostly AI Prize](https://www.mostlyaiprize.com/).

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tecnarca/mostlyai-engine-prize.git
   cd mostlyai-engine-prize
   ```

2. **Install `uv` (if not installed already)**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   For alternative installation methods, visit the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

3. **Create a virtual environment and install dependencies**:
   If using GPU, run:
   ```bash
   uv sync --frozen --extra gpu --python=3.10
   source .venv/bin/activate
   ```

## Running a training job

The `scripts/parametric_train.py` allows you to run a training job for both the Sequential and the Flat challenge.

The script has only one parameter, containing the path to the training dataset (readable with `pandas.read_csv`)
that you want to train the model on.
The script will automatically distinguish between a `flat` and `sequential` 
dataset solely off of the presence (or absence) of the `group_id` column.

This script, regardless of the challenge, is intended to be trained on a GPU,
so you should run these submission in a `g5.2xlarge` AWS EC2 instance.

#### Commands for training
   1. Flat
      ```bash
      python scripts/parametric_train.py flat-training.csv 
      ```
   2. Sequential
      ```bash
      python scripts/parametric_train.py sequential-training.csv 
      ```

When training is complete a csv file with identical size (for flat) or identical number of groups (for sequential)
can be found in a `output` folder that will be created. 
The filename will be in the format `[challenge_type]_[estimated_accuracy].csv` where `[challenge_type]` can be
`flat` or `seq` and `[estimated_accuracy]` will be a float number with 6 digits after comma representing the estimated
training overall accuracy.

Examples:
1. running `python scripts/parametric_train.py flat-training.csv` will create `output/flat_0.XXXXXX.csv` 
2. running `python scripts/parametric_train.py sequential-training.csv` will create `output/seq_0.XXXXXX.csv` 

Note that there is no relation from the input csv folder to the output folder. 
The output folder will always be in the folder you launch the training command from.

I have uploaded the Stage 1 datasets in `scripts/stage_1_datasets` in case you want to test the above commands.

### License

The whole repository was cloned from [here](https://github.com/mostly-ai/mostlyai-engine), 
for more information on the original package you can check out `ORIGINAL_README.md`

The modifications to the engine are released under the Apache 2.0 license, see `LICENSE` for more information.

The `mostlyai/engine/_tabular/training.py` and `mostlyai/engine/_tabular/argn.py` files were modified to improve performance
in the MOSTLY AI Prize competition. See commit history for more information on how the engine code was adapted.
