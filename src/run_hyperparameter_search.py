"""Run a hyperparameter search on a RoBERTa model fine-tuned on BoolQ.

Example usage:
    python run_hyperparameter_search.py BoolQ/
"""
import argparse
import boolq
import data_utils
import finetuning_utils
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizerFast,
    TrainingArguments,
    Trainer
    )
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray import tune
from ray.tune import CLIReporter

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the BoolQ dataset."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing the BoolQ dataset. Can be downloaded from https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip.",
)

args = parser.parse_args()

# Since the labels for the test set have not been released, we will use half of the
# validation dataset as our test dataset for the purposes of this assignment.
train_df = pd.read_json(f"{args.data_dir}/train.jsonl", lines=True, orient="records")
val_df, test_df = train_test_split(
    pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
    test_size=0.5,
)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
train_data = boolq.BoolQDataset(train_df, tokenizer)
val_data = boolq.BoolQDataset(val_df, tokenizer)
test_data = boolq.BoolQDataset(test_df, tokenizer)

## TODO: Initialize a transformers.TrainingArguments object here for use in
## training and tuning the model. Consult the assignment handout for some
## sample hyperparameter values.
training_args = TrainingArguments(
    output_dir="/scratch/lp2535/spring2022/nlu/hw/nlu_hw3/data/checkpoints/",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,
    # per_gpu_eval_batch_size=64,
    num_train_epochs=3,
    logging_steps=500,
    logging_first_step=True,
    save_steps=1000,
    evaluation_strategy = "epoch", # evaluate at the end of every epoch
)
trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=val_data,
    model_init=finetuning_utils.model_init,
    compute_metrics=finetuning_utils.compute_metrics,
)

# Default objective is the sum of all metrics
# when metrics are provided, so we have to maximize it.
tune_config = {
    'learning_rate': tune.uniform(1e-5, 5e-5)
}
# bayes_search = BayesOptSearch(
#    metric="objective", mode="max"
#    )
bayes_search = BasicVariantGenerator()
reporter = CLIReporter(
        parameter_columns={
            "learning_rate": "lr",
            "num_train_epochs": "num_epochs"
        },
        metric_columns=[
            "eval_accuracy", "eval_f1", "eval_loss"
        ])

best_model = trainer.hyperparameter_search(
    hp_space=lambda _: tune_config,
    search_alg=bayes_search,
    direction="maximize", 
    backend="ray", 
    n_trials=5,
    compute_objective= lambda metrics : metrics["eval_accuracy"],
    resources_per_trial={"cpu": 1, "gpu": 1},
    progress_reporter=reporter,
    local_dir="/scratch/lp2535/spring2022/nlu/hw/nlu_hw3/data/ray_reports",
    name="tune_transformer_lore"
)
print(best_model)
