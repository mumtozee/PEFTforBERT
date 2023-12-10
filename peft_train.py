import argparse
import ast
import gc
import random
import re
import json
import yaml
import pickle
import os
from pathlib import Path
from collections import defaultdict
from time import time
from dataclasses import dataclass
from tqdm import tqdm
import typing as tp
from itertools import chain
import shutil
import logging
import collections.abc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets
import transformers
from datasets import load_dataset, DatasetDict, ReadInstruction
from transformers import (
    AutoModelForMaskedLM,
    BertForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import evaluate

from peft import (
    get_peft_config,
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
    PromptTuningInit,
    PromptTuningConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    IA3Config,
    prepare_model_for_int8_training,
    PeftModelForTokenClassification,
)

logger = get_logger(__name__)


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def clear_peft_config(cfg: tp.Dict[str, tp.Any]):
    if cfg["peft_type"] is None:
        cfg["PeftArgs"] = None
        return
    keys = [k for k in cfg["PeftArgs"].keys() if k != cfg["peft_type"]]
    for k in keys:
        cfg["PeftArgs"].pop(k, None)


def gen_peft_config(config: tp.Dict[str, tp.Any]) -> PeftConfig:
    conf_dict = config["PeftArgs"][config["peft_type"]]
    peft_type = config["peft_type"]
    result = None
    if peft_type != "qlora" and config["quantized"]:
        raise ValueError("Quantization only supported with LoRa.")
    if peft_type == "qlora" and not config["quantized"]:
        raise ValueError("QLoRa needs quantization flag.")
    if peft_type == "lora" or peft_type == "qlora":
        result = LoraConfig(
            task_type=conf_dict["task_type"],
            inference_mode=conf_dict["inference_mode"],
            r=conf_dict["r"],
            lora_alpha=conf_dict["lora_alpha"],
            lora_dropout=conf_dict["lora_dropout"],
            bias=conf_dict["bias"],
        )
    elif peft_type == "prefix_tuning":
        result = PrefixTuningConfig(
            task_type=conf_dict["task_type"],
            inference_mode=conf_dict["inference_mode"],
            num_virtual_tokens=conf_dict["num_virtual_tokens"],
        )
    elif peft_type == "prompt_tuning":
        result = PromptTuningConfig(
            task_type=conf_dict["task_type"],
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=conf_dict["num_virtual_tokens"],
            tokenizer_name_or_path=config["model_name_or_path"],
            prompt_tuning_init_text=conf_dict["prompt_tuning_init_text"],
        )
    elif peft_type == "p_tuning":
        result = PromptEncoderConfig(
            task_type=conf_dict["task_type"],
            num_virtual_tokens=conf_dict["num_virtual_tokens"],
            encoder_hidden_size=conf_dict["encoder_hidden_size"],
        )
    elif peft_type == "ia3":
        result = IA3Config(
            task_type=conf_dict["task_type"],
            target_modules=conf_dict["target_modules"],
            feedforward_modules=conf_dict["feedforward_modules"],
        )
    return result


def get_args(args: argparse.Namespace) -> tp.Dict[str, tp.Any]:
    config = None
    user_cfg = None
    with open("./default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(args.config_path, "r") as f:
        user_cfg = yaml.safe_load(f)
    update(config, user_cfg)
    clear_peft_config(config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.lr is not None:
        config["TrainerArgs"]["lr"] = args.lr
    if args.n_epoch is not None:
        config["TrainerArgs"]["n_epoch"] = args.n_epoch
    if args.weight_decay is not None:
        config["TrainerArgs"]["weight_decay"] = args.weight_decay
    if args.warmup_ratio is not None:
        config["TrainerArgs"]["warmup_ratio"] = args.warmup_ratio
    if args.gradient_accumulation_steps is not None:
        config["TrainerArgs"][
            "gradient_accumulation_steps"
        ] = args.gradient_accumulation_steps
    if args.resume_train is not None:
        config["TrainerArgs"]["resume"] = args.resume_train
    if args.train_batch_size is not None:
        config["TrainerArgs"]["per_device_train_batch_size"] = args.train_batch_size
    if args.eval_batch_size is not None:
        config["TrainerArgs"]["per_device_eval_batch_size"] = args.eval_batch_size
    if args.checkpoint_dir is not None:
        config["TrainerArgs"]["checkpoint_dir"] = args.checkpoint_dir
    return config


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="A '.yaml' file with all necessary configurations",
    )
    parser.add_argument("--lr", type=float)
    parser.add_argument("--n_epoch", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-train", type=bool)
    parser.add_argument("--checkpoint-dir", type=str)
    return parser


def get_last_checkpoint(checkpoint_dir: str) -> str:
    max_step = 0
    for dir in Path(checkpoint_dir).iterdir():
        if not os.path.isdir(dir):
            pass
        basename = os.path.basename(str(dir))
        cur_step = int(basename)
        if cur_step > max_step:
            max_step = cur_step
    return os.path.join(checkpoint_dir, str(max_step))


def training_loop(
    model: AutoModelForMaskedLM,
    accelerator: Accelerator,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: tp.Dict[str, tp.Any],
):
    train_cfg = config["TrainerArgs"]
    if not os.path.exists(train_cfg["checkpoint_dir"]):
        os.makedirs(train_cfg["checkpoint_dir"])
    # Intantiate the optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": train_cfg["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=train_cfg["lr"])
    overrode_max_train_steps = False
    num_update_steps_per_epoch = (len(train_dataloader) - 1) // train_cfg[
        "gradient_accumulation_steps"
    ] + 1
    if train_cfg["max_steps"] is None:
        train_cfg["max_steps"] = train_cfg["n_epoch"] * num_update_steps_per_epoch
        overrode_max_train_steps = True
    # Instantiate the learning rate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(
            train_cfg["warmup_ratio"]
            * train_cfg["max_steps"]
            * train_cfg["gradient_accumulation_steps"]
        ),
        num_training_steps=train_cfg["max_steps"]
        * train_cfg["gradient_accumulation_steps"],
    )
    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = (len(train_dataloader) - 1) // train_cfg[
        "gradient_accumulation_steps"
    ] + 1
    if overrode_max_train_steps:
        train_cfg["max_steps"] = train_cfg["n_epoch"] * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    train_cfg["n_epoch"] = (
        train_cfg["max_steps"] - 1
    ) // num_update_steps_per_epoch + 1

    total_batch_size = (
        train_cfg["per_device_train_batch_size"]
        * accelerator.num_processes
        * train_cfg["gradient_accumulation_steps"]
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {train_cfg['n_epoch']}")
    logger.info(
        f"  Instantaneous batch size per device = {train_cfg['per_device_train_batch_size']}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {train_cfg['gradient_accumulation_steps']}"
    )
    logger.info(f"  Total optimization steps = {train_cfg['max_steps']}")

    progress_bar = tqdm(
        range(train_cfg["max_steps"]), disable=not accelerator.is_local_main_process
    )
    cur_step = 0
    # Now we train the model
    for epoch in range(train_cfg["n_epoch"]):
        clf_metrics = evaluate.load("accuracy")
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            # batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with accelerator.accumulate(model):
                if (
                    config["peft_type"] == "prompt_tuning"
                    or config["peft_type"] == "p_tuning"
                ):
                    batch["labels"] = F.pad(
                        batch["labels"],
                        pad=(
                            config["PeftArgs"][config["peft_type"]][
                                "num_virtual_tokens"
                            ],
                            0,
                        ),
                        mode="constant",
                        value=-100,
                    )
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), train_cfg["max_grad_norm"]
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                cur_step += 1

            if cur_step % train_cfg["log_steps"] == 0:
                labels = accelerator.gather(batch["labels"])
                idx = labels != -100
                preds = accelerator.gather(torch.argmax(outputs.logits, dim=-1))[idx]
                refs = labels[idx]
                res = clf_metrics.compute(predictions=preds, references=refs)
                accelerator.print(
                    f'Epoch {epoch} Step {cur_step}   Train loss: {accelerator.gather(loss).mean().item() :.3f}, Acc: {res["accuracy"] :.3f}'
                )

            if cur_step % train_cfg["eval_steps"] == 0:
                model.eval()
                clf_metrics = evaluate.load("accuracy")
                for _, batch in enumerate(eval_dataloader):
                    # We could avoid this line since we set the accelerator with `device_placement=True`.
                    # batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    if (
                        config["peft_type"] == "prompt_tuning"
                        or config["peft_type"] == "p_tuning"
                    ):
                        batch["labels"] = F.pad(
                            batch["labels"],
                            pad=(
                                config["PeftArgs"][config["peft_type"]][
                                    "num_virtual_tokens"
                                ],
                                0,
                            ),
                            mode="constant",
                            value=-100,
                        )
                    with torch.no_grad():
                        outputs = model(**batch)
                    loss = outputs.loss
                    labels = accelerator.gather(batch["labels"])
                    idx = labels != -100
                    preds = accelerator.gather(torch.argmax(outputs.logits, dim=-1))[
                        idx
                    ]
                    refs = labels[idx]
                    clf_metrics.add_batch(predictions=preds, references=refs)
                res = clf_metrics.compute()
                accelerator.print(
                    f'Epoch {epoch} Step {cur_step}   Val loss: {accelerator.gather(loss).mean().item() :.3f}, Acc: {res["accuracy"] :.3f}'
                )
                model.train()
            if cur_step % train_cfg["save_steps"] == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                os.makedirs(
                    os.path.join(train_cfg["checkpoint_dir"], str(cur_step)),
                    exist_ok=True,
                )
                unwrapped_model.save_pretrained(
                    os.path.join(train_cfg["checkpoint_dir"], str(cur_step)),
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    safe_serialization=True,
                )
                with open(
                    os.path.join(
                        train_cfg["checkpoint_dir"], str(cur_step), "config.yaml"
                    ),
                    "w+",
                ) as f:
                    yaml.safe_dump(config, f)
            if cur_step >= train_cfg["max_steps"]:
                return


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    config = get_args(args)
    train_cfg = config["TrainerArgs"]
    logger_cfg = config["logger"]

    if config["quantized"]:
        accelerator = Accelerator(
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            mixed_precision="bf16",
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"]
        )
    logging.basicConfig(
        format=logger_cfg["format"],
        datefmt=logger_cfg["datefmt"],
        level=logger_cfg["level"],
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if "seed" in config and config["seed"] is not None:
        set_seed(config["seed"])

    if accelerator.is_main_process:
        if "checkpoint_dir" in train_cfg and train_cfg["checkpoint_dir"] is not None:
            os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)
    accelerator.wait_for_everyone()
    oscar_raw = load_dataset(
        "wikitext",
        "wikitext-103-v1",
        split=ReadInstruction("train", to=10000, unit="abs"),
    )
    toker = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    if train_cfg["resume"]:
        if "checkpoint_dir" not in train_cfg or train_cfg["checkpoint_dir"] is None:
            raise ValueError("Cannot resume training if no checkpoint dir provided.")
        chkpt_name = get_last_checkpoint(train_cfg["checkpoint_dir"])
        if config["peft_type"] is not None:
            peft_cfg = PeftConfig.from_pretrained(chkpt_name)
            if not config["quantized"]:
                model = AutoModelForMaskedLM.from_pretrained(
                    peft_cfg.base_model_name_or_path
                )
            else:
                model = AutoModelForMaskedLM.from_pretrained(
                    peft_cfg.base_model_name_or_path, load_in_8bit=True
                )
                model = prepare_model_for_int8_training(model)
            model = PeftModel.from_pretrained(model, chkpt_name)
            if config["peft_type"] == "prefix_tuning":
                model.cls_layer_name = config["cls_layer_name"]
                model.num_labels = config["num_labels"]
        else:
            model = AutoModelForMaskedLM.from_pretrained(chkpt_name)
    else:
        if not config["quantized"]:
            model = AutoModelForMaskedLM.from_pretrained(config["model_name_or_path"])
        else:
            model = AutoModelForMaskedLM.from_pretrained(
                config["model_name_or_path"], load_in_8bit=True
            )
            model = prepare_model_for_int8_training(model)
        if config["peft_type"] is not None:
            peft_config = gen_peft_config(config)
            model = get_peft_model(model, peft_config)
            if config["peft_type"] == "prefix_tuning":
                model.cls_layer_name = config["cls_layer_name"]
                model.num_labels = config["num_labels"]

    if config.get("max_seq_length", None) is None:
        max_seq_length = toker.model_max_length
        if max_seq_length > 1024:
            max_seq_length = 1024
    else:
        if config["max_seq_length"] > toker.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({config['max_seq_length']}) is larger than the maximum length for the "
                f"model ({toker.model_max_length}). Using max_seq_length={toker.model_max_length}."
            )
        max_seq_length = min(config["max_seq_length"], toker.model_max_length)

    def tokenize(batch: tp.Dict[str, tp.Iterable]):
        encoded = toker(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
        return encoded

    def group_texts(examples: tp.Dict[str, tp.Iterable]):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    with accelerator.main_process_first():
        oscar_enc = oscar_raw.map(
            tokenize, batched=True, num_proc=4, remove_columns=oscar_raw.column_names
        )
        oscar_grouped = oscar_enc.map(
            group_texts, batched=True, batch_size=2000, num_proc=4
        )
    ds_train_devtest = oscar_grouped.train_test_split(test_size=0.2, seed=42)
    ds_devtest = ds_train_devtest["test"].train_test_split(test_size=0.5, seed=42)
    ds_splits = DatasetDict(
        {
            "train": ds_train_devtest["train"],
            "valid": ds_devtest["train"],
            "test": ds_devtest["test"],
        }
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=toker,
        mlm=True,
        return_tensors="pt",
        mlm_probability=config["mlm_proba"],
    )
    train_dataloader = DataLoader(
        ds_splits["train"],
        shuffle=True,
        collate_fn=collator,
        batch_size=train_cfg["per_device_train_batch_size"],
    )
    eval_dataloader = DataLoader(
        ds_splits["valid"],
        collate_fn=collator,
        batch_size=train_cfg["per_device_eval_batch_size"],
    )
    training_loop(model, accelerator, train_dataloader, eval_dataloader, config)


if __name__ == "__main__":
    main()
