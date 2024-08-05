import random
from time import time
from pathlib import Path
from copy import deepcopy
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset

from src.reinforce_trainer import ReinforceTrainer
from src.reinforce_config import ReinforceConfig
from src.get_reward import BertRewardGetter
from src.model_ops import model_add_, model_sub_, model_lerp_, model_slerp_
from src.reinforce_trainer import generate, score_sequence


random.seed(1)
np.random.seed(1)
torch.random.manual_seed(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--policy_name_or_path",
        type=str,
        default="lvwerra/gpt2-imdb",
    )
    parser.add_argument(
        "--reward_model_name_or_path",
        type=str,
        default="lvwerra/distilbert-imdb",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--ema_beta",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--kl_coef",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
    )
    return parser.parse_args()


def main(args):
    # load policy
    policy = AutoModelForCausalLM.from_pretrained(args.policy_name_or_path)
    ref_policy = AutoModelForCausalLM.from_pretrained(args.policy_name_or_path)
    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_name_or_path, padding_side="left")
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

    # load reward
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_name_or_path)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name_or_path)

    # prepare dataset
    def tokenize(examples):
        prompt_len = random.randint(5, 15)
        examples["input_ids"] = policy_tokenizer.encode(examples["text"])[:prompt_len]
        examples["query"] = policy_tokenizer.decode(examples["input_ids"])
        return examples

    dataset = load_dataset("stanfordnlp/imdb")
    prompt_dataset = dataset.map(tokenize, batched=False)

    test_dataset = prompt_dataset["test"].select(range(100))
    test_dataloader = DataLoader(
        test_dataset.remove_columns(["text", "label", "query"]),
        batch_size=64,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(policy_tokenizer),
    )

    train_dataset = prompt_dataset["train"].remove_columns(["text", "label", "query"])
    eval_dataset = prompt_dataset["test"].remove_columns(["text", "label", "query"])
    eval_dataset = eval_dataset.train_test_split(0.1, seed=1)["test"]

    # prepare trainer
    training_args = ReinforceConfig(
        output_dir="_",
        num_sample_generations=0,

        per_device_train_batch_size=args.batch_size,
        local_rollout_forward_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        seed=1,

        rloo_k=1,
        num_ppo_epochs=1,
        num_mini_batches=1,
        ema_beta=args.ema_beta,
        kl_coef=args.kl_coef,
        temperature=args.temperature,
    )
    training_args.total_episodes = args.train_steps * training_args.per_device_train_batch_size
    training_args.iterations = args.iterations
    training_args.runs = args.runs
    training_args.eta = args.eta

    policy_init = AutoModelForCausalLM.from_pretrained(args.policy_name_or_path).cuda()
    timestamp = int(time())

    for i in range(training_args.iterations):
        for m in range(training_args.runs):
            # initialize policies
            policy_m = deepcopy(policy_init)
            policy_ref_m = deepcopy(policy_init)

            # perform training
            train_dataset = train_dataset.shuffle(seed=1)
            trainer = ReinforceTrainer(
                training_args,
                tokenizer=policy_tokenizer,
                policy=policy_m,
                ref_policy=policy_ref_m,
                reward_model=reward_model,
                get_reward_func=BertRewardGetter(policy_tokenizer, reward_tokenizer).get_reward,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )
            trainer.args.output_dir = f"models/warp/{timestamp}/i{i+1}m{m+1}"
            trainer.train()
            policy_m.save_pretrained(Path(trainer.args.output_dir) / "policy", from_pt=True)
            policy_ref_m.save_pretrained(Path(trainer.args.output_dir) / "ref_policy", from_pt=True)
            trainer.save_state()

            # compute task vector inplace
            model_sub_(policy_m, policy_init)
            # collect slerp iterations inplace
            if m == 0:
                policy_slerp = policy_m
            else:
                model_slerp_(policy_slerp, policy_m, coeff=1/training_args.runs)
        # add init back to complete slerp
        model_add_(policy_slerp, policy_init)
        # update init inplace
        model_lerp_(policy_init, policy_slerp, coeff=training_args.eta)

    policy_init.save_pretrained(f"models/warp/{timestamp}/final/policy", from_pt=True)
    with open(f"models/warp/{timestamp}/args.json", "w") as f:
        f.write(trainer.args.to_json_string())

    # evaluate
    def evaluate_policy(policy, ref_policy, test_dataloader):
        rewards = 0
        kls = 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                prompts = data["input_ids"].to(policy.device)
                prompt_length = prompts.shape[1]
                prompts_responses, _ = generate(
                    policy,
                    prompts,
                    policy_tokenizer.pad_token_id,
                    trainer.generation_config,
                )
                logprobs = score_sequence(
                    policy,
                    prompts_responses,
                    prompt_length,
                    pad_token_id=policy_tokenizer.pad_token_id,
                    temperature=training_args.temperature
                )
                ref_logprobs = score_sequence(
                    ref_policy,
                    prompts_responses,
                    prompt_length,
                    pad_token_id=policy_tokenizer.pad_token_id,
                    temperature=training_args.temperature
                )
                _, reward, _ = trainer.get_reward(
                    reward_model, prompts_responses, policy_tokenizer.pad_token_id, prompt_length
                )
                kl = (logprobs.exp() * (logprobs - ref_logprobs)).sum(1)

                rewards += reward.sum().item()
                kls += kl.sum().item()
                total += prompts.shape[0]
        rewards /= total
        kls /= total
        return rewards, kls

    rewards, kls = 0, 0
    for _ in range(10):
        reward, kl = evaluate_policy(policy_init, ref_policy, test_dataloader)
        rewards += reward
        kls += kl
    rewards /= 10
    kls /= 10

    with open(f"models/warp/{timestamp}/metrics.csv", "w") as f:
        f.write(f"{rewards},{kls}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
