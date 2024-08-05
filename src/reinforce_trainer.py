import gc
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Callable

from tqdm import trange
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback

from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)

from src.reinforce_config import ReinforceConfig
from src.model_ops import model_lerp_


INVALID_LOGPROB = 1.0


def score_sequence(policy, prompts_responses, prompt_length, pad_token_id, temperature=0.7):
    responses = prompts_responses[:, prompt_length:]
    policy_output = forward(policy, prompts_responses, pad_token_id)
    response_logits = policy_output.logits[:, prompt_length - 1 : -1]
    response_logits /= temperature + 1e-7
    response_logprobs = F.log_softmax(response_logits, dim=-1)
    response_logprob = torch.gather(response_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
    return response_logprob


class ReinforceTrainer(Trainer):
    def __init__(
        self,
        config: ReinforceConfig,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        get_reward_func: Callable = get_reward,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        self.args = config
        args = config
        self.tokenizer = tokenizer
        self.policy = policy

        # disable `pad_token_id` and `eos_token_id` because we just want to
        # generate tokens without truncation / padding
        self.policy.generation_config.eos_token_id = None
        self.policy.generation_config.pad_token_id = None

        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.callbacks = callbacks
        self.get_reward = get_reward_func

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_updates = args.total_episodes // args.batch_size
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_updates // args.num_sample_generations)
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size, args.rloo_k, "`local_batch_size` must be a multiple of rloo_k"
        )  # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy, self.ref_policy, self.reward_model]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(num_training_steps=args.num_updates)

        #########
        ### trainer specifics
        #########
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        DEFAULT_CALLBACKS = [DefaultFlowCallback]
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        if self.callbacks is None:
            self.callbacks = default_callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer),
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.bf16, args.fp16
            )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.bf16, args.fp16
            )
            self.deepspeed = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    @property
    def generation_config(self):
        return GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(self.args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

    def train(self):
        args = self.args
        device = self.accelerator.device

        def repeat_generator():
            while True:
                yield from self.dataloader

        iter_dataloader = iter(repeat_generator())
        self.accelerator.print("===training policy===")
        global_step = 0
        start_time = time.time()
        self.policy.train()
        for update in trange(1, args.num_updates + 1):
            global_step += 1 * args.batch_size
            self.lr_scheduler.step()
            data = next(iter_dataloader)

            prompts = data["input_ids"].to(device)
            prompts = prompts.repeat(args.rloo_k, 1)
            prompt_length = prompts.shape[1]

            # generate responses on prompts
            with torch.no_grad():
                with unwrap_model_for_generation(self.policy, self.accelerator) as unwrapped_model:
                    prompts_responses, _ = generate(
                        unwrapped_model,
                        prompts,
                        self.tokenizer.pad_token_id,
                        self.generation_config,
                    )

            # get logprobs of generated sequences
            logprobs = score_sequence(
                self.policy,
                prompts_responses,
                prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=args.temperature
            )
            with torch.no_grad():
                ref_logprobs = score_sequence(
                    self.ref_policy,
                    prompts_responses,
                    prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=args.temperature
                )

            # postprocess responses
            responses = prompts_responses[:, prompt_length:]
            # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
            if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                responses = truncate_response(
                    args.stop_token_id, self.tokenizer.pad_token_id, responses
                )
                prompts_responses = torch.cat((prompts, responses), 1)

            # Response Processing 2. run reward model on the truncated responses
            sequence_lengths = first_true_indices(responses == self.tokenizer.pad_token_id) - 1
            _, rewards, _ = self.get_reward(
                self.reward_model, prompts_responses, self.tokenizer.pad_token_id, prompt_length
            )

            # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            contain_eos_token = torch.any(responses == self.tokenizer.eos_token_id, dim=-1)
            if args.non_eos_penalty:
                rewards = torch.where(contain_eos_token, rewards, torch.full_like(rewards, args.penalty_reward_value))

            # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

            # compute rewards
            with torch.no_grad():
                kl = logprobs - ref_logprobs
                non_score_reward = (-args.kl_coef * kl).sum(1)
                rlhf_reward = rewards + non_score_reward

                # vectorized RLOO advantages implementation
                rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
                # recover REINFORCE by setting rloo_k = 1
                baseline = 0
                if args.rloo_k > 1:
                    baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
                advantages = rlhf_reward - baseline
                advantages = advantages.flatten()
                torch.cuda.empty_cache()

            with self.accelerator.accumulate(self.policy):
                loss = torch.mean(-advantages * logprobs.sum(1))
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                # after updating policy parameters, update EMA for reference policy
                # not sure if works correctly with gradient accumulation!
                # if ema_beta == 1, nothing happens
                model_lerp_(self.ref_policy, self.policy, args.ema_beta)
            torch.cuda.empty_cache()

            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(global_step / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(rewards.mean()).mean().item()
                metrics["val/num_eos_tokens"] = (responses == self.tokenizer.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = global_step
                self.state.epoch = global_step / self.train_dataset_len  # used by self.log
                self.log(metrics)
            del kl, mean_kl, mean_entropy, rewards
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        tokenizer = self.tokenizer

        table = defaultdict(list)
        for batch in self.eval_dataloader:
            query = batch["input_ids"]
            with torch.no_grad():
                context_length = query.shape[1]
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    query_response, _ = generate(
                        unwrapped_model,
                        query,
                        tokenizer.pad_token_id,
                        self.generation_config,
                    )
                response = query_response[:, context_length:]
                postprocessed_response = response
                if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(args.stop_token_id, tokenizer.pad_token_id, response)
                table["query"].extend(gather_object(tokenizer.batch_decode(query, skip_special_tokens=True)))
                table["model response"].extend(gather_object(tokenizer.batch_decode(postprocessed_response)))

                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                _, score, _ = self.get_reward(
                    self.reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                )
                table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

            if sampling:
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df.iloc[0 : 0 + 5])
        if "wandb" in args.report_to:
            import wandb

            if wandb.run is not None:
                wandb.log({"completions": wandb.Table(dataframe=df)})
