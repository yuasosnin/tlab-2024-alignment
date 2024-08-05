import torch
from trl.trainer.utils import first_true_indices


class BertRewardGetter:
    def __init__(self, policy_tokenizer, reward_tokenizer):
        self.policy_tokenizer = policy_tokenizer
        self.reward_tokenizer = reward_tokenizer

    def get_reward(self, model, query_responses, pad_token_id, context_length):
        sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length

        attention_mask = query_responses != pad_token_id
        attention_mask[:, context_length:] = attention_mask[:, context_length:].cumprod(1)
        input_ids = torch.masked_fill(query_responses, ~attention_mask, pad_token_id)

        sentences = self.policy_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        tokenizer_output = self.reward_tokenizer(sentences, padding=True, return_tensors="pt").to(model.device)

        output = model(tokenizer_output.input_ids, tokenizer_output.attention_mask, return_dict=True)
        return (None, output.logits[:,1], sequence_lengths)
