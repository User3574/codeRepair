import numpy as np
import functools

# Cache it
@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    if "codellama" in tokenizer.name_or_path:
        return (
            tokenizer.bos_token_id,
            tokenizer.suffix_id,
            tokenizer.prefix_id,
            tokenizer.middle_id,
            0,
        )
    elif "deepseek-coder" in tokenizer.name_or_path:
        return (
            tokenizer.bos_token_id,
            tokenizer.encode("<｜fim▁hole｜>", add_special_tokens=False)[0],
            tokenizer.encode("<｜fim▁begin｜>", add_special_tokens=False)[0],
            tokenizer.encode("<｜fim▁end｜>", add_special_tokens=False)[0],
            tokenizer.encode("<pad>", add_special_tokens=False)[0],
        )
    elif "starcoder" in tokenizer.name_or_path:
        return (
            tokenizer.bos_token_id,
            tokenizer.encode("<fim_suffix>")[0],
            tokenizer.encode("<fim_prefix>")[0],
            tokenizer.encode("<fim_middle>")[0],
            tokenizer.encode("<fim_pad>")[0],
        )
    else:
        print('Unknown FIM tokenizer')
        
    return None


# Add the BOS token to the beginning of the list
def _bos_token_processing(prefix_token_list, bos_token):
    if bos_token is not None:
        prefix_token_list.insert(0, bos_token)
    return prefix_token_list


## Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    sample,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
    bos_token_id=None,
):
    prefix, middle, suffix = sample

    if truncate_or_pad:
        new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
        diff = new_length - len(sample)
        if diff > 0:
            if suffix.shape[0] <= diff:
                return sample
            suffix = suffix[: suffix.shape[0] - diff]
        elif diff < 0:
            suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

    # SPM
    if np.random.rand() < fim_spm_rate:
        prefix_special_tokens = _bos_token_processing(
            [prefix_tok_id, suffix_tok_id], bos_token_id
        )
        new_sample = np.concatenate(
            [
                prefix_special_tokens,
                suffix,
                [middle_tok_id],
                prefix,
                middle,
            ]
        )
    # PSM
    else:
        prefix_special_tokens = _bos_token_processing([prefix_tok_id], bos_token_id)
        new_sample = np.concatenate(
            [
                prefix_special_tokens,
                prefix,
                [suffix_tok_id],
                suffix,
                [middle_tok_id],
                middle,
            ]
        )
    return list(new_sample)
