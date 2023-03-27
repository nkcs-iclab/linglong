import torch

from typing import *


def _top_k_logits(logits, k):
    if k == 0:
        return logits
    logits_flat = logits.view(-1, logits.size(-1))
    indices = torch.topk(logits_flat, k, dim=-1)[1]
    mask = torch.zeros_like(logits_flat).scatter_(-1, indices, 1)
    mask = mask.view(logits.size())
    logits = logits.masked_fill((mask == 0), -1e10)
    return logits


def _top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    n_remove = sorted_indices_to_remove.sum(dim=-1)
    min_logits = sorted_logits.flip(-1).gather(-1, n_remove.unsqueeze(-1))
    logits[logits < min_logits] = -1e10
    return logits


def _past_shape(
        model_config: Dict[str, Any],
        batch_size: int = -1,
        n_ctx: int = -1,
) -> List[int]:
    return [
        batch_size,
        model_config['n_layer'],
        2,
        model_config['n_head'],
        n_ctx,
        model_config['n_embd'] // model_config['n_head'],
    ]


def sample(
        model_config: Dict[str, Any],
        config: Dict[str, Any],
        prompt_ids,
        model,
        end_id: int,
        device: str,
        candidates=None,
        tokenizer=None,
        pinyin_tokenizer=None,
        use_pinyin: bool = False,
        callback: Optional[Callable] = None,
):
    length = config.get('length')
    batch_size = config.get('batch_size', 1)
    temperature = config.get('temperature', 1.0)
    top_k = config.get('top_k', 1)
    top_p = config.get('top_p', 0.0)

    prompt_ids = torch.tensor(prompt_ids, dtype=torch.int32).to(device)
    context = prompt_ids.repeat(batch_size, 1, 1) if use_pinyin else prompt_ids.repeat(batch_size, 1)

    past = None
    prev = context
    output = context
    for i in range(length):
        with torch.no_grad():
            logits, presents = model(prev, past=past)
        presents = presents.view(_past_shape(model_config=model_config, batch_size=batch_size))
        logits = logits[:, -1]
        if candidates is None:
            logits = logits / temperature
            logits = _top_k_logits(logits, k=top_k)
            logits = _top_p_logits(logits, p=top_p)
            log_probs = torch.nn.functional.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1)
        else:
            target_logits = logits[:, candidates]
            arg_preds = torch.argmax(target_logits, dim=-1)
            prev = torch.tensor(candidates)[arg_preds].unsqueeze(-1).to(device)
        if use_pinyin:
            generated_tokens = tokenizer.convert_ids_to_tokens(prev.view((-1,)))
            pinyin_ids = pinyin_tokenizer.convert_tokenizer_tokens_to_ids(generated_tokens)
            pinyin_ids = pinyin_ids.view((-1, 1))
            prev = torch.cat((prev, pinyin_ids), dim=1).view((-1, 2, 1))
        if callback is not None:
            callback(prev[:, 0] if use_pinyin else prev)
        past = presents if past is None else torch.cat((past, presents), dim=-2)
        output = torch.cat((output, prev), dim=-1)
        if batch_size == 1 and (prev[0][0].item() if use_pinyin else prev.item()) == end_id:
            break

    return output[:, 0] if use_pinyin else output
