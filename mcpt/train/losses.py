import torch


def reward_loss(inputs, rewards):
    chosen_mean_scores = []
    rejected_mean_scores = []

    # Split the inputs and rewards into two parts, chosen and rejected.
    assert len(inputs.shape) == 2
    batch_size = inputs.shape[0] // 2
    seq_len = inputs.shape[1]

    chosen_ids = inputs[:batch_size]  # batch_size x seq_len
    rejected_ids = inputs[batch_size:]  # batch_size x seq_len
    chosen_rewards = rewards[:batch_size]
    rejected_rewards = rewards[batch_size:]

    # Compute pairwise loss. Only backprop on the different tokens before padding.
    loss = 0
    for i in range(batch_size):
        chosen_id = chosen_ids[i]
        rejected_id = rejected_ids[i]
        chosen_reward = chosen_rewards[i]
        rejected_reward = rejected_rewards[i]

        c_idxes = (chosen_id == 0).nonzero()
        c_idx = c_idxes[0].item() if len(c_idxes) > 0 else seq_len
        check_divergence = (chosen_id != rejected_id).nonzero()

        if len(check_divergence) == 0:
            end_idx = rejected_reward.size(-1)
            divergence_idx = end_idx - 1
            r_idx = c_idx
        else:
            # Check if there is any padding otherwise take length of sequence.
            r_idxes = (rejected_id == 0).nonzero()
            r_idx = r_idxes[0].item() if len(r_idxes) > 0 else seq_len
            end_idx = max(c_idx, r_idx)
            divergence_idx = check_divergence[0]
        assert divergence_idx > 0
        c_truncated_reward = chosen_reward[divergence_idx:end_idx]
        r_truncated_reward = rejected_reward[divergence_idx:end_idx]
        chosen_mean_scores.append(chosen_reward[c_idx - 1])  # use the end score for reference
        rejected_mean_scores.append(rejected_reward[r_idx - 1])
        loss += -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()

    loss = loss / batch_size
    chosen_mean_scores = torch.stack(chosen_mean_scores)
    rejected_mean_scores = torch.stack(rejected_mean_scores)
    return {
        'loss': loss,
        'chosen_mean_scores': chosen_mean_scores,
        'rejected_mean_scores': rejected_mean_scores,
    }
