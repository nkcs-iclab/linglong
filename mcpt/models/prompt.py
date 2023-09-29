import torch

from .basic import *


class MCPTPromptModel(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.n_embd = config['n_embd']
        self.wte = nn.Embedding(config['n_vocab'], self.n_embd)
        self.wpe = nn.Embedding(config['n_ctx'], self.n_embd)
        self.drop = nn.Dropout(config['embd_dropout'])
        self.blocks = nn.ModuleList([Block(config, blk_idx=i) for i in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(self.n_embd, eps=config['epsilon'])
        self.prompt_length = self.config['prompt_length']
        self.pseudo_token_id = config['pseudo_token_id']
        self.prompt_emb = nn.Embedding(self.prompt_length, self.n_embd)
        self.prompt_encoder_type = config['prompt_encoder_type']
        if self.prompt_encoder_type == "lstm":
            self.lstm_head = torch.nn.LSTM(input_size=self.n_embd,
                                           hidden_size=self.n_embd,
                                           num_layers=2,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(2 * self.n_embd, self.n_embd),
                                          nn.ReLU(),
                                          nn.Linear(self.n_embd, self.n_embd))
        elif self.prompt_encoder_type == "mlp":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.n_embd, self.n_embd),
                torch.nn.ReLU(),
                torch.nn.Linear(self.n_embd, self.n_embd))
        else:
            raise ValueError('unknown prompt_encoder_type.')

    def forward(self, inputs, past=None):
        bz = inputs.shape[0]
        if past is None:
            past_length = 0
            pasts = tuple([None] * len(self.blocks))
        else:
            past_length = past.size(-2)
            pasts = torch.unbind(past, dim=1)
        position_ids = torch.arange(past_length, inputs.size(-1) + past_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).view(-1, inputs.size(-1))
        inputs_embd = self.wte(inputs)
        position_embd = self.wpe(position_ids)
        if self.pseudo_token_id in inputs:
            replace_embeds = self.prompt_emb(torch.IntTensor([i for i in range(self.prompt_length)]).cuda()).unsqueeze(
                0)
            if self.prompt_encoder_type == "lstm":
                replace_embeds = self.lstm_head(replace_embeds)[0]
                replace_embeds = self.mlp_head(replace_embeds).squeeze()
            else:
                replace_embeds = self.mlp(replace_embeds).squeeze()
            blocked_indices = (inputs == self.pseudo_token_id).nonzero().reshape((bz, self.prompt_length, 2))[:, :, 1]  # bz
            for bidx in range(bz):
                for i in range(self.prompt_length):
                    inputs_embd[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        h = inputs_embd + position_embd
        h = self.drop(h)
        presents = []
        for blk_idx, past in enumerate(pasts):
            h, present = self.blocks[blk_idx](h, past)
            presents.append(present)
        present = torch.stack(presents, dim=1)
        h = self.ln_f(h)
        # TODO: 需要适配最新版本的 Model 输出！
        return {
            'hidden_states': h,
            'present': present,
        }
