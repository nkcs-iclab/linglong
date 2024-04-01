from transformers import PretrainedConfig


class LingLongConfig(PretrainedConfig):
    model_type = 'linglong'

    def __init__(
            self,
            vocab_size: int = 13312,
            n_positions: int = 1024,
            n_embd: int = 1024,
            n_layer: int = 24,
            n_head: int = 16,
            n_inner: int | None = None,
            activation_function: str = 'gelu_new',
            resid_pdrop: float = 0.1,
            embd_pdrop: float = 0.1,
            attn_pdrop: float = 0.1,
            layer_norm_epsilon: float = 1e-8,
            initializer_range: float = 0.02,
            scale_attn_weights: bool = True,
            use_cache: bool = True,
            bos_token_id: int = 10,
            eos_token_id: int = 8,
            pad_token_id: int = 0,
            scale_attn_by_inverse_layer_idx: bool = False,
            reorder_and_upcast_attn: bool = False,
            attn_mode: str = 'sparse',
            attn_stride: int | None = 128,
            attn_c: int | None = 8,
            use_pinyin: bool = False,
            backward: bool = False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.attn_mode = attn_mode
        self.attn_stride = attn_stride
        self.attn_c = attn_c
        self.use_pinyin = use_pinyin
        self.backward = backward

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
