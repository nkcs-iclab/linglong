from transformers.utils import logging

import linglong

logger = logging.get_logger(__name__)


def get_tokenizers(
        vocab_path: str | None = None,
        pinyin_vocab_path: str | None = None,
        pretrained_model: str | None = None,
        special_tokens: dict[str, str] | None = None,
        use_pinyin: bool = False,
        use_fast: bool = True,
        **kwargs,
) -> (
        tuple[linglong.LingLongTokenizer | linglong.LingLongTokenizerFast, linglong.PinyinTokenizer] |
        linglong.LingLongTokenizer |
        linglong.LingLongTokenizerFast
):
    def load_tokenizer_from_vocab() -> linglong.LingLongTokenizer | linglong.LingLongTokenizerFast:
        if vocab_path is None:
            raise ValueError('`vocab_path` must be provided if `pretrained_model` is None.')
        if use_fast:
            return linglong.LingLongTokenizerFast(vocab_path, **kwargs)
        return linglong.LingLongTokenizer(vocab_path, **kwargs)

    def load_pinyin_tokenizer_from_vocab() -> linglong.PinyinTokenizer:
        if pinyin_vocab_path is None:
            raise ValueError('`pinyin_vocab_path` must be provided if `pretrained_model` is None.')
        return linglong.PinyinTokenizer(
            vocab_file=pinyin_vocab_path,
            fallback=tokenizer,
            **kwargs,
        )

    if pretrained_model is not None:
        try:
            if use_fast:
                tokenizer = linglong.LingLongTokenizerFast.from_pretrained(pretrained_model, **kwargs)
            else:
                tokenizer = linglong.LingLongTokenizer.from_pretrained(pretrained_model, **kwargs)
            if vocab_path is not None:
                logger.warning(
                    f'Successfully loaded tokenizer from {pretrained_model}. '
                    f'Vocab file {vocab_path} is ignored.',
                )
        except (OSError, EnvironmentError):
            tokenizer = load_tokenizer_from_vocab()
            logger.warning(
                f'Cannot load tokenizer from {pretrained_model}. '
                f'Loading from vocab file {vocab_path}.'
            )
        try:
            if use_pinyin:
                pinyin_tokenizer = linglong.PinyinTokenizer.from_pretrained(
                    pretrained_model,
                    **kwargs,
                )
                if pinyin_vocab_path is not None:
                    logger.warning(
                        f'Successfully loaded pinyin tokenizer from {pretrained_model}. '
                        f'Vocab file {pinyin_vocab_path} is ignored.',
                    )
            else:
                pinyin_tokenizer = None
        except (OSError, EnvironmentError):
            pinyin_tokenizer = load_pinyin_tokenizer_from_vocab() if use_pinyin else None
            logger.warning(
                f'Cannot load pinyin tokenizer from {pretrained_model}. '
                f'Loading from vocab file {pinyin_vocab_path}.'
            )
    else:
        tokenizer = load_tokenizer_from_vocab()
        pinyin_tokenizer = load_pinyin_tokenizer_from_vocab() if use_pinyin else None
    if special_tokens is not None and len(new_special_tokens := list(
            set(special_tokens.values()) - set(pinyin_tokenizer.all_special_tokens),
    )) > 0:
        # noinspection PyTypeChecker
        tokenizer.add_special_tokens({
            'additional_special_tokens': new_special_tokens,
        })
        if use_pinyin:
            # noinspection PyTypeChecker
            pinyin_tokenizer.add_special_tokens({
                'additional_special_tokens': new_special_tokens,
            })
    if use_pinyin:
        return tokenizer, pinyin_tokenizer
    else:
        return tokenizer
