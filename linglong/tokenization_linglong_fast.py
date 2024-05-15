import string

from tokenizers import (
    Tokenizer as HFTokenizer,
    normalizers,
    pre_tokenizers,
    models,
    decoders,
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class LingLongTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = {'vocab_file': 'tokenizer.txt', 'tokenizer_file': 'tokenizer.json'}
    model_input_names = ['input_ids', 'attention_mask']

    class CustomDecoder:

        @staticmethod
        def decode_chain(tokens: list[str]) -> list[str]:
            new_tokens = []
            for token in tokens:
                if token.startswith('##'):
                    new_tokens.append(token[2:])
                else:
                    new_tokens.append(' ' + token)

            # Remove whitespaces between Chinese characters.
            # TODO: This will remove whitespaces between some English words as well. Need fix.
            alphabet_set = set(list(string.ascii_letters))
            for i in range(len(new_tokens)):
                if new_tokens[i][0] == ' ':
                    if new_tokens[i][1] not in alphabet_set or i == 0:
                        new_tokens[i] = new_tokens[i][1:]
            return new_tokens

    def __init__(
            self,
            vocab_file: str | None = None,
            tokenizer_file: str | None = None,
            do_lower_case: bool = True,
            do_basic_tokenize: bool = True,
            unk_token: str = '<unk>',
            sep_token: str = '<sep>',
            pad_token: str = '<pad>',
            cls_token: str = '<cls>',
            mask_token: str = '<mask>',
            bos_token: str = '<|startoftext|>',
            eos_token: str = '<|endoftext|>',
            tokenize_chinese_chars: bool = True,
            strip_accents: bool | None = None,
            **kwargs,
    ):
        backend_tokenizer = None
        if tokenizer_file is None:
            backend_tokenizer = HFTokenizer(
                models.WordPiece.from_file(
                    vocab=vocab_file,
                    unk_token=unk_token,
                    max_input_chars_per_word=100,
                ),
            )
            backend_tokenizer.add_special_tokens(
                [unk_token, sep_token, pad_token, cls_token, mask_token, bos_token, eos_token],
            )
            normalizer_sequence = [normalizers.Replace('\n', sep_token)]
            if do_basic_tokenize:
                normalizer_sequence.append(
                    normalizers.BertNormalizer(
                        handle_chinese_chars=tokenize_chinese_chars,
                        strip_accents=strip_accents,
                        lowercase=do_lower_case,
                    ),
                )
            backend_tokenizer.normalizer = normalizers.Sequence(normalizer_sequence)
            backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Digits(individual_digits=True),
            ])
        super().__init__(
            tokenizer_file=tokenizer_file,
            tokenizer_object=backend_tokenizer,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
        self._tokenizer.decoder = decoders.Decoder.custom(self.CustomDecoder())

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        files = self.backend_tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def save_pretrained(self, *args, **kwargs) -> tuple[str]:
        self._tokenizer.decoder = decoders.WordPiece()
        return super().save_pretrained(*args, **kwargs)
