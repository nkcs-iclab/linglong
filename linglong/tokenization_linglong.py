import shutil
import string
import pathlib
import pypinyin
import collections

from transformers.utils import logging
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.bert.tokenization_bert import (
    BasicTokenizer as BertBasicTokenizer,
    WordpieceTokenizer,
    load_vocab,
)

logger = logging.get_logger(__name__)


def _load_pinyin_vocab(vocab_file: str) -> dict[str, int]:
    vocab = collections.OrderedDict()
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        for token_i in token.strip().split():
            vocab[token_i] = index
    number_pinyin = {
        '0': 'líng',
        '1': 'yī',
        '2': 'èr',
        '3': 'sān',
        '4': 'sì',
        '5': 'wǔ',
        '6': 'liù',
        '7': 'qī',
        '8': 'bā',
        '9': 'jiǔ',
    }
    for k, v in number_pinyin.items():
        if k in vocab and v in vocab:
            # noinspection PyUnresolvedReferences
            vocab[k] = vocab[v]
    return vocab


class LingLongTokenizer(PreTrainedTokenizer):
    vocab_files_names = {'vocab_file': 'tokenizer.txt'}
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(
            self,
            vocab_file: str,
            do_lower_case: bool = True,
            do_basic_tokenize: bool = True,
            never_split: list[str] | None = None,
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
        self.vocab_file = vocab_file
        if not pathlib.Path(vocab_file).is_file():
            raise FileNotFoundError(f"Can't find a vocabulary file at path '{vocab_file}'.")
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        super().__init__(
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
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
        self.add_special_tokens({'additional_special_tokens': [f'<unused{i}>' for i in range(1, 11)]})

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def save_vocabulary(
            self,
            save_directory: str,
            filename_prefix: str | None = None,
    ) -> tuple[str | None, str | None]:
        if not pathlib.Path(save_directory).is_dir():
            logger.error(f'Vocabulary path ({save_directory}) should be a directory.')
            return None, None
        out_vocab_file = pathlib.Path(
            save_directory,
            (filename_prefix + '-' if filename_prefix else ''),
            self.vocab_files_names['vocab_file'],
        )
        if pathlib.Path(self.vocab_file).absolute() != out_vocab_file.absolute():
            shutil.copyfile(self.vocab_file, out_vocab_file)
        return str(out_vocab_file), None

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        text = text.split('\n')
        split_tokens = []
        if self.do_basic_tokenize:
            for text_piece in text:
                for token in self.basic_tokenizer.tokenize(text_piece, never_split=self.all_special_tokens):
                    for sub_token in self.wordpiece_tokenizer.tokenize(token):
                        split_tokens.append(sub_token)
                split_tokens.append(self.sep_token)
            split_tokens.pop()
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)

        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: str | list[str]) -> str:
        if isinstance(tokens, str):
            tokens = [tokens]
        whitespace_joined = ' '.join(tokens).replace(' ##', '').strip()

        # Remove whitespaces between Chinese characters.
        # TODO: This will remove whitespaces between some English words as well. Need fix.
        alphabet_set = set(list(string.ascii_letters))
        result = ''
        for i in range(len(whitespace_joined)):
            if whitespace_joined[i] == ' ' and whitespace_joined[i + 1] not in alphabet_set:
                continue
            result += whitespace_joined[i]
        return result


class PinyinTokenizer(PreTrainedTokenizer):
    vocab_files_names = {'vocab_file': 'pinyin_tokenizer.txt'}
    model_input_names = ['input_ids']

    def __init__(
            self,
            vocab_file: str,
            fallback,
            unk_token: str = '<unk>',
            sep_token: str = '<sep>',
            pad_token: str = '<pad>',
            cls_token: str = '<cls>',
            mask_token: str = '<mask>',
            bos_token: str = '<|startoftext|>',
            eos_token: str = '<|endoftext|>',
            **kwargs,
    ):
        self.vocab_file = vocab_file
        if not pathlib.Path(vocab_file).is_file():
            raise FileNotFoundError(f"Can't find a vocabulary file at path '{vocab_file}'.")
        self.vocab = _load_pinyin_vocab(vocab_file)
        self.idx_count = {}
        for tok, ids in self.vocab.items():
            self.idx_count.setdefault(ids, 0)
            self.idx_count[ids] += 1
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.fallback_tokenizer = fallback
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def save_vocabulary(
            self,
            save_directory: str,
            filename_prefix: str | None = None,
    ) -> tuple[str | None, str | None]:
        if not pathlib.Path(save_directory).is_dir():
            logger.error(f'Vocabulary path ({save_directory}) should be a directory.')
            return None, None
        out_vocab_file = pathlib.Path(
            save_directory,
            (filename_prefix + '-' if filename_prefix else ''),
            self.vocab_files_names['pinyin_vocab_file'],
        )
        if pathlib.Path(self.vocab_file).absolute() != out_vocab_file.absolute():
            shutil.copyfile(self.vocab_file, out_vocab_file)
        return str(out_vocab_file), None

    def _tokenize(self, text: str | list[str], **kwargs) -> list[str]:
        return [token[0] for token in pypinyin.pinyin(text, errors=lambda x: self.fallback_tokenizer.tokenize(x))]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_ids_to_tokens(self, ids: int | list[int], skip_special_tokens: bool = False) -> str | list[str]:
        tokens = super().convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
        if isinstance(tokens, str):
            tokens = [tokens]
        tokens = [
            self.unk_token if (self.idx_count.get(self._convert_token_to_id(token), 0) > 1) else token for token in
            tokens
        ]
        if len(tokens) == 1:
            return tokens[0]
        return tokens

    @staticmethod
    def convert_tokenizer_tokens_to_tokens(tokens: str | list[str]) -> str | list[str]:
        if isinstance(tokens, str):
            tokens = [tokens]
        pinyin_tokens = [
            pinyin_token[0]
            for token in tokens
            for pinyin_token in pypinyin.pinyin(token)
        ]
        if len(pinyin_tokens) == 1:
            return pinyin_tokens[0]
        return pinyin_tokens

    def convert_tokenizer_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        return self.convert_tokens_to_ids(self.convert_tokenizer_tokens_to_tokens(tokens))


class BasicTokenizer(BertBasicTokenizer):

    def __init__(
            self,
            do_lower_case: bool = True,
            never_split: list[str] | None = None,
            tokenize_chinese_chars: bool = True,
            strip_accents: bool | None = None,
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            never_split=never_split,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
        )

    def _tokenize_chinese_chars(self, text: str) -> str:
        output = []
        for char in text:
            cp = ord(char)
            # `char.isdigit()` is for cases like '123456'.
            # Instead of `['123456']`, we want it to be tokenized as `['1', '2', '3', '4', '5', '6']`.
            if self._is_chinese_char(cp) or char.isdigit():
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)
