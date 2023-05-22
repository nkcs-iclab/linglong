import jieba
import string
import pathlib
import pypinyin
import collections

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.bert.tokenization_bert import (
    BasicTokenizer as BertBasicTokenizer,
    WordpieceTokenizer,
    load_vocab,
)
from typing import *


def _load_pinyin_vocab(vocab_file: str) -> Dict[str, int]:
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
            # To fix a linting bug in PyCharm.
            # noinspection PyUnresolvedReferences
            vocab[k] = vocab[v]
    return vocab


class Tokenizer(PreTrainedTokenizer):

    def __init__(
            self,
            vocab_file: str,
            do_lower_case: bool = True,
            do_basic_tokenize: bool = True,
            never_split: Optional[List[str]] = None,
            unk_token: str = '[UNK]',
            sep_token: str = '[SEP]',
            pad_token: str = '[PAD]',
            cls_token: str = '[CLS]',
            mask_token: str = '[MASK]',
            tokenize_chinese_chars: bool = True,
            strip_accents: Optional[bool] = None,
            **kwargs,
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            strip_accents=strip_accents,
            **kwargs,
        )
        if not pathlib.Path(vocab_file).is_file():
            raise FileNotFoundError(f"Can't find a vocabulary file at path '{vocab_file}'.")
        else:
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
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        raise NotImplementedError

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        text = text.split('\n')
        split_tokens = []
        if self.do_basic_tokenize:
            for text_piece in text:
                for token in self.basic_tokenizer.tokenize(text_piece, never_split=self.all_special_tokens):
                    for sub_token in self.wordpiece_tokenizer.tokenize(token):
                        split_tokens.append(sub_token)
                split_tokens.append('[SEP]')
            split_tokens.pop()
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)

        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_string_to_ids(self, text: str, **kwargs) -> List[int]:
        return self.convert_tokens_to_ids(self.tokenize(text, **kwargs))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: Union[str, List[str]]) -> str:
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

    def convert_ids_to_string(self, ids: Union[int, List[int]], **kwargs) -> str:
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(ids, **kwargs))


class PinyinTokenizer(PreTrainedTokenizer):

    def __init__(
            self,
            vocab_file: str,
            fallback,
            unk_token: str = '[UNK]',
            sep_token: str = '[SEP]',
            pad_token: str = '[PAD]',
            cls_token: str = '[CLS]',
            mask_token: str = '[MASK]',
            **kwargs,
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        if not pathlib.Path(vocab_file).is_file():
            raise FileNotFoundError(f"Can't find a vocabulary file at path '{vocab_file}'.")
        self.vocab = _load_pinyin_vocab(vocab_file)
        self._fallback = fallback
        self.convert_id_to_token = None
        self.convert_tokens_to_string = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        raise NotImplementedError

    def tokenize(self, text: Union[str, List[str]], **kwargs) -> List[str]:
        return self._tokenize(text, **kwargs)

    def _tokenize(self, text: Union[str, List[str]], **kwargs) -> List[str]:
        return [token[0] for token in pypinyin.pinyin(text, errors=lambda x: self._fallback.tokenize(x))]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_string_to_ids(self, text: Union[str, List[str]], **kwargs) -> List[int]:
        return self.convert_tokens_to_ids(self.tokenize(text, **kwargs))

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    @staticmethod
    def convert_tokenizer_tokens_to_tokens(tokens: Union[str, List[str]]) -> Union[str, List[str]]:
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

    def convert_tokenizer_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        return self.convert_tokens_to_ids(self.convert_tokenizer_tokens_to_tokens(tokens))


class BasicTokenizer(BertBasicTokenizer):

    def __init__(
            self,
            do_lower_case: bool = True,
            never_split: Optional[List[str]] = None,
            tokenize_chinese_chars: bool = True,
            strip_accents: Optional[bool] = None,
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


class CPM2WordpieceTokenizer:

    def __init__(
            self,
            vocab: Dict[str, int],
            unk_token: str = '[UNK]',
            max_input_chars_per_word: int = 200,
    ):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token: str) -> List[str]:
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return [self.unk_token]
        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = ''.join(chars[start:end])
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                sub_tokens.append(self.unk_token)
                start += 1
                continue
            sub_tokens.append(cur_substr)
            start = end
        return sub_tokens


class CPM2Tokenizer:

    def __init__(
            self,
            vocab_file: str,
            unk_token: str = '[UNK]',
            max_len: Optional[str] = None,
            max_sentinels: int = 190,
    ):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = load_vocab(vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.wordpiece_tokenizer = CPM2WordpieceTokenizer(vocab=self.encoder)
        self.translator = str.maketrans(" \n", "\u2582\u2583")
        self.sentinel_list = [self.encoder['<s_{}>'.format(i)] for i in range(max_sentinels)]
        self.unk_token = unk_token

    def __len__(self) -> int:
        return len(self.encoder)

    @staticmethod
    def load_vocab(vocab_file: str) -> Dict[str, int]:
        vocab = collections.OrderedDict()
        index = 0
        with open(vocab_file, 'r') as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    def tokenize(self, text: str) -> List[str]:
        output_tokens = []
        for x in jieba.cut(text, cut_all=False):
            x = x.translate(self.translator)
            output_tokens.extend(self.wordpiece_tokenizer.tokenize(x))
        return output_tokens

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self.encoder.get(tokens, self.encoder[self.unk_token])
        return [self.encoder.get(x, self.encoder[self.unk_token]) for x in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.decoder[ids]
        return [self.decoder[int(x)] for x in ids]

    @staticmethod
    def convert_tokens_to_string(tokens: Union[str, List[str]]) -> str:
        if isinstance(tokens, str):
            return tokens
        return ''.join(tokens)

    def convert_ids_to_string(self, ids: Union[int, List[int]]) -> str:
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))

    def convert_string_to_ids(self, text: str) -> List[int]:
        return self.convert_tokens_to_ids(self.tokenize(text))
