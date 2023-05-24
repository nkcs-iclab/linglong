import jieba
import collections

from typing import *
from transformers.models.bert.tokenization_bert import load_vocab


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
    ):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = load_vocab(vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.wordpiece_tokenizer = CPM2WordpieceTokenizer(vocab=self.encoder)
        self.translator = str.maketrans(" \n", "\u2582\u2583")
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
