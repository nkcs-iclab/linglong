import pickle
import pathlib
import warnings
import dataclasses
import numpy as np

import linglong


@dataclasses.dataclass
class EvaluationDatasetConfig:
    input_path: str | pathlib.Path
    output_path: str | pathlib.Path
    template_id: int
    special_tokens: dict[str, str]
    use_pinyin: bool = False
    model_path: str | None = None
    vocab_path: str | None = None
    pinyin_vocab_path: str | None = None
    split: str = 'test'
    use_cache: bool = False
    extra_config: dict | None = None

    def __post_init__(self):
        self.input_path = pathlib.Path(self.input_path)
        self.output_path = pathlib.Path(self.output_path)


class EvaluationDatasetBase:

    def __init__(self, config: EvaluationDatasetConfig):
        self.config = config
        self.input_file = next(self.config.input_path.glob(f'{self.config.split}*'))
        self.tokenizer = linglong.get_tokenizers(
            vocab_path=self.config.vocab_path,
            pinyin_vocab_path=self.config.pinyin_vocab_path,
            pretrained_model=self.config.model_path,
            special_tokens=self.config.special_tokens,
            use_pinyin=self.config.use_pinyin,
        )
        self.pinyin_tokenizer = None
        if self.config.use_pinyin:
            self.tokenizer, self.pinyin_tokenizer = self.tokenizer
        self.file_format = None
        self.candidates = None

        self.config.output_path.mkdir(parents=True, exist_ok=True)

    def _load_file(self, path: str) -> list | dict:
        return linglong.load_file(path, format=self.file_format)

    @staticmethod
    def _discard_obj(obj, discarded: list, reason: str | None = None):
        warnings.warn(f'The pinyin information of {linglong.prettify(obj)} is discarded. Reason: {reason}')
        warnings.warn(f'{len(discarded)} items are discarded.')
        discarded.append(obj)

    # noinspection PyMethodMayBeStatic
    def _post_process(self, data: list[dict]) -> list[dict]:
        return data

    def _templatize(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        return getattr(self, f'_template_{self.config.template_id}')(obj)

    def _prepend_start_token(self, parts: list) -> list:
        parts.insert(0, self.tokenizer.bos_token)
        return parts

    def _append_end_token(self, parts: list) -> list:
        parts.append(self.tokenizer.eos_token)
        return parts

    def _add_start_and_end_tokens(self, parts: list) -> list:
        self._prepend_start_token(parts)
        self._append_end_token(parts)
        return parts

    def _process(self) -> list[dict]:
        data, discarded = [], []
        objs = self._load_file(str(self.input_file))
        for i in linglong.trange(len(objs)):
            parts = self._templatize(objs[i])
            if len(parts) == 2:
                parts, label = parts
                extra = {}
            else:
                parts, label, extra = parts
            input_ids, pinyin_input_ids, label_ids = self._encode(parts, label)
            if self.config.use_pinyin and len(input_ids) != len(pinyin_input_ids):
                self._discard_obj(
                    objs[i],
                    discarded,
                    reason=f'`input_ids` and `pinyin_input_ids` have different lengths: '
                           f'{len(input_ids)} vs {len(pinyin_input_ids)}. '
                           f'(most likely due to omitted control characters.)',
                )
                pinyin_input_ids = np.zeros_like(input_ids)
            data.append({
                'input_ids': np.asarray(input_ids),
                **({'pinyin_input_ids': np.asarray(pinyin_input_ids)} if self.config.use_pinyin else {}),
                **({'label_ids': np.asarray(label_ids)} if label_ids is not None else {}),
                **extra,
            })
        return data

    def _encode(
            self,
            data_parts: list,
            label_parts: list | None,
    ) -> tuple[list[int], list[int] | None, list[int] | None]:
        label_ids = None
        input_ids = self._convert_parts_to_ids(data_parts)
        pinyin_input_ids = self._convert_parts_to_ids(data_parts, use_pinyin=True) if self.config.use_pinyin else None
        if label_parts is not None:
            label_ids = self._convert_parts_to_ids(label_parts)
        return input_ids, pinyin_input_ids, label_ids

    def _convert_parts_to_ids(
            self,
            parts: list,
            use_pinyin: bool = False,
    ) -> list[int]:
        tokenizer = self.pinyin_tokenizer if use_pinyin else self.tokenizer
        ids = []
        for part in parts:
            if isinstance(part, str):
                ids.extend(tokenizer.encode(part))
            elif isinstance(part, list):
                ids.extend(tokenizer.convert_tokens_to_ids(part))
            elif isinstance(part, dict):
                ids.extend(tokenizer.encode(part.get('pinyin' if use_pinyin else 'text', part['text'])))
        return ids

    def prepare(self) -> tuple[list[dict], list[str] | None]:
        save_path = self.config.output_path / f'{self.config.split}-template-{self.config.template_id}.pkl'
        if save_path.is_file() and self.config.use_cache:
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = self._process()
            data = self._post_process(data)
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
        return data, self.candidates
