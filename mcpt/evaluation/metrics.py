import re
import copy
import json
import mmap
import jieba
import rouge
import pickle
import pathlib
import collections
import numpy as np

from typing import *
from thefuzz import process
from simstring.searcher import Searcher
from simstring.database.dict import DictDatabase
from simstring.measure.jaccard import JaccardMeasure
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import mcpt


def generation_metric(
        y_true: List[Optional[np.ndarray]],
        y_pred: List[np.ndarray],
        **kwargs,
) -> Optional[Dict[str, Any]]:
    character_metric = GenerationMetric(kwargs['tokenizer'])
    y_true_str_list = []
    y_pred_str_list = []
    for label, pred in zip(y_true, y_pred):
        if label is None:
            return None
        if isinstance(label, np.ndarray):
            label = [label]
        label = [
            kwargs['tokenizer'].convert_ids_to_string(_)
            for _ in label
        ]
        pred = kwargs['tokenizer'].convert_ids_to_string(pred)
        character_metric.update_state(
            [list(_) for _ in label],
            list(pred)
        )
        y_true_str_list.append(label)
        y_pred_str_list.append(pred)
    word_metric = _rouge_scores(reference=y_true_str_list, hypothesis=y_pred_str_list)
    return {
        'character_based': character_metric.result(),
        'word_based': word_metric,
    }


def accuracy_metric(
        y_true: List[Optional[np.ndarray]],
        y_pred: List[np.ndarray],
        **_,
) -> Optional[Dict[str, Any]]:
    correct = 0
    for label, pred in zip(y_true, y_pred):
        if label is None:
            return None
        if np.array_equal(label, pred):
            correct += 1
    return _report_accuracy(correct, len(y_true))


def segmentation_metric(mode: str = 'basic', adjust: Optional[str] = None) -> Callable:
    if mode == 'basic':
        compare_fn = _compare_segments_basic
    elif mode == 'cuge':
        compare_fn = _compare_segments_cuge
    else:
        raise ValueError('Unsupported mode for `segmentation_metric`.')

    def segmentation_metric_(
            y_true: List[Optional[np.ndarray]],
            y_pred: List[np.ndarray],
            **kwargs,
    ) -> Optional[Dict[str, Any]]:
        len_label = 0
        len_pred = 0
        len_correct = 0
        if adjust == 'edit_distance':
            y_pred_ = []
            for x, y in zip(kwargs['x'], y_pred):
                edited_y = _min_edit(
                    # TODO: Move the hard-coded indices to the config.
                    x=x[0][0][6:-6] if kwargs['config']['model_config'].get('use_pinyin', False) else x[0][6:-6],
                    y=np.delete(y, np.where(y == kwargs['special_token_ids']['segment_separator'])),
                )
                y_pred_.append(_reconstruct_y(y, edited_y, kwargs['special_token_ids']['segment_separator']))
            y_pred = y_pred_
        for label, pred in zip(y_true, y_pred):
            if label is None:
                return None
            len_label_, len_pred_, len_correct_ = compare_fn(
                label, pred, kwargs['special_token_ids']['segment_separator']
            )
            len_label += len_label_
            len_pred += len_pred_
            len_correct += len_correct_
        p = len_correct / len_pred if len_pred > 0 else 0.0
        r = len_correct / len_label if len_label > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        return {
            'precision': p,
            'recall': r,
            'f1': f,
        }

    return segmentation_metric_


def dataset_math23k_metric(
        y_true: List[Optional[np.ndarray]],
        y_pred: List[np.ndarray],
        **kwargs,
) -> Optional[Dict[str, Any]]:
    correct = 0
    for label, pred in zip(y_true, y_pred):
        if label is None:
            return None
        try:
            label = list(label)
            label = label[label.index(kwargs['special_token_ids']['part_separator']) + 1:]
            if kwargs['config']['model_config'].get('backward', False):
                label, pred = label[::-1], pred[::-1]
            pred = kwargs['tokenizer'].convert_ids_to_string(pred)
            label = kwargs['tokenizer'].convert_ids_to_string(label)
            line = f'{pred}={label}'
            line = line.replace('[', '(').replace(']', ')')
            line = line.replace('^', '**')

            # Special cases.
            line = line.replace('千米/小时', '')

            if re.compile(r'\d\(').search(line):
                line = _remove_mixed_number(line)
            if '%' in line:
                line = _remove_percentage(line)
            pred, label = line.split('=')
            if np.isclose(float(eval(pred)), float(eval(label))):
                correct += 1
        except Exception as e:
            print('ERROR:', e)
    return _report_accuracy(correct, len(y_true))


def dataset_kbqa_metric(
        y_true: List[Optional[np.ndarray]],
        y_pred: List[np.ndarray],
        **kwargs,
) -> Optional[Dict[str, Any]]:
    if any(_ is None for _ in y_true):
        return None

    with mcpt.running('Loading the knowledge graph', timer=True):
        if pathlib.Path(kwargs['config']['extra_config']['kg_cache']).is_file():
            with open(kwargs['config']['extra_config']['kg_cache'], 'rb') as f:
                kg = pickle.load(f)
        else:
            kg = {}
            with open(kwargs['config']['extra_config']['kg_path'], 'rb') as f:
                map_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                for line in iter(map_file.readline, b''):
                    obj = json.loads(line.decode('utf-8'))
                    if obj['subject'].lower() not in kg:
                        kg[obj['subject'].lower()] = {}
                    kg[obj['subject'].lower()][obj['relation'].lower()] = obj['object']
            with open(kwargs['config']['extra_config']['kg_cache'], 'wb') as f:
                pickle.dump(kg, f)

    with mcpt.running('Loading the search database', timer=True):
        if pathlib.Path(kwargs['config']['extra_config']['kg_search_cache']).is_file():
            with open(kwargs['config']['extra_config']['kg_search_cache'], 'rb') as f:
                searcher = pickle.load(f)
        else:
            db = DictDatabase(CharacterNgramFeatureExtractor(2))
            for a in kg.keys():
                db.add(a)
            searcher = Searcher(db, JaccardMeasure())
            with open(kwargs['config']['extra_config']['kg_search_cache'], 'wb') as f:
                pickle.dump(searcher, f)

    correct = 0
    for idx, (label, pred) in enumerate(mcpt.tqdm(list(zip(y_true, y_pred)))):
        label = label[np.where(label == kwargs['special_token_ids']['segment_separator'])[0][-1] + 1:]
        try:
            separator_idx = np.where(pred == kwargs['special_token_ids']['segment_separator'])[0][0]
        except Exception as e:
            print('ERROR:', e)
            continue
        a = pred[:separator_idx]
        relation = pred[separator_idx + 1:]
        if kwargs['config']['model_config'].get('backward', False):
            label, a, relation = label[::-1], a[::-1], relation[::-1]
        label = kwargs['tokenizer'].convert_ids_to_string(label)
        a = kwargs['tokenizer'].convert_ids_to_string(a)
        relation = kwargs['tokenizer'].convert_ids_to_string(relation)

        log_item = {
            'id': idx,
            'original_output': {
                'a': a,
                'relation': relation,
            },
        }

        if a not in kg:
            revised_a = process.extractOne(a, searcher.search(a, 0.15))
            a = revised_a[0] if revised_a else process.extractOne(a, kg.keys())[0]
        if relation not in kg[a]:
            relation = process.extractOne(relation, kg[a].keys())[0]
        b = kg[a][relation]

        if kwargs['config']['verbose'] >= 2:
            with open(kwargs['output_path'] + '-metric.jsonl', 'a') as f:
                log_item['revised_output'] = {
                    'a': a,
                    'relation': relation,
                }
                log_item['query_result'] = b
                log_item['target'] = label
                f.write(json.dumps(log_item, ensure_ascii=False) + '\n')

        # MCPT tokenizer cannot generate whitespaces in `label`.
        if label.replace(' ', '').lower() == b.replace(' ', '').lower():
            correct += 1

    return _report_accuracy(correct, len(y_true))


def _report_accuracy(correct: int, total: int):
    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total,
    }


# Functions used by `segmentation_metric`.

def _compare_segments_basic(label: np.ndarray, pred: np.ndarray, separator_id: int) -> Tuple[int, int, int]:
    label_sep_index = np.where(label == separator_id)[0]
    pred_sep_index = np.where(pred == separator_id)[0]
    label_words_count = len(label_sep_index) + 1
    pred_words_count = len(pred_sep_index) + 1
    label_sep_index = np.concatenate(([-1], label_sep_index, [len(label)]))
    pred_sep_index = np.concatenate(([-1], pred_sep_index, [len(pred)]))
    label_split = [label[pos + 1:label_sep_index[index + 1]].tolist() for index, pos in enumerate(label_sep_index[:-1])]
    pred_split = [pred[pos + 1:pred_sep_index[index + 1]].tolist() for index, pos in enumerate(pred_sep_index[:-1])]

    acc_words_count = 0
    for word in label_split:
        if word in pred_split:
            acc_words_count += 1
    return label_words_count, pred_words_count, acc_words_count


def _compare_segments_cuge(label: np.ndarray, pred: np.ndarray, separator_id: int) -> Tuple[int, int, int]:
    label_sep_index = np.where(label == separator_id)[0]
    pred_sep_index = np.where(pred == separator_id)[0]
    label_words_count = len(label_sep_index) + 1
    pred_words_count = len(pred_sep_index) + 1
    label_sep_index = np.concatenate(([-1], label_sep_index, [len(label)]))
    pred_sep_index = np.concatenate(([-1], pred_sep_index, [len(pred)]))

    label_index = []
    pos = 0
    for index, pos_ in enumerate(label_sep_index[:-1]):
        word_pos = [pos]
        pos += label_sep_index[index + 1] - pos_ - 1
        word_pos.append(pos)
        label_index.append(word_pos)
    pred_index = []
    pos = 0
    for index, pos_ in enumerate(pred_sep_index[:-1]):
        word_pos = [pos]
        pos += pred_sep_index[index + 1] - pos_ - 1
        word_pos.append(pos)
        pred_index.append(word_pos)

    acc_words_count = 0
    for word_pos in label_index:
        if word_pos in pred_index:
            acc_words_count += 1
    return label_words_count, pred_words_count, acc_words_count


def _min_edit(x: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
    d = np.zeros((len(y) + 1, len(x) + 1))
    p = np.zeros_like(d)
    d[0] = np.arange(len(x) + 1)
    d[:, 0] = np.arange(len(y) + 1)
    p[0, 1:] = 3
    p[1:, 0] = 2
    for i in range(1, len(y) + 1):
        for j in range(1, len(x) + 1):
            if y[i - 1] == x[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                choices = [d[i - 1][j - 1] + 1, d[i - 1][j] + 1, d[i][j - 1] + 1]
                min_idx = np.argmin(choices)
                d[i][j] = choices[min_idx]
                p[i][j] = min_idx + 1
    i = len(y)
    j = len(x)
    output = []
    while i >= 0 and j >= 0:
        if p[i][j] == 1:
            output.append({'op': 'replace', 'src': y[i - 1], 'dest': x[j - 1]})
            i -= 1
            j -= 1
        elif p[i][j] == 2:
            output.append({'op': 'delete', 'dest': y[i - 1]})
            i -= 1
        elif p[i][j] == 3:
            output.append({'op': 'insert', 'dest': x[j - 1]})
            j -= 1
        else:
            if i > 0:
                output.append({'dest': y[i - 1]})
            i -= 1
            j -= 1
    return output[::-1]


def _reconstruct_y(y: np.ndarray, edited_y: List[Dict[str, Any]], separator_id: int) -> np.ndarray:
    y_idx = 0
    reconstructed_y = []
    for obj in edited_y:
        while y_idx < len(y):
            if y[y_idx] == separator_id:
                reconstructed_y.append(separator_id)
                y_idx += 1
            else:
                break
        op = obj.get('op', 'noop')
        dest = obj.get('dest')
        if op == 'delete':
            y_idx += 1
        elif op == 'insert':
            reconstructed_y.append(dest)
            reconstructed_y.append(separator_id)
        else:
            reconstructed_y.append(dest)
            y_idx += 1
    output = []
    last_token_id = None
    for token_id in reconstructed_y:
        if token_id != separator_id:
            output.append(token_id)
        else:
            if last_token_id != separator_id:
                output.append(token_id)
        last_token_id = token_id
    if len(output) > 0 and output[-1] == separator_id:
        output = output[:-1]
    return np.asarray(output)


# Functions used by `dataset_math23k_metric`.

def _remove_percentage(line: str) -> str:
    pattern = re.compile(r'\d+\.*\d*%')
    while re_result := pattern.search(line):
        span = re_result.span()
        line = f'{line[:span[0]]}({line[span[0]:span[1] - 1]}/100){line[span[1]:]}'
    return line


def _remove_mixed_number(line: str) -> str:
    pattern = re.compile(r'\d\(')
    while re_result := pattern.search(line):
        span = re_result.span()
        end = span[1] - 1
        line = f'{line[:end]}+{line[end:]}'
    return line


# Functions used by `generation_metric`.

def _rouge_scores(reference: List[List[str]], hypothesis: List[str]) -> Dict[str, Any]:
    rouge_name = ['rouge-1', 'rouge-2', 'rouge-l']
    item_name = ['r', 'p', 'f']
    result = {}

    def split_words(line: str) -> List[str]:
        return list(jieba.cut(''.join(line.strip().split())))

    for rouge_name_i in rouge_name:
        result[rouge_name_i] = {}
        for item_name_i in item_name:
            result[rouge_name_i][item_name_i] = []
    for refs, hyp in zip(reference, hypothesis):
        result_i = []
        hyp = ' '.join(split_words(hyp))
        for ref in refs:
            ref = ' '.join(split_words(ref))
            result_i.append(rouge.Rouge().get_scores(refs=ref, hyps=hyp)[0])
        for rouge_name_i in rouge_name:
            for item_name_i in item_name:
                result[rouge_name_i][item_name_i].append(max([_[rouge_name_i][item_name_i] for _ in result_i]))
    for rouge_name_i in rouge_name:
        for item_name_i in item_name:
            result[rouge_name_i][item_name_i] = np.mean(result[rouge_name_i][item_name_i])
    return result


def _lcs(a: str, b: str) -> int:
    if len(a) < len(b):
        b, a = a, b
    lengths = [[0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
    for j in range(1, len(b) + 1):
        for i in range(1, len(a) + 1):
            if a[i - 1] == b[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
    return lengths[len(a)][len(b)]


class Ngrams:

    def __init__(self, ngrams: Optional[Any] = None, exclusive: bool = True):
        if ngrams is None:
            ngrams = {}
        if exclusive:
            self._ngrams = set(ngrams)
        else:
            self._ngrams = list(ngrams)
        self.exclusive = exclusive

    def add(self, e):
        if self.exclusive:
            # noinspection PyUnresolvedReferences
            self._ngrams.add(e)
        else:
            self._ngrams.append(e)

    def __len__(self) -> int:
        return len(self._ngrams)

    def intersection(self, other: 'Ngrams') -> 'Ngrams':
        if self.exclusive:
            # noinspection PyUnresolvedReferences
            inter_set = self._ngrams.intersection(other._ngrams)
            return Ngrams(inter_set, exclusive=True)
        other_list = copy.deepcopy(other._ngrams)
        inter_list = []
        for e in self._ngrams:
            try:
                i = other_list.index(e)
            except ValueError:
                continue
            other_list.pop(i)
            inter_list.append(e)
        return Ngrams(inter_list, exclusive=False)

    def union(self, *ngrams: 'Ngrams') -> 'Ngrams':
        if self.exclusive:
            union_set = self._ngrams
            for other in ngrams:
                union_set = union_set.union(other._ngrams)
            return Ngrams(union_set, exclusive=True)
        union_list = copy.deepcopy(self._ngrams)
        for other in ngrams:
            union_list.extend(other._ngrams)
        return Ngrams(union_list, exclusive=False)


class GenerationMetric:

    def __init__(self, tokenizer: mcpt.tokenization.Tokenizer):
        self._reference = []
        self._hypothesis = []
        self._tokenizer = tokenizer

    def update_state(self, reference: List[List[str]], hypothesis: List[str]):
        self._reference.append(reference)
        self._hypothesis.append(hypothesis)

    def _calc_bleu_k(self, k) -> float:
        weights = [1. / k] * k + (4 - k) * [0.]
        try:
            # noinspection PyTypeChecker
            bleu = corpus_bleu(
                self._reference,
                self._hypothesis,
                weights=weights,
                smoothing_function=SmoothingFunction().method3,
            )
        except ZeroDivisionError:
            print('The bleu is invalid')
            bleu = 0.
        return bleu

    def _calc_distinct_k(self, k) -> float:
        d = {}
        tot = 0
        for hypothesis in self._hypothesis:
            for i in range(0, len(hypothesis) - k):
                key = tuple(hypothesis[i:i + k])
                d[key] = 1
                tot += 1
        if tot > 0:
            dist = len(d) / tot
        else:
            print('The distinct is invalid.')
            dist = 0.
        return dist

    def _calc_unigram_f1(self) -> Tuple[float, List[float]]:
        f1_scores = []
        for hyp, refs in zip(self._hypothesis, self._reference):
            scores = []
            for ref in refs:
                cross = collections.Counter(hyp) & collections.Counter(ref)
                cross = sum(cross.values())
                # noinspection PyTypeChecker
                p = cross / max(len(hyp), 1e-10)
                r = cross / len(ref)
                f1 = 2 * p * r / max(p + r, 1e-10)
                scores.append(f1)
            f1_scores.append(max(scores))
        return float(np.mean(f1_scores)), f1_scores

    def _calc_rouge_l(self, beta=1.2) -> Tuple[float, List[float]]:
        scores = []
        for hypothesis, reference in zip(self._hypothesis, self._reference):
            p_lcs = []
            r_lcs = []
            for reference_i in reference:
                lcs_length = _lcs(reference_i, hypothesis)
                # noinspection PyTypeChecker
                p_lcs.append(lcs_length / max(len(hypothesis), 1e-10))
                r_lcs.append(lcs_length / len(reference_i))
            p_lcs_max = max(p_lcs)
            r_lcs_max = max(r_lcs)
            if p_lcs_max != 0 and r_lcs_max != 0:
                f_lcs = ((1 + beta ** 2) * p_lcs_max * r_lcs_max) / float(r_lcs_max + beta ** 2 * p_lcs_max)
            else:
                f_lcs = 0.0
            scores.append(f_lcs)
        return float(np.mean(scores)), scores

    @staticmethod
    def _get_ngrams(n: int, text: List[str], exclusive: bool = True) -> Ngrams:
        ngram_set = Ngrams(exclusive=exclusive)
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _get_word_ngrams(self, n: int, sentences: List[List[str]], exclusive: bool = True) -> Ngrams:
        words = [x for y in sentences for x in y]
        return self._get_ngrams(n, words, exclusive=exclusive)

    @staticmethod
    def _f_r_p_rouge_n(reference_count: int, overlapping_count: int) -> float:
        # Handle edge case. This isn't mathematically correct, but it's good enough.
        if reference_count == 0:
            return 0.
        return overlapping_count / reference_count

    def _calc_rouge_n(self, n: int = 2, exclusive: bool = True):
        if len(self._hypothesis) <= 0:
            raise ValueError('The hypothesis list is empty.')
        if len(self._reference) <= 0:
            raise ValueError('The reference list is empty.')

        evaluated_ngrams = self._get_word_ngrams(n, self._hypothesis, exclusive=exclusive)
        reference = [x[0] for x in self._reference]
        reference_ngrams = self._get_word_ngrams(n, reference, exclusive=exclusive)
        reference_count = len(reference_ngrams)

        # Gets the overlapping ngrams between evaluated and reference.
        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        return self._f_r_p_rouge_n(reference_count, overlapping_count)

    def result(self) -> Dict[str, float]:
        result = {
            **{f'dist-{k}': self._calc_distinct_k(k) for k in range(3, 5)},
            **{f'bleu-{k}': self._calc_bleu_k(k) for k in range(4, 5)},
        }

        f1, _ = self._calc_unigram_f1()
        result['unigram-f1'] = f1

        rl, _ = self._calc_rouge_l()
        result['rouge-l'] = rl

        result['rouge-1'] = self._calc_rouge_n(n=1)
        result['rouge-2'] = self._calc_rouge_n(n=2)

        return result
