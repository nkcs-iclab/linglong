import fire
import json
import time
import pathlib

from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModelForCausalLM
from torch.utils.data import DataLoader

import linglong
import linglong.evaluation

accelerator = Accelerator()


@accelerator.on_main_process
def log_generation(
        id_: list[int],
        input_ids: list[list[int]],
        label_ids: list[list[int]] | None,
        generated_ids: list[list[int]],
        tokenizer: linglong.LingLongTokenizer | linglong.LingLongTokenizerFast,
        output_path: pathlib.Path,
):
    output_file = output_path.with_suffix('.jsonl')
    with output_file.open('a', encoding='utf-8') as fp:
        for id_, input_id, generated_id, label_id in zip(id_, input_ids, generated_ids, label_ids or [None] * len(id_)):
            log_entry = {
                'id': id_,
                'prompt': tokenizer.decode(input_id),
            }
            if label_id is not None:
                log_entry['label'] = tokenizer.decode(label_id)
            log_entry['predict'] = tokenizer.decode(generated_id)
            fp.write(json.dumps(log_entry, ensure_ascii=False) + '\n')


def main(
        dataset: str,
        input_path: str,
        cache_path: str,
        output_path: str,
        per_device_batch_size: int,
        dataset_config: str,
        vocab_path: str | None = None,
        pinyin_vocab_path: str | None = None,
        use_cache: bool = False,
        output_path_template: str = '{dataset}-{model}-{split}-{template_id}-{timestamp}',
        special_tokens: dict[str, str] | None = None,
):
    is_main_process = accelerator.is_main_process
    device = accelerator.device

    with linglong.running('Loading configs', is_main_process=is_main_process) as spinner:
        special_tokens = {
            'part_separator': '<unused1>',
            'segment_separator': '<unused2>',
            **(special_tokens or {}),
        }
        config = linglong.merge_configs({
            'dataset': dataset,
            'input_path': input_path,
            'output_path': cache_path,
            'vocab_path': vocab_path,
            'pinyin_vocab_path': pinyin_vocab_path,
            'use_cache': use_cache,
            'special_tokens': special_tokens,
        }, linglong.load_config(dataset_config, key=dataset))
        spinner.write(linglong.prettify(config))
        if isinstance(config['model'], str):
            model_path = config['model']
            peft_model = None
        else:
            model_path = config['model']['base']
            peft_model = config['model']['peft']
        output_path = pathlib.Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        output_path = output_path / output_path_template.format(
            dataset=dataset,
            model=pathlib.Path(peft_model or model_path).stem,
            split=config['split'],
            template_id=config['template_id'],
            timestamp=time.time(),
        )

    with linglong.running('Loading the model', is_main_process=is_main_process, timer=True):
        model = linglong.LingLongForCausalLM.from_pretrained(model_path, device_map=device)
        if peft_model is not None:
            model = PeftModelForCausalLM.from_pretrained(model, peft_model, device_map=device)
        config['use_pinyin'] = model.config.use_pinyin
        tokenizer = linglong.get_tokenizers(
            vocab_path=vocab_path,
            pretrained_model=model_path,
            special_tokens=special_tokens,
        )

    with linglong.running(f'Loading {dataset} dataset', is_main_process=is_main_process):
        if not is_main_process:
            config['use_cache'] = True
        with accelerator.main_process_first():
            dataset = linglong.datasets.evaluation.load(config)
            data, candidates = dataset.prepare()
        if candidates is not None:
            candidate_ids = tokenizer.encode(candidates)
            bad_words_ids = [[id_] for id_ in range(tokenizer.vocab_size) if id_ not in candidate_ids]
        else:
            bad_words_ids = None
        dataset = linglong.data.DictDataset(data)
        data_loader = DataLoader(
            dataset,
            batch_size=per_device_batch_size * accelerator.num_processes,
            collate_fn=linglong.data.padded_batch,
        )

    with linglong.running('Evaluating', is_main_process=is_main_process, spinner=False):
        accelerator.wait_for_everyone()
        outputs = []
        for global_batch in linglong.tqdm(data_loader, is_main_process=is_main_process):
            with accelerator.split_between_processes(global_batch) as batch:
                id_ = batch.pop('id')
                label_ids = batch.pop('label_ids') if 'label_ids' in batch else None
                batch = {k: v.to(device) for k, v in batch.items()}
                generated_ids = model.generate(
                    **batch,
                    **(config.get('generation_config') or {}),
                    bad_words_ids=bad_words_ids,
                )
                id_ = [x.cpu().item() for x in gather_object(id_)]
                input_ids = [x.cpu().tolist() for x in gather_object(batch['input_ids'])]
                if label_ids is not None:
                    label_ids = [x.cpu().tolist() for x in gather_object(label_ids)]
                generated_ids = [x.cpu().tolist() for x in gather_object(generated_ids)]
                for i in range(len(id_)):
                    generated_ids[i] = generated_ids[i][len(input_ids[i]):]
                    input_ids[i] = [x for x in input_ids[i] if x != tokenizer.pad_token_id]
                    if label_ids is not None:
                        label_ids[i] = [x for x in label_ids[i] if x != -100]
                    if tokenizer.eos_token_id in generated_ids[i]:
                        eos_idx = generated_ids[i].index(tokenizer.eos_token_id)
                    else:
                        eos_idx = len(generated_ids[i])
                    generated_ids[i] = generated_ids[i][:eos_idx]
                log_generation(id_, input_ids, label_ids, generated_ids, tokenizer, output_path)
            outputs.extend([
                {
                    'id': id_,
                    'input_ids': input_ids,
                    **({'label_ids': label_ids} if label_ids is not None else {}),
                    'generated_ids': generated_ids,
                } for id_, input_ids, label_ids, generated_ids in
                zip(id_, input_ids, label_ids or [None] * len(id_), generated_ids)
            ])

    if is_main_process:
        unique_ids = {}
        for idx, item in enumerate(outputs):
            if item['id'] not in unique_ids:
                unique_ids[item['id']] = idx
        unique_outputs = [outputs[idx] for idx in unique_ids.values()]
        eval_metric = linglong.evaluation.get_metric(config.get('evaluation_metric'))(
            tokenizer=tokenizer,
        )
        result = eval_metric(unique_outputs)
        print(linglong.prettify(result))


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
