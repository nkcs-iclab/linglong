import dataclasses

from transformers import (
    Trainer,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.utils import check_min_version

import linglong
import linglong.records


@dataclasses.dataclass
class ModelArguments:
    pretrained_model: str | None = dataclasses.field(
        default=None,
        metadata={'help': 'Pretrained model path'},
    )

    model_config: str | None = dataclasses.field(
        default=None,
        metadata={'help': 'Model config path'},
    )


@dataclasses.dataclass
class DataArguments:
    training_data: str = dataclasses.field(
        metadata={'help': 'Training data path'},
    )

    training_meta: str = dataclasses.field(
        default='train-meta.json',
        metadata={'help': 'The meta file\'s filename of the training data'},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    is_main_process = training_args.process_index == 0

    with linglong.running('Loading configs', is_main_process=is_main_process) as spinner:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant': False}
        spinner.write(model_args)
        spinner.write(data_args)
        spinner.write(training_args)

    with linglong.running('Loading the model', is_main_process=is_main_process, timer=True) as spinner:
        if model_args.pretrained_model is not None:
            model = linglong.LingLongLMHeadModel.from_pretrained(model_args.pretrained_model)
        elif model_args.model_config is not None:
            model = linglong.LingLongLMHeadModel(linglong.LingLongConfig.from_json_file(model_args.model_config))
        else:
            raise ValueError('Either pretrained_model or model_config must be provided.')
        model_config = model.config
        linglong.print_trainable_parameters(model, is_main_process=is_main_process, print_fn=spinner.write)
        spinner.write(model)

    with linglong.running('Loading the dataset', is_main_process=is_main_process, timer=True):
        train_dataset = linglong.records.load(
            path=data_args.training_data,
            meta=data_args.training_meta,
            use_pinyin=model_config.use_pinyin,
        )

    with linglong.running('Training', is_main_process=is_main_process, spinner=False):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == '__main__':
    check_min_version('4.39.3')
    linglong.init()
    main()
