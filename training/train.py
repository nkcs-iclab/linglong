import deepspeed

from dataclasses import dataclass, field
from transformers import (
    Trainer,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.utils import check_min_version

import linglong
import linglong.records

check_min_version('4.39.3')


@dataclass
class ModelArguments:
    pretrained_model: str | None = field(
        default=None,
        metadata={'help': 'Pretrained model path'},
    )

    model_config: str | None = field(
        default=None,
        metadata={'help': 'Model config path'},
    )


@dataclass
class DataArguments:
    training_data: str = field(
        metadata={'help': 'Training data path'},
    )

    training_meta: str = field(
        default='train-meta.json',
        metadata={'help': 'The meta file\'s filename of the training data'},
    )

    load_attention_mask: bool = field(
        default=True,
        metadata={'help': 'Whether to load the attention mask'},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    comm = linglong.Comm(
        size=deepspeed.comm.get_world_size(),
        rank=deepspeed.comm.get_rank(),
        local_rank=training_args.local_rank,
    )

    with linglong.running('Loading configs', comm=comm) as spinner:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant': False}
        spinner.write(model_args)
        spinner.write(data_args)
        spinner.write(training_args)

    with linglong.running('Loading the model', comm=comm, timer=True) as spinner:
        if model_args.pretrained_model is not None:
            model = linglong.LingLongLMHeadModel.from_pretrained(model_args.pretrained_model)
        elif model_args.model_config is not None:
            model = linglong.LingLongLMHeadModel(linglong.LingLongConfig.from_pretrained(model_args.model_config))
        else:
            raise ValueError('Either pretrained_model or model_config must be provided.')
        model_config = model.config
        if training_args.gradient_checkpointing:
            model.config.use_cache = False
        linglong.print_trainable_parameters(model, comm=comm, print_fn=spinner.write)
        spinner.write(model)

    with linglong.running('Loading the dataset', comm=comm, timer=True):
        train_dataset = linglong.records.load(
            path=data_args.training_data,
            meta=data_args.training_meta,
            use_pinyin=model_config.use_pinyin,
            load_attention_mask=data_args.load_attention_mask,
        )

    with linglong.running('Training', comm=comm, spinner=False):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == '__main__':
    linglong.init()
    main()
