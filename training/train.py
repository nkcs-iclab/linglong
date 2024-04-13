import dataclasses

import peft
from transformers import (
    Trainer,
    HfArgumentParser,
    TrainingArguments,
)
from peft import LoraConfig, TaskType

import linglong
import linglong.data.tfrecord


@dataclasses.dataclass
class ModelArguments:
    pretrained_model: str | None = dataclasses.field(
        default=None,
        metadata={'help': 'Pretrained model path.'},
    )
    model_config: str | None = dataclasses.field(
        default=None,
        metadata={'help': 'Model config path.'},
    )
    use_peft_lora: bool = dataclasses.field(
        default=False,
        metadata={"help": 'Enables PEFT LoRA for training.'},
    )
    lora_alpha: int = dataclasses.field(
        default=16,
        metadata={'help': 'The scaling factor for the low-rank matrices.'},
    )
    lora_dropout: float = dataclasses.field(
        default=0.1,
        metadata={'help': 'The dropout probability of the LoRA layers.'},
    )
    lora_r: int = dataclasses.field(
        default=64,
        metadata={'help': 'The dimension of the low-rank matrices.'},
    )
    lora_target_modules: str = dataclasses.field(
        default='c_attn,c_proj,c_fc',
        metadata={'help': 'Comma separated list of target modules to apply LoRA layers to.'},
    )


@dataclasses.dataclass
class DataArguments:
    training_data: str = dataclasses.field(
        metadata={'help': 'Training data path.'},
    )
    training_meta: str = dataclasses.field(
        default='train-meta.json',
        metadata={'help': 'The meta file\'s filename of the training data.'},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    is_main_process = training_args.process_index == 0

    with linglong.running('Loading configs', is_main_process=is_main_process) as spinner:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant': False}
        spinner.write(linglong.prettify({
            'model_args': model_args,
            'data_args': data_args,
            'training_args': training_args,
        }))

    with linglong.running('Loading the model', is_main_process=is_main_process, timer=True) as spinner:
        if model_args.pretrained_model is not None:
            model = linglong.LingLongForCausalLM.from_pretrained(model_args.pretrained_model)
        elif model_args.model_config is not None:
            model = linglong.LingLongForCausalLM(linglong.LingLongConfig.from_json_file(model_args.model_config))
        else:
            raise ValueError('Either pretrained_model or model_config must be provided.')
        if model_args.use_peft_lora:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                target_modules=model_args.lora_target_modules.split(','),
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                fan_in_fan_out=True,
                task_type=TaskType.CAUSAL_LM,
            )
            model = peft.get_peft_model(model, lora_config)
        linglong.print_trainable_parameters(model, is_main_process=is_main_process, print_fn=spinner.write)
        spinner.write(model)

    with linglong.running('Loading the dataset', is_main_process=is_main_process, timer=True):
        train_dataset = linglong.data.tfrecord.load_tfrecord_dataset(
            path=data_args.training_data,
            meta=data_args.training_meta,
            use_pinyin=model.config.use_pinyin,
        )

    with linglong.running('Training', is_main_process=is_main_process, spinner=False):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == '__main__':
    linglong.init()
    main()
