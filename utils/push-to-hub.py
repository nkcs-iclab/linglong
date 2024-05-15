import fire

from transformers import GenerationConfig

from linglong.configuration_linglong import LingLongConfig
from linglong.modeling_linglong import LingLongForCausalLM
from linglong.tokenization_linglong import Tokenizer


def main(
        repo_name: str,
        pretrained_model: str,
        model_commit_message: str = 'update model',
        tokenizer_commit_message: str = 'update tokenizer',
        push_model: bool = False,
        push_tokenizer: bool = False,
):
    LingLongConfig.register_for_auto_class()
    LingLongForCausalLM.register_for_auto_class('AutoModelForCausalLM')
    Tokenizer.register_for_auto_class()

    tokenizer = Tokenizer.from_pretrained(pretrained_model)
    model = LingLongForCausalLM.from_pretrained(pretrained_model)
    generation_config = GenerationConfig(do_sample=True, max_length=1024, top_k=20)
    model.generation_config = generation_config
    model.config._name_or_path = repo_name
    if push_model:
        model.push_to_hub(repo_name, commit_message=model_commit_message)
    if push_tokenizer:
        tokenizer.push_to_hub(repo_name, commit_message=tokenizer_commit_message)


if __name__ == '__main__':
    fire.Fire(main)
