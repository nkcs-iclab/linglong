import yaml
import fire


def main(rank):
    file = "../common/model-configs/317M-WSZ1024L24-LoRA.yaml"
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    config['lora_attn_dim'] = rank
    config['lora_attn_alpha'] = rank * 2
    with open(file, 'w') as f:
        yaml.safe_dump(config, f)


if __name__ == '__main__':
    fire.Fire(main)
