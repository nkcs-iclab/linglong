import yaml
import fire


def main(path, dataset):
    file = "configs/local.yaml"
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    # config[dataset]['model']['checkpoint'] = path
    config[dataset]['model_lora']['checkpoint'] = path
    with open(file, 'w') as f:
        yaml.safe_dump(config, f)


if __name__ == '__main__':
    fire.Fire(main)
