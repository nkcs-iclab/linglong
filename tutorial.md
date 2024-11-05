# 不同微调方式及对应测试方法

## 仅进行lora微调

### 生成微调数据

    1. 修改linglong-mcpt\datasets\configs\fine-tuning\local.yaml
        '''
        Math23K:
        template_id: 0
        '''
    2. 修改linglong-mcpt\datasets\fine-tuning.sh
       '''--model_config /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-LoRA.yaml'''
    3. 执行 bash linglong-mcpt\datasets\fine-tuning.sh

### 微调练模型
    1. 调整参数 linglong-mcpt\common\model-configs\317M-WSZ1024L24-LoRA.yaml
    2. 修改update_rank.py
        '''
        file =
        '''
    3. 执行 linglong-mcpt\training\ft-lora-math.sh

### 测试微调后的模型
    1. 修改 linglong-mcpt\evaluation\lora-math.sh
        '''
        for r in 390 406 422 438 454 480 496 2048
        BASE_DATA_DIR=
        '''
    2. 修改 linglong-mcpt\evaluation\update_rank.py
        '''
        file =
        '''
    3. 修改 linglong-mcpt\evaluation\update_config.py
        '''
        # config[dataset]['model']['checkpoint'] = path
        config[dataset]['model_lora']['checkpoint'] = path
        '''
    4. 修改 linglong-mcpt\evaluation\configs\local.yaml
        '''
        Math23K:
          evaluation_method: generation
          evaluation_metric: dataset_math23k_metric
          model:
            checkpoint: /home/lidongwen/lidongwen/model/300M/V12-renamed-all.pt
            config: /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-LoRA-eva.yaml
          model_lora:
            checkpoint:
          split: dev
          template_id: 1
        '''
    5. 执行 linglong-mcpt\evaluation\lora-math.sh

## 仅进行prompt微调
### 生成微调数据
    1. 修改linglong-mcpt\datasets\configs\fine-tuning\local.yaml
        '''
        Math23K:
          template_id: 10
          special_tokens:
            prompt_token: '[unused88]'
        '''
    2. 修改/home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml
        '''
        prompt_length: xxx
        '''
    3. 修改linglong-mcpt\datasets\fine-tuning.sh
       '''--model_config /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml'''
    4. 执行 bash linglong-mcpt\datasets\fine-tuning.sh


### 微调练模型
    1. 调整参数 linglong-mcpt\common\model-configs\317M-WSZ1024L24-PROMPT.yaml
    2. 调整linglong-mcpt\training\ft-prompt-math.sh
        '''
        PromptLength=
        '''
    3. 执行 linglong-mcpt\training\ft-prompt-math.sh

### 测试微调后的模型
    1. 修改 linglong-mcpt\evaluation\prompt-math.sh
        '''
        BASE_DATA_DIR
        version=
        '''
    2. 修改 linglong-mcpt\evaluation\update_config.py
        '''
        config[dataset]['model']['checkpoint'] = path
        # config[dataset]['model_lora']['checkpoint'] = path
        '''
    3. 修改 linglong-mcpt\evaluation\configs\local.yaml
        '''
        Math23K:
          evaluation_method: generation
          evaluation_metric: dataset_math23k_metric
          model:
            checkpoint: /home/lidongwen/lidongwen/model/300M/V12.pt
            config: /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml
          special_tokens:
            part_separator: '[unused1]'
          split: dev
          template_id: 10
        '''
    4. 执行 linglong-mcpt\evaluation\prompt-math.sh

## 先prompt微调再进行模型全参数量微调
### 生成微调数据
    1. 修改linglong-mcpt\datasets\configs\fine-tuning\local.yaml
        '''
        Math23K:
          template_id: 10
          special_tokens:
            prompt_token: '[unused88]'
        '''
    2. 修改/home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml
        '''
        prompt_length: xxx
        '''
    3. 修改linglong-mcpt\datasets\fine-tuning.sh
       '''--model_config /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml'''
    4. 执行 bash linglong-mcpt\datasets\fine-tuning.sh

### 微调练模型
    1. 调整参数 linglong-mcpt\common\model-configs\317M-WSZ1024L24-PROMPT.yaml
    2. 调整linglong-mcpt\training\ft-prompt-math.sh
       '''
       PromptLength=
       '''
    3. 执行 linglong-mcpt\training\ft-prompt-math.sh

### 测试微调后的模型
    1. 修改 linglong-mcpt\evaluation\prompt-math.sh
        '''
        BASE_DATA_DIR
        version=
        '''
    2. 修改 linglong-mcpt\evaluation\update_config.py
        '''
        config[dataset]['model']['checkpoint'] = path
        # config[dataset]['model_lora']['checkpoint'] = path
        '''
    3. 修改 linglong-mcpt\evaluation\configs\local.yaml
        '''
        Math23K:
          evaluation_method: generation
          evaluation_metric: dataset_math23k_metric
          model:
            checkpoint: /home/lidongwen/lidongwen/model/300M/V12.pt
            config: /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml
          special_tokens:
            part_separator: '[unused1]'
          split: dev
          emplate_id: 10
        '''
    4. 执行 linglong-mcpt\evaluation\prompt-math.sh
### 选择效果较好的一个模型
### 微调练模型
    1. 调整参数 linglong-mcpt\common\model-configs\317M-WSZ1024L24-PROMPT.yaml
    2. 调整linglong-mcpt\training\ft-prompt-all-two-stage-math.sh
        '''
        PromptLength=
        load_model=
        '''
    3. 执行 linglong-mcpt\training\ft-prompt-all-two-stage-math.sh
### 测试微调后的模型
    1. 修改 linglong-mcpt\evaluation\prompt-math.sh
        '''
        BASE_DATA_DIR
        version=
        '''
    2. 修改 linglong-mcpt\evaluation\update_config.py
        '''
        config[dataset]['model']['checkpoint'] = path
        # config[dataset]['model_lora']['checkpoint'] = path
        '''
    3. 修改 linglong-mcpt\evaluation\configs\local.yaml
        '''
        Math23K:
          evaluation_method: generation
          evaluation_metric: dataset_math23k_metric
          model:
            checkpoint: /home/lidongwen/lidongwen/model/300M/V12.pt
            config: /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml
          special_tokens:
            part_separator: '[unused1]'
          split: dev
          template_id: 10
        '''
    4. 执行 linglong-mcpt\evaluation\prompt-math.sh

## 先prompt微调再进行模型lora微调
### 生成微调数据
    1. 修改linglong-mcpt\datasets\configs\fine-tuning\local.yaml
        '''
        Math23K:
          template_id: 10
          special_tokens:
            prompt_token: '[unused88]'
        '''
    2. 修改/home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml
        '''
        prompt_length: xxx
        '''
    3. 修改linglong-mcpt\datasets\fine-tuning.sh
       '''--model_config /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml'''
    4. 执行 bash linglong-mcpt\datasets\fine-tuning.sh

### 微调练模型
    1. 调整参数 linglong-mcpt\common\model-configs\317M-WSZ1024L24-PROMPT.yaml
    2. 调整linglong-mcpt\training\ft-prompt-math.sh
        '''
        PromptLength=
        '''
    3. 执行 linglong-mcpt\training\ft-prompt-math.sh

### 测试微调后的模型
    1. 修改 linglong-mcpt\evaluation\prompt-math.sh
        '''
        BASE_DATA_DIR
        version=
        '''
    2. 修改 linglong-mcpt\evaluation\update_config.py
        '''
        config[dataset]['model']['checkpoint'] = path
        # config[dataset]['model_lora']['checkpoint'] = path
        '''
    3. 修改 linglong-mcpt\evaluation\configs\local.yaml
        '''
        Math23K:
          evaluation_method: generation
          evaluation_metric: dataset_math23k_metric
          model:
            checkpoint: /home/lidongwen/lidongwen/model/300M/V12.pt
            config: /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml
          special_tokens:
            part_separator: '[unused1]'
          split: dev
          template_id: 10
        '''
    4. 执行 linglong-mcpt\evaluation\prompt-math.sh
### 选择效果较好的一个模型
### 微调练模型
    1. 调整参数 linglong-mcpt\common\model-configs\317M-WSZ1024L24-PROMPT-LORA.yaml
    2. 修改模型参数文件中参数的命名
    3. 调整linglong-mcpt\training\ft-prompt-lora-two-stage-math.sh
        '''
        PromptLength=
        load_model=
        '''
    4. 执行 linglong-mcpt\training\ft-prompt-lora-two-stage-math.sh
### 测试微调后的模型
    1. 修改 linglong-mcpt\evaluation\prompt-lora-math.sh
        '''
        for r in  8 32 128 192 256 374 512
        BASE_DATA_DIR
        version=
        '''
    2. 修改 linglong-mcpt\evaluation\update_config.py
        '''
        # config[dataset]['model']['checkpoint'] = path
        config[dataset]['model_lora']['checkpoint'] = path
        '''
    3. 修改 linglong-mcpt\evaluation\update_rank.py
            '''
            file =
            '''
    4. 修改 linglong-mcpt\evaluation\configs\local.yaml
        '''
        Math23K:
          evaluation_method: generation
          evaluation_metric: dataset_math23k_metric
          model:
            checkpoint: 效果较好的模型
            config: /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT-LoRA.yaml
          model_lora:
            checkpoint:
          special_tokens:
            part_separator: '[unused1]'
          split: dev
          template_id: 10
        '''
    4. 执行 linglong-mcpt\evaluation\prompt-lora-math.sh

## prompt微调+LoRA微调
### 生成微调数据
    1. 修改linglong-mcpt\datasets\configs\fine-tuning\local.yaml
        '''
        Math23K:
          template_id: 10
          special_tokens:
            prompt_token: '[unused88]'
        '''
    2. 修改linglong-mcpt\datasets\fine-tuning.sh
        '''--model_config /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT-LoRA.yaml'''
    3. 执行 bash linglong-mcpt\datasets\fine-tuning.sh
### 微调练模型
    1. 调整参数 linglong-mcpt\common\model-configs\317M-WSZ1024L24-PROMPT-LoRA.yaml
    2. 调整linglong-mcpt\training\ft-prompt-lora-math.sh
        '''
        PromptLength=
        '''
    3. 修改update_rank.py
        '''
        file =
        '''
    4. 执行 linglong-mcpt\training\ft-prompt-lora-math.sh
### 测试微调后的模型
    1. 修改 linglong-mcpt\evaluation\prompt-lora-math.sh
        '''
        for r in  8 32 128 192 256 374 512
        BASE_DATA_DIR=
    '''
    2. 修改 linglong-mcpt\evaluation\update_rank.py
        '''
        file =
        '''
    3. 修改 linglong-mcpt\evaluation\update_config.py
        '''
        config[dataset]['model']['checkpoint'] = path
        # config[dataset]['model_lora']['checkpoint'] = path
        '''
    4. 修改 linglong-mcpt\evaluation\configs\local.yaml
        '''
        Math23K:
          evaluation_method: generation
          evaluation_metric: dataset_math23k_metric
          model:
            checkpoint: /home/lidongwen/lidongwen/model/300M/V12.pt
            config: /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT-LoRA.yaml
          special_tokens:
            part_separator: '[unused1]'
          split: dev
          template_id: 10
        '''
    5. 执行 linglong-mcpt\evaluation\prompt-lora-math.sh
## prompt微调+全参数微调
### 生成微调数据
    1. 修改linglong-mcpt\datasets\configs\fine-tuning\local.yaml
        '''
        Math23K:
          template_id: 10
          special_tokens:
            prompt_token: '[unused88]'
        '''
    2. 修改/home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml
        '''
        prompt_length: xxx
        '''
    3. 修改linglong-mcpt\datasets\fine-tuning.sh
       '''--model_config /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml'''
    4. 执行 bash linglong-mcpt\datasets\fine-tuning.sh

### 微调练模型
    1. 调整参数 linglong-mcpt\common\model-configs\317M-WSZ1024L24-PROMPT.yaml
    2. 调整linglong-mcpt\training\ft-prompt-all-math.sh
        '''
        PromptLength=
        '''
    3. 执行 linglong-mcpt\training\ft-prompt-all-math.sh

### 测试微调后的模型
    1. 修改 linglong-mcpt\evaluation\prompt-math.sh
        '''
        BASE_DATA_DIR
        version=
        '''
    2. 修改 linglong-mcpt\evaluation\update_config.py
        '''
        config[dataset]['model']['checkpoint'] = path
        # config[dataset]['model_lora']['checkpoint'] = path
        '''
    3. 修改 linglong-mcpt\evaluation\configs\local.yaml
        '''
        Math23K:
          evaluation_method: generation
          evaluation_metric: dataset_math23k_metric
          model:
            checkpoint: /home/lidongwen/lidongwen/model/300M/V12.pt
            config: /home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24-PROMPT.yaml
          special_tokens:
            part_separator: '[unused1]'
          split: dev
          template_id: 10
        '''
    4. 执行 linglong-mcpt\evaluation\prompt-math.sh
