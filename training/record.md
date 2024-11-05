1. 先微调prompt模型（MCPTPromptModel）
  ### 手动重新生成微调数据
  linglong-mcpt\datasets\fine-tuning-prompt.sh
  linglong-mcpt\datasets\configs\fine-tuning\local.yaml
  ### 重新训练模型
  需要调整的参数文件：
    linglong-mcpt\common\model-configs\317M-WSZ1024L24-PROMPT.yaml
    linglong-mcpt\training\ft-prompt-math.sh
  需要调整微调脚本用于固定参数，目的在于只微调prompt部分的参数而不微调全部参数，所以使用train-prompt脚本
  ### 测试微调后的模型

2. 微调LoRA模型
  2.1 重命名微调后的模型
  w->weight
  b->bias
  2.2 重新训练模型
  需要调整的参数文件：
    linglong-mcpt\common\model-configs\317M-WSZ1024L24-PROMPT-LoRA.yaml
    linglong-mcpt\training\ft-prompt-lora-math.sh
  需要调整微调脚本用于固定参数，目的在于只微调prompt部分的参数而不微调全部参数，所以适用train-prompt脚本