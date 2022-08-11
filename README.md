# vanilla-gpt

Minimal, annotated implementation of GPT in PyTorch. STILL IN DEVELOPMENT

- [] implement classification (clf) as in finetune-transformer-lm
- [] repair weight tying
- [] ensure attention schemas are the same with finetune-transformer-lm
- [] create inference script
- [] host model weights and allow download (.pt or .bin?)
- [] annotate the model
- [] add easy evaluation / dataloaders
- [x] use Conv1D instead of nn.Linear
- [] use nn.Conv1D instead of custom Conv1D
- [] add dropouts for training
- [] initialise weights with the same scheme as finetune-transformer-lm
- [] is nn.LayerNorm the same as finetune-transformer-lm implementation of norm?