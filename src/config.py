import yaml
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 256
    vocab_size: int = 30000

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-9
    weight_decay: float = 0.01
    warmup_steps: int = 4000
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1
    save_dir: str = "checkpoints"
    log_dir: str = "logs"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)


def load_config(config_path: str) -> tuple:
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    model_config = ModelConfig.from_dict(config_dict['model'])
    training_config = TrainingConfig.from_dict(config_dict['training'])

    return model_config, training_config