
# 使用特定种子运行主模型训练
# ！！！（运行前需要将“/src/model.py”的第6行改为“from .utils import positional_encoding”）
python train.py --config configs/base.yaml --device cuda --seed 42

# 主模型评估结果
# ！！！（运行前需要将“/src/model.py”的第6行改为“from utils import positional_encoding”）
python test_.py --config configs/base.yaml --checkpoint checkpoints/best_model.pt --seed 42 --device cuda

# 不同头数注意力的transformer对比实验
# 对消融实验整体的配置进行了调整，但是他们之间只有头数不同
# ！！！（运行前需要将“/src/model.py”的第6行改为“from .utils import positional_encoding”）
python train.py --config configs/ablation_2.yaml --device cuda --seed 42
python train.py --config configs/ablation_4.yaml --device cuda --seed 42
python train.py --config configs/ablation_8.yaml --device cuda --seed 42

# 测试集上评估不同头数
# ！！！（运行前需要将“/src/model.py”的第6行改为“from utils import positional_encoding”）
#python evaluate_trained_models.py --config configs/ablation_8.yaml --seed 42 --device cuda
python test.py --config configs/ablation_2.yaml --checkpoint ablation_models/2/best_model.pt --seed 42 --device cuda
python test.py --config configs/base.yaml --checkpoint checkpoints/best_model.pt --seed 42 --device cuda
python test.py --config configs/ablation_8.yaml --checkpoint ablation_models/8/best_model.pt --seed 42 --device cuda


#消融
#无位置编码的消融transformer，配置使用之前消融的8头
# ！！！（运行前需要将“/src/model.py”的第6行改为“from utils import positional_encoding”）
python train_no_positional.py --config configs/ablation_4.yaml --seed 42 --device cuda --epochs 50
python test_no_positional.py --config configs/base.yaml --checkpoint no_positional_model/best_model.pt --device cuda
