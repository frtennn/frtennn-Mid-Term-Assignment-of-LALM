# Transformer从零实现

一个完整的Transformer模型从零实现，用于IWSLT2017英德数据集的机器翻译任务。

## 要求
- Python 3.8+
- PyTorch 2.4.0
- 推荐使用2080Ti及以上服务器

## 目录
transformer-from-scratch/
├── src/                        
│   ├── model.py             
│   ├── train.py              
│   ├── data_loader.py         
│   ├── config.py              
│   └── utils.py            
├── configs/
│   └── base.yaml               
├── scripts/
│   └── run.sh                  
├── requirements.txt          
├── README.md                  
├── train.py                   
├── test.py                    
├── train_no_positional.py     
└── test_no_positional.py      

## 运行指南
### 1.环境准备
```bash
conda create -n tf python=3.8
conda activate tf
pip install -r requirements.txt
```
### 2.训练模型
```bash
# ！！！（运行前需要将“/src/model.py”的第6行改为“from .utils import positional_encoding”）
python train.py --config configs/base.yaml --device cuda --seed 42
```

### 3.模型评估并输出样例
```bash
# ！！！（运行前需要将“/src/model.py”的第6行改为“from utils import positional_encoding”）
python test_.py --config configs/base.yaml --checkpoint checkpoints/best_model.pt --seed 42 --device cuda
```

### 4.不同头数注意力的transformer对比实验
```bash
# ！！！（运行前需要将“/src/model.py”的第6行改为“from .utils import positional_encoding”）
python train.py --config configs/ablation_2.yaml --device cuda --seed 42
python train.py --config configs/ablation_4.yaml --device cuda --seed 42
python train.py --config configs/ablation_8.yaml --device cuda --seed 42
```

### 5.评估不同头数的模型
```bash
# ！！！（运行前需要将“/src/model.py”的第6行改为“from utils import positional_encoding”）
python test.py --config configs/ablation_2.yaml --checkpoint ablation_models/2/best_model.pt --seed 42 --device cuda
python test.py --config configs/base.yaml --checkpoint checkpoints/best_model.pt --seed 42 --device cuda
python test.py --config configs/ablation_8.yaml --checkpoint ablation_models/8/best_model.pt --seed 42 --device cuda
```

### 6.消融位置编码
```bash
#无位置编码的消融transformer，配置使用之前消融的8头
# ！！！（运行前需要将“/src/model.py”的第6行改为“from utils import positional_encoding”）
python train_no_positional.py --config configs/ablation_4.yaml --seed 42 --device cuda --epochs 50
python test_no_positional.py --config configs/base.yaml --checkpoint no_positional_model/best_model.pt --device cuda
```




