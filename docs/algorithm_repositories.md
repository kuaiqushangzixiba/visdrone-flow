# 流量预测模块可借鉴算法与仓库

## 首选工程框架

| 技术 | 仓库 |
|---|---|
| LibCity 交通预测统一框架 | https://github.com/LibCity/Bigscity-LibCity |
| Torch Spatiotemporal / tsl | https://github.com/TorchSpatiotemporal/tsl |
| PyTorch Geometric Temporal | https://github.com/benedekrozemberczki/pytorch_geometric_temporal |

## 推荐模型

| 模型 | 仓库 | 适配建议 |
|---|---|---|
| Graph WaveNet | https://github.com/nnzhan/Graph-WaveNet | 主力候选。自适应邻接矩阵，适合低空网格隐含相关性。 |
| AGCRN | https://github.com/LeiBAI/AGCRN | 主力候选。节点自适应图卷积，适合航路关系不完整场景。 |
| PDFormer | https://github.com/BUAABIGSCity/PDFormer | 高精度候选。适合长时段、多区域预测。 |
| DCRNN | https://github.com/liyaguang/DCRNN | 经典扩散图卷积，适合有明确航路方向的图。 |
| STGCN | https://github.com/VeritasYin/STGCN_IJCAI-18 | 简洁高速，可作为深度学习基线。 |
| ASTGCN | https://github.com/guoshnBJTU/ASTGCN-2019-pytorch | 适合周期性强的流量预测。 |
| MTGNN | https://github.com/nnzhan/MTGNN | 适合多变量时间序列和自学习图结构。 |
| XGBoost | https://github.com/dmlc/xgboost | 工程基线和小数据阶段备选。 |

## 本项目当前选择

本项目先实现 `SpatialTemporalRidge`，因为它：

- 依赖少，Anaconda base 环境可直接运行。
- 使用同一套网格时空数据接口。
- 可解释，适合作为 P0 生产基线。
- 后续 Graph WaveNet、AGCRN、PDFormer 可以替换模型层，不需要推翻数据标准。

