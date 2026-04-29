# VisDrone Flow Forecasting

低空无人机三维网格流量预测算法模块。项目不包含前端，只负责接收标准化网格时空数据，训练模型，并向前端/业务系统返回预测流量、拥堵分数和预警等级。

## 当前实现

- 标准数据结构：`grid_id + height_layer + time_slot`
- 训练模型：`SpatialTemporalRidge`
- 兜底模型：`HistoricalAverage`
- 空间电磁环境分析：FSPL / SINR / 干扰热力图基线
- 三维空间资源分配：约束贪心任务分配基线
- 航线规划与导航分析：三维网格 A*
- 低空飞行安全评估：CPA/TCPA 冲突检测 + 网格风险矩阵
- 支持空间邻接边：水平邻接、垂直高度层邻接、航路连接都可输入为 `edges.csv`
- 支持命令行训练、预测
- 支持标准库 HTTP API，无需 FastAPI/Uvicorn

## 快速运行

使用你的 Anaconda Python：

```powershell
C:\ProgramData\anaconda3\python.exe run.py generate-sample --records examples/sample_flow.csv --edges examples/sample_edges.csv
C:\ProgramData\anaconda3\python.exe run.py train --records examples/sample_flow.csv --edges examples/sample_edges.csv --artifact artifacts/flow_model.pkl
C:\ProgramData\anaconda3\python.exe run.py predict --artifact artifacts/flow_model.pkl --records examples/sample_flow.csv --edges examples/sample_edges.csv --output artifacts/predictions.json
```

启动算法服务：

```powershell
C:\ProgramData\anaconda3\python.exe run.py serve --artifact artifacts/flow_model.pkl --host 127.0.0.1 --port 8010
```

接口：

- `GET /health`
- `POST /predict`

## 数据主键

所有算法模块后续都应统一使用：

```text
cell_key = grid_id + height_layer + time_slot
```

其中 `grid_id` 建议使用 GB/T 39409-2020《北斗网格位置码》或项目指定的国家标准网格编码。当前代码不伪造官方编码实现，只把 `grid_id` 当作外部标准网格码接入。

## 推荐后续深度模型

当前项目先交付可运行基线。后续可在同一数据接口下接入：

- Graph WaveNet
- AGCRN
- PDFormer
- DCRNN
- STGCN

仓库地址见 [docs/algorithm_repositories.md](docs/algorithm_repositories.md)。

## 四类作业模型

生成样例并运行：

```powershell
C:\ProgramData\anaconda3\python.exe run.py generate-operational-sample
C:\ProgramData\anaconda3\python.exe run.py analyze-em --cells examples/sample_cells.csv --transmitters examples/sample_transmitters.csv --output artifacts/em_analysis.json
C:\ProgramData\anaconda3\python.exe run.py allocate-resources --cells examples/sample_cells.csv --edges examples/sample_operational_edges.csv --uavs examples/sample_uavs.csv --tasks examples/sample_tasks.csv --output artifacts/resource_allocation.json
C:\ProgramData\anaconda3\python.exe run.py plan-route --cells examples/sample_cells.csv --edges examples/sample_operational_edges.csv --start-grid BDG-L18-R00-C00 --start-height 1 --end-grid BDG-L18-R03-C03 --end-height 1 --output artifacts/route_plan.json
C:\ProgramData\anaconda3\python.exe run.py assess-safety --cells examples/sample_cells.csv --uavs examples/sample_uavs.csv --output artifacts/safety_assessment.json
```

具体方法见 [docs/operational_models.md](docs/operational_models.md)。
