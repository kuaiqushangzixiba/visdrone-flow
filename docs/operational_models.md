# 四类低空算法模型实现

本项目已实现四个可运行的 P0 基线模型。所有模型统一使用：

```text
grid_id + height_layer + time_slot
```

其中 `grid_id` 由外部标准网格系统提供，推荐使用 GB/T 39409-2020 北斗网格位置码。代码不会伪造国家标准编码，只消费标准编码结果。

## 1. 空间电磁环境分析模型

文件：`src/visdrone_flow/electromagnetic.py`

方法：链路预算 + FSPL + 对数距离损耗 + 干扰叠加 + SINR。

核心公式：

```text
rssi = tx_power + tx_gain + rx_gain - path_loss
SINR = signal_power / (interference + noise)
communication_stability = f(RSSI, SINR)
```

输出：

- `rssi_dbm`
- `sinr_db`
- `interference_dbm`
- `interference_level`
- `communication_stability`
- `avoid_flag`
- `recommended_action`

后续升级点：把 `_path_loss_db` 替换为 Sionna RT 射线追踪结果。

## 2. 三维空间资源分配模型

文件：`src/visdrone_flow/allocation.py`

方法：约束贪心任务分配。

约束：

- 无人机载荷约束
- 剩余航程约束
- 航线可达性
- 网格容量约束
- 禁飞区通过航线代价抑制

输出：

- `task_id`
- `uav_id`
- `status`
- `route_grid_sequence`
- `distance_m`
- `allocation_score`
- `load_ratio`

后续升级点：替换为 OR-Tools CP-SAT / VRP 或 pymoo 多目标优化。

## 3. 航线规划与导航分析模型

文件：`src/visdrone_flow/routing.py`

方法：三维网格 A*。

代价函数：

```text
cost =
  distance
+ height_change
+ congestion_score
+ em_interference
+ weather_penalty
+ risk_score
+ no_fly_penalty
```

输出：

- `found`
- `route`
- `total_cost`
- `distance_m`
- `risk_cost`

后续升级点：接入 OMPL、RRT*、D* Lite、CBS/ECBS、多机动态重规划。

## 4. 低空飞行安全评估分析模型

文件：`src/visdrone_flow/safety.py`

方法：CPA/TCPA 冲突检测 + 网格风险矩阵。

冲突指标：

- `tcpa_s`
- `horizontal_cpa_m`
- `vertical_cpa_m`
- `conflict_risk`

网格风险因子：

- 禁飞区
- 风速
- 能见度
- 人口密度
- 拥堵
- 电磁干扰
- 外部风险分

输出：

- `overall_risk_score`
- `risk_level`
- `recommended_action`
- `conflicts`
- `grid_risks`

后续升级点：接入 NASA DAIDALUS / ICAROUS / WellClear。

## 命令示例

```powershell
C:\ProgramData\anaconda3\python.exe run.py generate-operational-sample
C:\ProgramData\anaconda3\python.exe run.py analyze-em --cells examples/sample_cells.csv --transmitters examples/sample_transmitters.csv --output artifacts/em_analysis.json
C:\ProgramData\anaconda3\python.exe run.py allocate-resources --cells examples/sample_cells.csv --edges examples/sample_operational_edges.csv --uavs examples/sample_uavs.csv --tasks examples/sample_tasks.csv --output artifacts/resource_allocation.json
C:\ProgramData\anaconda3\python.exe run.py plan-route --cells examples/sample_cells.csv --edges examples/sample_operational_edges.csv --start-grid BDG-L18-R00-C00 --start-height 1 --end-grid BDG-L18-R03-C03 --end-height 1 --output artifacts/route_plan.json
C:\ProgramData\anaconda3\python.exe run.py assess-safety --cells examples/sample_cells.csv --uavs examples/sample_uavs.csv --output artifacts/safety_assessment.json
```

