# 项目算法总结

本文档用于快速查看当前项目已经实现的算法数量、算法原理、核心公式、输入输出和对应代码位置。

## 1. 总览

当前项目按业务模块统计，共有 **5 个算法模块**：

| 序号 | 模块 | 当前实现方法 | 主要代码 |
|---|---|---|---|
| 1 | 三维空间流量预测 | Historical Average + Spatial Temporal Ridge | `src/visdrone_flow/models/` |
| 2 | 空间电磁环境分析 | Link Budget + FSPL + SINR | `src/visdrone_flow/electromagnetic.py` |
| 3 | 三维空间资源分配 | 约束贪心任务分配 | `src/visdrone_flow/allocation.py` |
| 4 | 航线规划与导航分析 | 三维网格 A* | `src/visdrone_flow/routing.py` |
| 5 | 低空飞行安全评估 | CPA/TCPA + 网格风险矩阵 | `src/visdrone_flow/safety.py` |

按代码中的可调用算法类统计，共有 **6 个算法实现**：

| 序号 | 算法类 | 文件 |
|---|---|---|
| 1 | `HistoricalAverageModel` | `src/visdrone_flow/models/historical_average.py` |
| 2 | `SpatialTemporalRidgeModel` | `src/visdrone_flow/models/spatial_temporal_ridge.py` |
| 3 | `ElectromagneticEnvironmentModel` | `src/visdrone_flow/electromagnetic.py` |
| 4 | `ResourceAllocationModel` | `src/visdrone_flow/allocation.py` |
| 5 | `AStarRoutePlanner` | `src/visdrone_flow/routing.py` |
| 6 | `SafetyAssessmentModel` | `src/visdrone_flow/safety.py` |

## 2. 统一数据标准

所有算法统一围绕同一个空间时间单元工作：

```text
cell_key = grid_id + height_layer + time_slot
```

其中：

| 字段 | 含义 |
|---|---|
| `grid_id` | 标准空间网格编码，建议接入 GB/T 39409-2020 北斗网格位置码 |
| `height_layer` | 高度层编号 |
| `time_slot` | 时间片 |
| `node_id` | 代码内部节点编号，格式为 `grid_id#Hheight_layer` |

公共数据结构主要在：

```text
src/visdrone_flow/schemas.py
src/visdrone_flow/grid.py
src/visdrone_flow/state_io.py
```

## 3. 三维空间流量预测算法

### 3.1 业务目标

根据历史低空网格流量、邻接网格流量、天气、电磁、任务数量、禁飞状态等特征，预测未来若干时间片内每个网格高度层的流量、密度、拥堵分数和预警等级。

### 3.2 当前实现一：Historical Average

代码：

```text
src/visdrone_flow/models/historical_average.py
```

原理：

对同一网格节点、同一小时的历史流量取平均值。如果该小时没有历史数据，则回退到该节点全局均值；如果节点也没有历史数据，则回退到全局均值。

公式：

```text
node_id = grid_id + "#H" + height_layer

pred_flow(node, hour) =
  mean(flow_in | node_id = node, hour(time_slot) = hour)

fallback:
  pred_flow(node) = mean(flow_in | node_id = node)
  pred_flow(global) = mean(flow_in)
```

适用场景：

| 场景 | 说明 |
|---|---|
| 数据量少 | 深度模型不稳定时作为兜底模型 |
| 快速基线 | 用于和复杂模型比较 |
| 生产兜底 | 主模型异常时返回稳定预测 |

### 3.3 当前实现二：Spatial Temporal Ridge

代码：

```text
src/visdrone_flow/models/spatial_temporal_ridge.py
src/visdrone_flow/features.py
```

原理：

把低空网格流量预测转换成监督学习问题。每个样本由目标网格自身历史滞后特征、邻居网格历史均值、外生特征和时间周期特征组成，然后用 Ridge 回归预测未来多个时间步。

输入特征：

```text
x =
[
  flow_lag_1,
  flow_lag_2,
  ...,
  flow_lag_history_steps,
  neighbor_flow_lag_1,
  neighbor_flow_lag_2,
  neighbor_flow_lag_3,
  flow_out,
  occupancy,
  avg_speed,
  task_count,
  weather_wind,
  weather_visibility,
  em_interference,
  no_fly_flag,
  route_capacity,
  hour_sin,
  hour_cos,
  dow_sin,
  dow_cos
]
```

监督学习形式：

```text
X: [samples, features]
Y: [samples, horizon_steps]
```

Ridge 目标函数：

```text
min_w ||Y - XW||_2^2 + alpha * ||W||_2^2
```

邻居均值：

```text
neighbor_flow(node, t) =
  sum(weight_i * flow(neighbor_i, t)) / sum(weight_i)
```

时间周期编码：

```text
hour_sin = sin(2 * pi * minute_of_day / 1440)
hour_cos = cos(2 * pi * minute_of_day / 1440)
dow_sin  = sin(2 * pi * day_of_week / 7)
dow_cos  = cos(2 * pi * day_of_week / 7)
```

输出：

```text
pred_flow
pred_density = pred_flow / route_capacity
congestion_score = pred_density
warning_level
confidence
```

运行命令：

```powershell
C:\ProgramData\anaconda3\python.exe run.py generate-sample
C:\ProgramData\anaconda3\python.exe run.py train --records examples/sample_flow.csv --edges examples/sample_edges.csv --artifact artifacts/flow_model.pkl
C:\ProgramData\anaconda3\python.exe run.py predict --artifact artifacts/flow_model.pkl --records examples/sample_flow.csv --edges examples/sample_edges.csv --output artifacts/predictions.json
```

后续升级方向：

| 模型 | 用途 |
|---|---|
| Graph WaveNet | 自适应邻接矩阵，适合隐藏空间相关性 |
| AGCRN | 节点自适应图卷积 |
| PDFormer | Transformer 类高精度交通预测 |
| DCRNN | 有向扩散图卷积，适合明确航路方向 |
| STGCN | 简洁高速的时空图卷积基线 |

## 4. 空间电磁环境分析算法

代码：

```text
src/visdrone_flow/electromagnetic.py
```

### 4.1 业务目标

对每个低空网格高度层计算通信强弱、干扰强度、SINR、通信稳定度和是否需要避让。

### 4.2 当前方法

当前是 P0 工程基线模型：

```text
链路预算 + FSPL 自由空间损耗 + 对数距离损耗 + 干扰叠加 + SINR
```

它不是高精度射线追踪，也不是全波电磁仿真。优点是依赖少、速度快、可直接接入业务数据。

### 4.3 核心公式

三维距离：

```text
d = sqrt((x_cell - x_tx)^2 + (y_cell - y_tx)^2 + (z_cell - z_tx)^2)
```

自由空间路径损耗：

```text
FSPL(dB) = 32.44 + 20log10(f_MHz) + 20log10(d_km)
```

附加损耗：

```text
excess_loss = 10 * (path_loss_exponent - 2) * log10(d_m)

path_loss =
  FSPL
+ excess_loss
+ terrain_loss
+ building_loss
+ weather_loss
```

接收功率：

```text
RSSI = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss
```

噪声功率：

```text
noise_dbm = -174 + 10log10(bandwidth_hz) + noise_figure_db
```

干扰：

```text
interference_mw = sum(rx_power_mw from non-serving transmitters and jammers)
```

信干噪比：

```text
SINR = signal_power / (interference + noise)
SINR_dB = 10log10(SINR)
```

通信稳定度：

```text
rssi_score = clip((RSSI - min_rssi_dbm) / 35, 0, 1)
sinr_score = clip((SINR_dB - min_sinr_db) / 24, 0, 1)

communication_stability =
  0.45 * rssi_score + 0.55 * sinr_score
```

干扰等级：

```text
interference_dbm > -65  -> 3 强干扰
interference_dbm > -80  -> 2 中干扰
interference_dbm > -95  -> 1 弱干扰
otherwise               -> 0 低干扰
```

### 4.4 输入输出

输入：

```text
cells.csv:
  grid_id, height_layer, center_x_m, center_y_m, center_z_m,
  terrain_loss_db, building_loss_db, weather_wind

transmitters.csv:
  transmitter_id, x_m, y_m, z_m, frequency_mhz,
  bandwidth_mhz, tx_power_dbm, tx_gain_dbi, rx_gain_dbi,
  noise_figure_db, role
```

输出：

```text
grid_id
height_layer
best_transmitter_id
rssi_dbm
sinr_db
noise_dbm
interference_dbm
interference_level
communication_stability
avoid_flag
recommended_action
```

运行命令：

```powershell
C:\ProgramData\anaconda3\python.exe run.py generate-operational-sample
C:\ProgramData\anaconda3\python.exe run.py analyze-em --cells examples/sample_cells.csv --transmitters examples/sample_transmitters.csv --output artifacts/em_analysis.json
```

后续升级方向：

```text
把 _path_loss_db() 替换成 Sionna RT 射线追踪结果。
```

## 5. 三维空间资源分配算法

代码：

```text
src/visdrone_flow/allocation.py
```

### 5.1 业务目标

给多架无人机、多任务、多空域网格分配任务和航线，要求满足载荷、航程、空域容量、禁飞区、路径可达性等约束。

### 5.2 当前方法

当前实现为约束贪心分配：

```text
1. 按任务优先级从高到低排序
2. 对每个任务遍历所有无人机
3. 用 A* 计算无人机到任务起点、任务起点到终点的路线
4. 检查载荷、航程、路线可达性
5. 计算分配代价
6. 选择代价最低的无人机执行任务
7. 更新无人机剩余航程、剩余载荷、网格容量占用
```

### 5.3 约束

载荷约束：

```text
remaining_payload(uav) >= required_payload(task)
```

航程约束：

```text
distance(uav -> origin) + distance(origin -> destination)
  <= max_range_m * battery_pct / 100
```

容量约束：

```text
reserved_count(grid_node) <= route_capacity(grid_node)
```

可达性约束：

```text
AStarRoutePlanner must find a route
```

### 5.4 目标函数

当前贪心选择分配分数最低的无人机：

```text
allocation_score =
  total_distance
+ mission_risk_cost
+ capacity_penalty
- task_priority * 100
```

容量惩罚：

```text
if load > capacity:
  capacity_penalty += 10000 * (load - capacity)
else:
  capacity_penalty += 50 * load / capacity
```

### 5.5 输入输出

输入：

```text
cells.csv
edges.csv
uavs.csv
tasks.csv
```

输出：

```text
task_id
uav_id
status
reason
route_grid_sequence
distance_m
allocation_score
load_ratio
```

运行命令：

```powershell
C:\ProgramData\anaconda3\python.exe run.py allocate-resources --cells examples/sample_cells.csv --edges examples/sample_operational_edges.csv --uavs examples/sample_uavs.csv --tasks examples/sample_tasks.csv --output artifacts/resource_allocation.json
```

后续升级方向：

| 方法 | 作用 |
|---|---|
| OR-Tools CP-SAT | 处理严格离散约束、时间片占用、容量限制 |
| OR-Tools VRP | 多无人机任务路径分配 |
| pymoo NSGA-II | 多目标优化，平衡时间、风险、距离、负载 |

## 6. 航线规划与导航分析算法

代码：

```text
src/visdrone_flow/routing.py
```

### 6.1 业务目标

在三维低空网格中，从起点网格到终点网格生成一条合法、低成本、低风险的航线。

### 6.2 当前方法

当前使用三维网格 A*：

```text
f(n) = g(n) + h(n)
```

其中：

```text
g(n) = 起点到当前节点的累计真实代价
h(n) = 当前节点到目标节点的欧氏距离启发式
```

### 6.3 单步代价函数

```text
step_cost =
  distance_weight * step_distance * edge_weight
+ height_change_weight * abs(z_current - z_next)
+ node_risk(next_node)
```

节点风险：

```text
node_risk =
  congestion_weight * congestion_score
+ em_weight * em_interference
+ risk_weight * risk_score
+ risk_weight * visibility_penalty
+ risk_weight * wind_penalty
+ no_fly_penalty
```

禁飞区：

```text
if no_fly_flag == 1:
  node_risk = 1_000_000
```

能见度惩罚：

```text
visibility_penalty = max(0, (2000 - visibility) / 2000)
```

风速惩罚：

```text
wind_penalty = max(0, (weather_wind - 10) / 10)
```

欧氏距离：

```text
distance =
  sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)
```

### 6.4 输入输出

输入：

```text
cells.csv:
  grid_id, height_layer, center_x_m, center_y_m, center_z_m,
  congestion_score, em_interference, risk_score,
  weather_wind, weather_visibility, no_fly_flag

edges.csv:
  source_grid_id, source_height_layer,
  target_grid_id, target_height_layer,
  weight, edge_type, directed
```

输出：

```text
found
route
total_cost
distance_m
risk_cost
message
```

运行命令：

```powershell
C:\ProgramData\anaconda3\python.exe run.py plan-route --cells examples/sample_cells.csv --edges examples/sample_operational_edges.csv --start-grid BDG-L18-R00-C00 --start-height 1 --end-grid BDG-L18-R03-C03 --end-height 1 --output artifacts/route_plan.json
```

后续升级方向：

| 方法 | 作用 |
|---|---|
| D* Lite | 动态环境重规划 |
| RRT* / PRM | 连续空间采样式路径规划 |
| OMPL | 成熟运动规划库 |
| B-spline / Minimum Snap | 航迹平滑与飞行可行性约束 |
| CBS / ECBS | 多无人机冲突消解 |

## 7. 低空飞行安全评估算法

代码：

```text
src/visdrone_flow/safety.py
```

### 7.1 业务目标

对当前低空空域和无人机状态进行风险评估，输出整体风险、冲突对、危险网格、预警等级和处置建议。

### 7.2 当前方法

当前模型由两部分组成：

```text
1. CPA/TCPA 无人机冲突检测
2. 网格风险矩阵
```

### 7.3 CPA/TCPA 冲突检测

相对位置：

```text
r = p_ownship - p_intruder
```

相对速度：

```text
v = v_ownship - v_intruder
```

最近接近时间：

```text
TCPA = - (r dot v) / ||v||^2
TCPA = clip(TCPA, 0, lookahead_s)
```

最近接近距离：

```text
p_cpa = r + v * TCPA
horizontal_cpa = sqrt(p_cpa_x^2 + p_cpa_y^2)
vertical_cpa = abs(p_cpa_z)
```

冲突风险：

```text
h_ratio = max(0, 1 - horizontal_cpa / horizontal_separation)
v_ratio = max(0, 1 - vertical_cpa / vertical_separation)
time_ratio = max(0, 1 - TCPA / lookahead_s)

conflict_risk =
  0.45 * h_ratio
+ 0.35 * v_ratio
+ 0.20 * time_ratio
```

默认阈值：

```text
horizontal_separation_m = 120
vertical_separation_m = 30
lookahead_s = 300
```

### 7.4 网格风险矩阵

风险因子：

```text
no_fly
weather_wind
weather_visibility
population_density
congestion_score
em_interference
risk_score
```

网格风险：

```text
grid_risk =
  max(
    no_fly,
    0.20 * wind
  + 0.20 * visibility
  + 0.20 * population
  + 0.15 * congestion
  + 0.15 * em
  + 0.10 * inherited_risk
  )
```

整体风险：

```text
overall_risk =
  0.55 * max(grid_risk)
+ 0.45 * max(conflict_risk)
```

风险等级：

```text
risk >= 0.85 -> severe
risk >= 0.65 -> high
risk >= 0.35 -> attention
otherwise    -> normal
```

处置建议：

```text
severe    -> reject_or_reroute
high      -> manual_review_or_delay
attention -> monitor
normal    -> allow
```

### 7.5 输入输出

输入：

```text
cells.csv
uavs.csv
```

输出：

```text
summary:
  overall_risk_score
  risk_level
  recommended_action
  conflict_count
  unsafe_grid_count

conflicts:
  ownship_id
  intruder_id
  tcpa_s
  horizontal_cpa_m
  vertical_cpa_m
  conflict_risk
  warning_level
  recommended_action

grid_risks:
  grid_id
  height_layer
  grid_risk_score
  warning_level
  risk_level
  recommended_action
```

运行命令：

```powershell
C:\ProgramData\anaconda3\python.exe run.py assess-safety --cells examples/sample_cells.csv --uavs examples/sample_uavs.csv --output artifacts/safety_assessment.json
```

后续升级方向：

| 方法 | 作用 |
|---|---|
| NASA DAIDALUS | Detect-and-Avoid 告警与机动建议 |
| NASA ICAROUS | 无人机地理围栏、安全监控 |
| WellClear | 空中最小安全间隔标准 |
| Monte Carlo | 失效概率、坠落风险、天气扰动验证 |

## 8. 命令汇总

生成流量预测样例：

```powershell
C:\ProgramData\anaconda3\python.exe run.py generate-sample
```

训练流量预测模型：

```powershell
C:\ProgramData\anaconda3\python.exe run.py train --records examples/sample_flow.csv --edges examples/sample_edges.csv --artifact artifacts/flow_model.pkl
```

执行流量预测：

```powershell
C:\ProgramData\anaconda3\python.exe run.py predict --artifact artifacts/flow_model.pkl --records examples/sample_flow.csv --edges examples/sample_edges.csv --output artifacts/predictions.json
```

生成四类作业模型样例：

```powershell
C:\ProgramData\anaconda3\python.exe run.py generate-operational-sample
```

运行电磁分析：

```powershell
C:\ProgramData\anaconda3\python.exe run.py analyze-em --cells examples/sample_cells.csv --transmitters examples/sample_transmitters.csv --output artifacts/em_analysis.json
```

运行资源分配：

```powershell
C:\ProgramData\anaconda3\python.exe run.py allocate-resources --cells examples/sample_cells.csv --edges examples/sample_operational_edges.csv --uavs examples/sample_uavs.csv --tasks examples/sample_tasks.csv --output artifacts/resource_allocation.json
```

运行航线规划：

```powershell
C:\ProgramData\anaconda3\python.exe run.py plan-route --cells examples/sample_cells.csv --edges examples/sample_operational_edges.csv --start-grid BDG-L18-R00-C00 --start-height 1 --end-grid BDG-L18-R03-C03 --end-height 1 --output artifacts/route_plan.json
```

运行安全评估：

```powershell
C:\ProgramData\anaconda3\python.exe run.py assess-safety --cells examples/sample_cells.csv --uavs examples/sample_uavs.csv --output artifacts/safety_assessment.json
```

运行测试：

```powershell
C:\ProgramData\anaconda3\python.exe -m pytest -q
```

## 9. 当前阶段说明

当前实现是可运行的 P0 算法基线，重点是：

```text
统一数据结构
统一时空网格
可训练
可预测
可输出 JSON
可被前端或后端服务调用
```

后续如果需要提升精度，可以在保持输入输出不变的前提下替换内部模型：

| 当前模块 | 后续可替换模型 |
|---|---|
| 流量预测 | Graph WaveNet, AGCRN, PDFormer, DCRNN, STGCN |
| 电磁分析 | Sionna RT, QuaDRiGa, ns-3 |
| 资源分配 | OR-Tools CP-SAT, OR-Tools VRP, pymoo |
| 航线规划 | OMPL, RRT*, D* Lite, CBS/ECBS |
| 安全评估 | NASA DAIDALUS, ICAROUS, WellClear |

