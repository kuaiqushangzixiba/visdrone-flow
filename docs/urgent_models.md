# 紧急优先模型实施说明

当前阶段只优先推进两个模型：

1. 三维空间流量预测模型
2. 三维空间资源分配模型

其他模型保留为后续扩展能力，不作为当前主线。

## 1. 三维空间流量预测模型

### 目标

整合历史飞行数据、模拟感知数据等，经时空对齐与特征工程构建标准化数据集；通过历史学习捕捉时空特征，结合在线学习优化，测算特定空域无人机流量分布。

### 当前代码

| 能力 | 代码 |
|---|---|
| 历史流量预测基线 | `src/visdrone_flow/models/historical_average.py` |
| 时空特征 Ridge 模型 | `src/visdrone_flow/models/spatial_temporal_ridge.py` |
| 在线学习 SGD 模型 | `src/visdrone_flow/models/online_spatial_temporal_sgd.py` |
| 历史/感知数据对齐 | `src/visdrone_flow/perception.py` |
| 特征工程 | `src/visdrone_flow/features.py` |

### 标准化数据集

统一对齐键：

```text
grid_id + height_layer + time_slot
```

历史飞行数据字段：

```text
flow_in, flow_out, occupancy, avg_speed, task_count,
weather_wind, weather_visibility, em_interference,
no_fly_flag, route_capacity
```

模拟感知数据字段：

```text
detected_uav_count
sensor_confidence
simulated_density
simulated_avg_speed
simulated_em_interference
```

融合逻辑：

```text
task_count = task_count + detected_uav_count * sensor_confidence
occupancy = max(occupancy, simulated_density)
avg_speed = avg_speed if available else simulated_avg_speed
em_interference = max(em_interference, simulated_em_interference)
```

### 模型公式

监督学习形式：

```text
X = [历史滞后流量, 邻居流量均值, 外生特征, 时间周期特征]
Y = [未来 1..H 个时间步流量]
```

Ridge：

```text
min_W ||Y - XW||_2^2 + alpha * ||W||_2^2
```

在线学习 SGD：

```text
W_t = W_{t-1} - eta * gradient(loss(y_t, x_t W_{t-1}) + alpha * ||W||_2^2)
```

输出：

```text
pred_flow
pred_density
congestion_score
warning_level
confidence
```

### 命令

```powershell
C:\ProgramData\anaconda3\python.exe run.py generate-sample
C:\ProgramData\anaconda3\python.exe run.py train --model online --records examples/sample_flow.csv --edges examples/sample_edges.csv --perception examples/sample_perception.csv --artifact artifacts/online_flow_model.pkl
C:\ProgramData\anaconda3\python.exe run.py predict --artifact artifacts/online_flow_model.pkl --records examples/sample_flow.csv --edges examples/sample_edges.csv --perception examples/sample_perception.csv --output artifacts/online_predictions.json
```

## 2. 三维空间资源分配模型

### 目标

实现无人机飞行轨迹规划、空域资源分配、安全间隔管理及任务负载均衡；自动输出优化后的无人机飞行计划、空域资源分配方案及任务执行建议。

### 当前代码

| 能力 | 代码 |
|---|---|
| 三维 A* 航迹规划 | `src/visdrone_flow/routing.py` |
| 时间片资源分配 | `src/visdrone_flow/allocation.py` |
| 网格/无人机/任务读取 | `src/visdrone_flow/state_io.py` |

### 资源分配逻辑

处理流程：

```text
1. 按任务优先级排序
2. 对每个任务遍历可用无人机
3. 用 A* 计算 UAV -> 任务起点 -> 任务终点
4. 检查载荷、航程、可达性
5. 构建 flight_plan: route node + ETA + time slot
6. 检查安全缓冲时间片内的容量占用
7. 加入负载均衡惩罚
8. 选择 allocation_score 最低方案
```

航程约束：

```text
distance(uav -> origin) + distance(origin -> destination)
  <= max_range_m * battery_pct / 100
```

时间片：

```text
slot = floor((eta - task_start_time) / time_granularity)
```

安全间隔管理：

```text
同一 node_id 在 slot ± safety_buffer_slots 范围内检查容量占用
```

分配目标函数：

```text
allocation_score =
  total_distance
+ mission_risk_cost
+ capacity_penalty
+ balance_penalty
- task_priority * 100
```

容量惩罚：

```text
if load > capacity:
  capacity_penalty += capacity_weight * (load - capacity)
else:
  capacity_penalty += 50 * load / capacity
```

负载均衡：

```text
balance_penalty =
  assigned_task_count(uav) * balance_weight
+ assigned_distance(uav) * 0.05
```

输出：

```text
task_id
uav_id
status
route_grid_sequence
flight_plan: [{seq, node_id, eta, slot}]
distance_m
allocation_score
capacity_penalty
balance_penalty
load_ratio
task_execution_advice
```

### 命令

```powershell
C:\ProgramData\anaconda3\python.exe run.py generate-operational-sample
C:\ProgramData\anaconda3\python.exe run.py allocate-resources --cells examples/sample_cells.csv --edges examples/sample_operational_edges.csv --uavs examples/sample_uavs.csv --tasks examples/sample_tasks.csv --time-granularity-minutes 5 --safety-buffer-slots 1 --output artifacts/resource_allocation.json
```

