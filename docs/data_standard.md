# 低空流量预测数据标准

## 主键

```text
cell_key = grid_id + height_layer + time_slot
```

字段含义：

- `grid_id`: 外部标准空间网格编码，建议 GB/T 39409-2020 北斗网格位置码。
- `height_layer`: 高度层编号，从 0 开始。
- `time_slot`: 时间片起点，ISO 8601。

## 训练数据 records.csv

必填字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| grid_id | string | 标准网格码 |
| height_layer | int | 高度层 |
| time_slot | datetime | 时间片 |
| flow_in | float | 进入该网格的无人机流量 |

推荐字段：

| 字段 | 说明 |
|---|---|
| flow_out | 离开该网格流量 |
| occupancy | 占用率 |
| avg_speed | 平均速度 |
| task_count | 任务数量 |
| weather_wind | 风速 |
| weather_visibility | 能见度 |
| em_interference | 电磁干扰强度 |
| no_fly_flag | 是否禁飞 |
| route_capacity | 该网格容量 |
| height_ref | AGL / MSL / ELLIPSOID |

## 空间边 edges.csv

| 字段 | 说明 |
|---|---|
| source_grid_id | 起点网格 |
| source_height_layer | 起点高度层 |
| target_grid_id | 终点网格 |
| target_height_layer | 终点高度层 |
| weight | 边权重 |
| edge_type | adjacent / vertical / route / correlation |
| directed | 是否有向 |

## 输出

每个预测点输出：

| 字段 | 说明 |
|---|---|
| grid_id | 网格码 |
| height_layer | 高度层 |
| future_time_slot | 预测时间 |
| horizon_index | 第几个预测步 |
| pred_flow | 预测流量 |
| pred_density | 预测密度，等于 pred_flow / route_capacity |
| congestion_score | 拥堵分数 |
| warning_level | 0 正常、1 关注、2 拥堵、3 严重 |
| confidence | 置信度 |

