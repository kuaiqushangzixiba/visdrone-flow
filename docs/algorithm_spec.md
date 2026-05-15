# VisDrone 算法模块规范文档

> **项目**：低空无人机三维网格流量预测与空域管理算法服务
> **统一数据主键**：`cell_key = grid_id + height_layer + time_slot`
> **节点内部编号**：`node_id = grid_id + "#H" + height_layer`
> **grid_id 标准**：建议接入 GB/T 39409-2020《北斗网格位置码》

---

## 算法一：三维空间流量预测模型

| 字段 | 内容 |
|------|------|
| **算法名称** | 三维空间流量预测模型 |
| **负责人** | 算法工程师 |
| **更新时间** | 2026-05-14 |
| **对应代码** | `models/spatial_temporal_ridge.py`、`models/historical_average.py`、`models/online_spatial_temporal_sgd.py`、`features.py`、`perception.py` |

### 1. 一句话简介

整合历史飞行数据与模拟感知数据，经时空对齐与特征工程构建标准化数据集；通过历史学习捕捉时空特征，结合在线学习优化，测算特定空域无人机流量分布。

### 2. 输入与输出

**输入 (Input)：**

| 参数/数据 | 类型 | 说明 |
|-----------|------|------|
| `records.csv` | CSV (Pydantic: `FlowRecord`) | 历史飞行流量记录，必含 `grid_id`、`height_layer`、`time_slot`、`flow_in`；可选 `flow_out`、`occupancy`、`avg_speed`、`task_count`、`weather_wind`、`weather_visibility`、`em_interference`、`no_fly_flag`、`route_capacity` |
| `edges.csv` | CSV (Pydantic: `GridEdge`) | 网格邻接边，含 `source_grid_id`、`source_height_layer`、`target_grid_id`、`target_height_layer`、`weight`、`edge_type`（`adjacent`/`vertical`/`route`）、`directed` |
| `perception.csv`（可选） | CSV | 模拟感知数据，含 `detected_uav_count`、`sensor_confidence`、`simulated_density`、`simulated_avg_speed`、`simulated_em_interference`，按 `grid_id + height_layer + time_slot` 对齐 |

**输出 (Output)：**

| 结果 | 类型 | 说明 |
|------|------|------|
| `PredictionResponse` | JSON (Pydantic) | 含 `request_id`、`model_name`、`generated_at`、`predictions[]`；每个 `PredictionPoint` 含 `grid_id`、`height_layer`、`future_time_slot`、`horizon_index`、`pred_flow`、`pred_density`、`congestion_score`、`warning_level`、`confidence` |

**预警等级：**

| congestion_score | warning_level |
|---|---|
| ≥ 1.2 | severe (3) |
| ≥ 0.85 | congested (2) |
| ≥ 0.65 | attention (1) |
| < 0.65 | normal (0) |

### 3. 依赖技术栈

| 层级 | 技术 |
|------|------|
| 核心模型 | `SpatialTemporalRidgeModel`（Ridge 回归 α=1.0）、`HistoricalAverageModel`（三层回退）、`OnlineSpatialTemporalSGDModel`（增量 SGD） |
| 特征工程 | `PanelBuilder` 稀疏→稠密矩阵转换、28 维特征构建、时间周期编码（sin/cos） |
| 框架 | Python 3.10+、scikit-learn（Pipeline / StandardScaler / Ridge / SGDRegressor）、NumPy、Pandas |
| 序列化 | `ModelBundle` pickle 格式 |

### 4. 算法流程与模块联系

**步骤一（预处理—数据归一化与感知融合）：**
- `normalize_records()`：time_slot 转 datetime → height_layer 转 int → 生成 node_id → 缺失值填 0
- `build_standard_flow_dataset()`（perception.py）：将感知数据按 cell_key 对齐合并到流量记录中，更新 task_count（+detected_uav_count×confidence）、occupancy（取 max）、avg_speed（优先真实值）、em_interference（取 max）
- `PanelBuilder`：将稀疏的 `(time_slot, node_id)` 记录转换为稠密的 `[T×N]` 浮点矩阵，NaN 用列均值填充

**步骤二（核心计算—特征构建+模型训练/预测）：**
- `build_supervised()`（features.py）：Sliding Window 构建训练样本 —— history_steps=12 步历史 → 28 维特征向量，horizon_steps=6 步未来 → 6 维目标向量
- 28 维特征构成：
  - 12 维 `flow_lag_1~12`：目标节点自身流量滞后
  - 3 维 `neighbor_flow_lag_1~3`：邻接网格加权均值
  - 9 维外生特征：`flow_out`、`occupancy`、`avg_speed`、`task_count`、`weather_wind`、`weather_visibility`、`em_interference`、`no_fly_flag`、`route_capacity`
  - 4 维时间编码：`hour_sin`、`hour_cos`、`dow_sin`、`dow_cos`
- 模型 Pipeline：`StandardScaler` → `Ridge(α=1.0, random_state=42)`，目标函数：`min_w ||Y - XW||²₂ + α||W||²₂`
- 或使用 `OnlineSpatialTemporalSGDModel`：`StandardScaler.partial_fit()` + `SGDRegressor(loss="squared_error", penalty="l2", learning_rate="invscaling")`，每个 horizon 独立回归器
- 兜底：`HistoricalAverageModel` 三层回退 —— (node_id, hour) → node_id → global_mean

**步骤三（后处理—置信度与预警计算）：**
- 预测值裁剪：`y = max(y, 0)`
- density = pred_flow / route_capacity
- congestion_score = max(0, density)
- confidence = 0.92 - z×0.08 - horizon×0.035，z = |pred - μ| / σ，范围 [0.2, 0.98]

**模块间数据流向：**

```
[records.csv + edges.csv + perception.csv]
  → [PanelBuilder 稠密化]
    → [Feature Engineering 28维]
      → [Pipeline(StandardScaler→Ridge) / SGDRegressor]
        → [置信度计算 + 预警分类]
          → [PredictionResponse JSON]
  → 下游消费者：算法4(资源分配, congestion_score) / 算法9(航线规划, node_risk) / 算法10(安全评估, grid_risk)
```

---

## 算法二：地形跟随、地形回避模型

| 字段 | 内容 |
|------|------|
| **算法名称** | 地形跟随、地形回避模型 |
| **负责人** | 算法工程师 |
| **更新时间** | 2026-05-14 |
| **对应代码** | 全新规划，暂无代码 |

### 1. 一句话简介

通过前视传感器获取地形数据，使无人机沿地形轮廓飞行并保持恒定安全高度；同时模拟调整飞行姿态（俯仰角、滚转角）与油门参数，确保纵向高度稳定，生成绕开障碍物的横向避障指令，明确避障方向、距离及姿态调整参数。

### 2. 输入与输出

**输入 (Input)：**

| 参数/数据 | 类型 | 说明 |
|-----------|------|------|
| 地形高程数据 (DEM) | GeoTIFF（30m 分辨率 DSM/DTM） | 区域地形网格，含每个像素的海拔高度 |
| LiDAR 前视点云 | LAS/LAZ 实时流 | 每秒约 50k–300k 点，含 x/y/z/ intensity，用于前视地形与障碍物检测 |
| UAV 当前状态 | JSON (Pydantic: `UavState`) | `uav_id`、`x_m`、`y_m`、`z_m`、`vx_mps`、`vy_mps`、`vz_mps`、`pitch_deg`、`roll_deg`、`yaw_deg`、`heading_deg`、`speed_mps` |
| 航线参考路径 | JSON (Pydantic: `RoutePlanRequest`) | 由算法 9 航线规划输出的目标航点序列 |
| 控制参数配置 | YAML | `h_clearance`：安全离地高度（15~60m）；`t_lookahead`：前视时间（2~5s）；`max_pitch_deg`：最大俯仰角限制（±25°）；`max_roll_deg`：最大滚转角限制（±45°）；`safe_width`：横向安全宽度（≥ 翼展×1.5） |

**输出 (Output)：**

| 结果 | 类型 | 说明 |
|------|------|------|
| `TerrainFollowingCommand` | JSON | `pitch_angle_deg`（期望俯仰角）、`throttle_adjust`（油门归一化值 0.5~1.2）、`altitude_target_m`（目标高度） |
| `ObstacleAvoidanceCommand` | JSON | `avoid_direction`（LEFT/RIGHT/UP/DOWN/HOVER）、`avoid_distance_m`、`roll_angle_deg`、`pitch_angle_deg`、`throttle_adjust`、`duration_s`、`return_to_path`（bool） |
| `FlightAttitudeCommand`（合并） | JSON | 时间戳 + 上述两个指令的联合输出，供飞控执行 |

### 3. 依赖技术栈

| 层级 | 技术 |
|------|------|
| 高程处理 | GDAL / Rasterio（DEM 读取与查询） |
| LiDAR 处理 | PDAL / Open3D（点云去噪、下采样、聚类） |
| 控制算法 | 级联 PID（外环位置/速度→内环姿态角）、前馈补偿 |
| 避障算法 | 几何避障（左右通路评估 + 转向解算）、Pure Pursuit 回线 |
| 运行环境 | Python 3.10+、NumPy、SciPy |

### 4. 算法流程与模块联系

**步骤一（预处理—地形与障碍物感知）：**
- DEM 加载与裁剪：根据 UAV 当前区域加载对应 DEM 瓦片，构建空间索引（R-tree）
- LiDAR 点云去噪：去除孤立点/飞点 → 统计滤波 → 体素下采样（voxel size = 1m）
- 障碍物聚类：DBSCAN/eigenvalue-based 分割 → 提取障碍物边界框（x_center, y_center, width, height）
- 前视点查询：`d_lookahead = v_current × t_lookahead`，向前视角方向投影查询地形高程

**步骤二（核心计算—纵向地形跟随控制）：**
- 前视点地形查询：`z_terrain = DEM.query(x + d_lookahead×cos(ψ), y + d_lookahead×sin(ψ))`
- 期望高度：`z_desired = z_terrain + h_clearance`
- 外环（高度 PID）：`e_alt = z_desired - z_current → pitch_cmd = Kp_alt×e_alt + Ki_alt×∫e_alt + Kd_alt×de_alt/dt`
- 内环（俯仰 PID）：`e_pitch = pitch_cmd - pitch_current → elevator_cmd = Kp_pitch×e_pitch + Ki_pitch×∫e_pitch + Kd_pitch×de_pitch/dt`
- 油门速度控制：`e_vel = v_desired - v_current → throttle_cmd = Kp_thr×e_vel + Ki_thr×∫e_vel + Kd_thr×de_vel/dt + throttle_ff`
- 前馈补偿：`throttle_ff = m×g×(sinθ + cosθ) / T_max`

**步骤三（核心计算—横向避障控制）：**
- 左右通路评估：以 UAV 航向为中心线，向左/右各扫描 safe_width，测量自由空间
- 方向决策：`IF left_clearance ≥ right_clearance AND left_clearance ≥ safe_width → LEFT；ELIF right_clearance ≥ safe_width → RIGHT；ELIF vertical_clearance ≥ safe_height → UP/DOWN；ELSE → HOVER`
- 避障距离：`d_avoid = max(d_min, v×t_response + v²/(2×a_brake))`
- 转向角：`turn_angle = atan2(w_obs/2 + margin, d_detection)`
- 滚转角指令：`roll_cmd = clamp(Kp_roll×turn_angle, -max_roll, +max_roll)`
- 偏航率：`yaw_rate = g×tan(roll_cmd)/v_current`

**步骤四（后处理—回线与指令输出）：**
- 越过障碍物后，Pure Pursuit 算法平滑返回 A* 参考路径
- 当横向误差 < 1m 时切回航线跟踪模式
- 输出包含 return_to_path 标志和预计回线时间

**模块间数据流向：**

```
[DEM 高程 + LiDAR 点云 + UAV 状态]
  → [地形查询 + 障碍物聚类]
    → [高度 PID (外环) + 俯仰 PID (内环)] → pitch_cmd, throttle_cmd
    → [通路评估 + 避障解算] → roll_cmd, yaw_rate, avoid_direction
      → [指令合并 + 回线规划]
        → FlightAttitudeCommand JSON
  → 上游：算法9(航线规划, 提供参考路径)
  → 下游：算法6(碰撞风险, 避障指令用于风险再评估)
```

---

## 算法三：空间电磁环境分析模型

| 字段 | 内容 |
|------|------|
| **算法名称** | 空间电磁环境分析模型 |
| **负责人** | 算法工程师 |
| **更新时间** | 2026-05-14 |
| **对应代码** | `electromagnetic.py` |

### 1. 一句话简介

对无人机所处低空空间进行电磁兼容性分析及电磁干扰测算，对高频（雷达站）、中频（通信基站）、低频（电力设施）干扰源进行差异化分析，通过热力图和等高线直观展示区域的电磁环境分布。

### 2. 输入与输出

**输入 (Input)：**

| 参数/数据 | 类型 | 说明 |
|-----------|------|------|
| `cells.csv` | CSV (Pydantic: `GridCell`) | 必含 `grid_id`、`height_layer`、`center_x_m`、`center_y_m`、`center_z_m`；可选 `terrain_loss_db`、`building_loss_db`、`weather_wind` |
| `transmitters.csv` | CSV (Pydantic: `Transmitter`) | 必含 `transmitter_id`、`x_m`、`y_m`、`z_m`、`frequency_mhz`；可选 `bandwidth_mhz`（默认10MHz）、`tx_power_dbm`（默认30）、`tx_gain_dbi`、`rx_gain_dbi`、`noise_figure_db`（默认7）、`role`（`base_station`/`relay`/`uav`/`jammer`） |
| `interference_type_config`（新增） | YAML | 干扰源分类规则：`radar_station`（高频 >3GHz）、`base_station`（中频 1~3GHz）、`power_facility`（低频 <1GHz） |
| 区域边界 (AOI) | GeoJSON | 分析区域范围，用于热力图/等高线裁剪（如仙居全域） |

**输出 (Output)：**

| 结果 | 类型 | 说明 |
|------|------|------|
| EM Analysis JSON | JSON | `grid_id`、`height_layer`、`best_transmitter_id`、`rssi_dbm`、`sinr_db`、`noise_dbm`、`interference_dbm`、`interference_level`（0~3）、`communication_stability`（0~1）、`avoid_flag`、`recommended_action` |
| InterferenceTypeStats（新增） | JSON | 按干扰源类型分组统计：各类型的影响网格数、平均干扰强度、最大干扰位置 |
| EM Heatmap（新增） | GeoTIFF/PNG | 干扰强度空间插值热力图，叠加在区域地图上；叠加高频/中频/低频干扰源位置标注 |
| EM Contour（新增） | GeoJSON | 等干扰强度等高线，划分高干扰区（红色 > -65dBm）、中干扰区（黄色 -80~-65dBm）、低干扰区（绿色 < -95dBm） |

**干扰等级与处置建议：**

| interference_dbm | interference_level | 说明 | recommended_action |
|---|---|---|---|
| > -65 dBm | 3 | 强干扰 | 换频或增加中继 |
| > -80 dBm | 2 | 中干扰 | 调整路线或升高高度 |
| > -95 dBm | 1 | 弱干扰 | 监控 |
| ≤ -95 dBm | 0 | 低干扰 | 正常飞行 |

### 3. 依赖技术栈

| 层级 | 技术 |
|------|------|
| 传播模型 | FSPL（自由空间路径损耗）、对数距离损耗（exponent=2.2） |
| 信号模型 | Link Budget、SINR（信干噪比） |
| 可视化 | Matplotlib / Folium / GDAL（热力图插值与等高线生成） |
| 空间分析 | SciPy（IDW/Kriging 空间插值）、Shapely |
| 运行环境 | Python 3.10+、NumPy、Pandas |

### 4. 算法流程与模块联系

**步骤一（预处理—数据加载与干扰源分类）：**
- 读取 cells + transmitters → 按 role/frequency_mhz 分类干扰源
- 高频源（>3GHz）：雷达站 → 独立标注
- 中频源（1~3GHz）：通信基站/中继 → 独立标注
- 低频源（<1GHz）：电力设施/广播 → 独立标注
- Jammers 无论频率均单独标记

**步骤二（核心计算—链路预算与 SINR）：**
- 三维距离：`d = sqrt((x_cell-tx_x)² + (y_cell-tx_y)² + (z_cell-tx_z)²)`
- FSPL：`32.44 + 20·log₁₀(f_MHz) + 20·log₁₀(d_km)`
- 附加损耗：`excess = 10×(exponent-2)×log₁₀(d_m)` + `terrain_loss` + `building_loss` + `weather_loss`（max 8dB）
- RSSI：`tx_power + tx_gain + rx_gain - path_loss`
- 噪声：`-174 + 10·log₁₀(BW_hz) + NF`
- SINR：`10·log₁₀(signal_mw / (interference_mw + noise_mw))`
- 通信稳定度：`0.45×rssi_score + 0.55×sinr_score`

**步骤三（后处理—可视化与分类输出）：**
- 按干扰源类型分组统计影响范围和强度
- IDW/Kriging 空间插值 → 生成干扰强度热力图
- 等值线追踪 → 生成等高线 (GeoJSON)
- 按阈值着色：高干扰区红色（> -65dBm）、中干扰区黄色、低干扰区绿色

**模块间数据流向：**

```
[cells.csv + transmitters.csv + interference_type_config]
  → [干扰源分类 (高频/中频/低频/Jammer)]
    → [FSPL + excess_loss + building/terrain/weather loss]
      → [RSSI → SINR → communication_stability]
        → [avoid_flag + recommended_action]
        → [空间插值 → 热力图 + 等高线]
  → 下游消费者：算法1(流量, em_interference 外生特征) / 算法9(航线, node_risk em_weight=60) / 算法10(安全, grid_risk em=15%)
```

---

## 算法四：三维空间资源分配模型

| 字段 | 内容 |
|------|------|
| **算法名称** | 三维空间资源分配模型 |
| **负责人** | 算法工程师 |
| **更新时间** | 2026-05-14 |
| **对应代码** | `allocation.py` |

### 1. 一句话简介

实现多无人机飞行轨迹规划、空域资源分配、安全间隔管理及任务负载均衡；结合高精度三维地理数据构建可视化三维空域管理环境，自动输出优化后的飞行计划、空域资源分配方案及任务执行建议。

### 2. 输入与输出

**输入 (Input)：**

| 参数/数据 | 类型 | 说明 |
|-----------|------|------|
| `cells.csv` | CSV (Pydantic: `GridCell`) | 空域网格含 `grid_id`、`height_layer`、`center_x/y/z_m`、`route_capacity`（默认 8~12）、`congestion_score`、`no_fly_flag` |
| `edges.csv` | CSV (Pydantic: `GridEdge`) | 邻接边 w/ `weight`（1.0 默认）、`edge_type` |
| `uavs.csv` | CSV (Pydantic: `UavState`) | 无人机状态含 `uav_id`、`grid_id`、`height_layer`、`x/y/z_m`、`speed_mps`、`battery_pct`、`max_range_m`、`payload_capacity_kg`、`current_payload_kg`、`priority` |
| `tasks.csv` | CSV (Pydantic: `MissionTask`) | 任务需求含 `task_id`、`origin_grid_id/height`、`dest_grid_id/height`、`priority`、`required_payload_kg`、`earliest_time`、`latest_time` |
| 3D 地理数据（新增） | CityGML / Cesium 3D Tiles | 建筑高度、禁飞区/限制区边界、特殊空域走廊 |
| 用户调整参数（新增） | JSON/YAML | `time_granularity_minutes`（默认5）、`safety_buffer_slots`（默认1）、任务优先级覆盖、空域容量调整系数 |

**输出 (Output)：**

| 结果 | 类型 | 说明 |
|------|------|------|
| Allocation Result | JSON | `task_id`、`uav_id`、`status`（assigned / unassigned）、`reason`、`route_grid_sequence`（node_id 列表）、`distance_m`、`allocation_score`、`load_ratio` |
| Flight Plan（新增） | JSON | 时间轴展开的飞行计划：各 UAV 在每时间片的 grid_id → route_capacity 占用表 |
| Resource Utilization（新增） | JSON/GeoTIFF | 空域容量使用率图：各网格 load/capacity 比值 |
| Visualization（新增） | Cesium/Deck.gl | 3D 视图展示分配结果、路径、容量热点 |

### 3. 依赖技术栈

| 层级 | 技术 |
|------|------|
| 分配算法 | 约束贪心（P0 基线）→ OR-Tools CP-SAT / VRP（后续） |
| 路径规划 | `AStarRoutePlanner`（`routing.py`，子调用） |
| 空间管理 | `ReservationPolicy`：时间片粒度 + 安全缓冲槽位 |
| 可视化 | Cesium ion / deck.gl、GeoPandas |
| 运行环境 | Python 3.10+、Pandas、NumPy |

### 4. 算法流程与模块联系

**步骤一（预处理—约束空间构建）：**
- 读取 cells/edges → 构建网格容量索引
- 读取 uavs → 计算 `remaining_range = max_range × battery%`、`remaining_payload = payload_capacity - current_payload`
- 读取 tasks → 按 priority 降序排序
- 初始化 `reserved` 容量表（UAV+time_slot 二维索引）

**步骤二（核心计算—约束贪心分配）：**
- 对每个 task（按优先级）：
  - 遍历所有 UAV → 检查四大约束：
    1. 载荷：`remaining_payload(uav) ≥ required_payload(task)`
    2. 航程：`A*_distance(uav→origin→dest) ≤ remaining_range(uav)`
    3. 容量：`reserved_count ≤ route_capacity`（考虑时间片粒度）
    4. 可达性：A* 必须找到完整路径
  - 计算分配分数：`score = total_distance + risk_cost + capacity_penalty - priority×100`
  - 选择 score 最小的 UAV
- 更新 reserved 容量表、remaining_range、remaining_payload

**步骤三（后处理—飞行计划生成与可视化）：**
- 时间轴展开：每 5min 时间片记录各 UAV 位置和网格占用
- 容量使用率：`load_ratio = max(reserved / capacity across route)`
- 3D 可视化：路径线 + 网格着色（按容量使用率红/黄/绿）

**模块间数据流向：**

```
[cells + edges + uavs + tasks + 3D地理数据]
  → [约束空间构建]
    → [任务优先级排序]
      → [A* 子调用(算法9) 计算每条 UAV→Task 路径]
        → [四大约束检查]
          → [分配分数计算 → 选择最低代价 UAV]
            → [reserved 容量表更新]
              → [飞行计划 JSON + 容量热力图]
  → 上游依赖：算法9(航线规划, A* sub-call) / 算法1(流量预测, congestion_score)
  → 下游消费者：算法5(监控, 计划vs实际对比) / 算法10(安全, 计划路径的risk评估)
```

---

## 算法五：飞行活动监控分析模型

| 字段 | 内容 |
|------|------|
| **算法名称** | 飞行活动监控分析模型 |
| **负责人** | 算法工程师 |
| **更新时间** | 2026-05-14 |
| **对应代码** | `perception.py`（数据融合部分）、`schemas.py`（`UavState` Pydantic 模型）；主体全新 |

### 1. 一句话简介

构建虚拟与现实同步的动态监控系统，将无人机位置、状态等数据映射到三维数字空间，实现飞行活动全生命周期监测、分析与优化，包括第一视角、全局俯瞰视角、局部放大视角，并自动输出针对性决策建议。

### 2. 输入与输出

**输入 (Input)：**

| 参数/数据 | 类型 | 说明 |
|-----------|------|------|
| UAV 遥测流 | JSON Stream / WebSocket（≥1Hz） | `uav_id`、`timestamp`、`x/y/z_m`、`vx/vy/vz_mps`、`pitch/roll/yaw_deg`、`battery_pct`、`signal_strength`、`sensor_status`、`gps_fix_quality` |
| 飞行计划 | JSON | 来自算法 4 输出的 Flight Plan，含 planned_waypoints 与 scheduled_times |
| 3D 场景模型 | 3D Tiles / glTF | 区域地形、建筑、空域边界三维模型 |
| 感知传感器数据 | JSON | `detected_uav_count`、`sensor_confidence`、周围障碍物相对位置 |

**输出 (Output)：**

| 结果 | 类型 | 说明 |
|------|------|------|
| `DigitalTwinState` | JSON | 实时 3D 场景中所有 UAV 的位置/姿态映射，含 `uav_id`、`x/y/z_m`、`orientation_quat`、`velocity_vector`、`battery_pct`、`trail_history` |
| `MonitoringAlert` | JSON | 异常检测告警：`deviation_from_plan`（航线偏离 m）、`speed_anomaly`（异常速度）、`geofence_violation`（越界）、`battery_critical`（低电量）、`signal_loss`（信号中断） |
| `DecisionSuggestion` | JSON | 自动决策建议：`attitude_adjustment`（姿态修正）、`route_optimization`（航线优化建议）、`risk_mitigation`（风险规避措施）、`emergency_landing`（紧急备降建议） |
| Multi-View Stream | WebSocket | 第一视角（FPS）、全局俯瞰（Top-down）、局部放大（Zoom-in）三路视频流 |

### 3. 依赖技术栈

| 层级 | 技术 |
|------|------|
| 状态估计 | Kalman Filter / Extended Kalman Filter（EKF，优化 IMU+GPS 融合） |
| 数字孪生 | Cesium / Unity（3D 实时渲染）、Cesium ion（3D Tiles 托管） |
| 实时通信 | WebSocket（Python `websockets` / `fastapi`） |
| 异常检测 | 统计过程控制（SPC）、Isolation Forest / LSTM Autoencoder（后续） |
| 运行环境 | Python 3.10+、Node.js（Web 前端）、CesiumJS |

### 4. 算法流程与模块联系

**步骤一（预处理—遥测数据接入与时间对齐）：**
- 订阅 UAV 遥测 WebSocket → 解析 JSON → 时间对齐（按时间戳排序缓冲，1s 窗口内聚合）
- GPS 坐标转本地 ENU 坐标（以区域中心为原点）
- 缺失值处理：线性插值（最大间隔 2s），丢失 >5s 标记 signal_loss
- 传感器置信度加权：GPS 精度 + IMU 漂移补偿

**步骤二（核心计算—状态估计与数字孪生同步）：**
- Kalman Filter 状态向量：`[x, y, z, vx, vy, vz, roll, pitch, yaw]`
- 预测步：基于运动模型（恒定速度 + 噪声协方差 Q）
- 更新步：融合 GPS + IMU + 气压计 + 视觉里程计等多传感器
- 数字孪生映射：估计状态 → Cesium Entity 更新（position / orientation / model）
- 偏差检测：planned vs estimated 的横向误差 > 10m → deviation_alert

**步骤三（后处理—告警与决策建议）：**
- 规则引擎：越界检测（geofence 多边形包含判断）、低电量预警（< 15%）、信号丢失（>5s 无更新）
- 统计异常检测：速度/加速度是否超出 3σ 范围
- 决策建议生成：根据告警类型从预定义决策模板库匹配
- 多视角渲染分发：WebSocket 推送（FPS / Top-down / Zoom-in）

**模块间数据流向：**

```
[UAV 遥测流 + 飞行计划 + 3D 场景模型]
  → [时间对齐 + ENU 坐标转换]
    → [Kalman Filter 状态估计]
      → [偏差检测 + 异常检测]
        → [数字孪生 Cesium Entity 更新]
        → [告警 + 决策建议生成]
  → 上游依赖：算法4(资源分配, FlightPlan 作为监控基准)
  → 下游消费者：算法10(安全评估, 实时UAV状态作为冲突检测输入) / 算法6(碰撞风险, 实时轨迹)
```

---

## 算法六：障碍物碰撞风险分析模型

| 字段 | 内容 |
|------|------|
| **算法名称** | 障碍物碰撞风险分析模型 |
| **负责人** | 算法工程师 |
| **更新时间** | 2026-05-14 |
| **对应代码** | 全新规划，暂无代码 |

### 1. 一句话简介

通过多传感器数据融合实现飞行环境全面分析，对静态障碍物（建筑物、通信高塔、桥梁、电力输电线塔、高大树木、山峰等）进行环境感知、风险量化评估和路径规划，以红/黄/绿三级色标输出风险可视化结果。

### 2. 输入与输出

**输入 (Input)：**

| 参数/数据 | 类型 | 说明 |
|-----------|------|------|
| 障碍物 GIS 图层 | GeoPackage / Shapefile | 静态障碍物分类：`building`（建筑高度≥15m）、`tower`（通信塔/电力塔高度+mast diameter）、`bridge`（桥梁桥面+桥塔）、`tree`（高大树木 树冠半径+高度）、`peak`（山峰 DSM 高点） |
| 点云数据 | LAS/LAZ（实时或预处理） | LiDAR 动态障碍物检测，含 x/y/z/intensity/classification |
| UAV 当前状态 + 计划航线 | JSON | `uav_id`、`x/y/z_m`、`vx/vy/vz_mps`、`planned_trajectory`（来自算法9的 path waypoints） |
| 配置参数 | YAML | `horizontal_buffer`（水平安全距离 120m）、`vertical_buffer`（垂直安全距离 30m）、`risk_threshold_high`（0.85）、`risk_threshold_medium`（0.65） |

**输出 (Output)：**

| 结果 | 类型 | 说明 |
|------|------|------|
| `CollisionRiskAssessment` | JSON | 每个 potential collision pair：`obstacle_id`、`obstacle_type`、`min_distance_m`、`tcpa_s`、`collision_probability`、`risk_level`（`high` ≥0.85 / `medium` ≥0.65 / `low` < 0.65） |
| `RiskZoneMap` | GeoJSON | 空间栅格化后的风险区域：高风险区（红色 polygon）、中风险区（黄色）、低风险区（绿色） |
| `AvoidancePathSuggestion` | JSON | 修改后的安全路径建议（调用算法 9 的重规划或算法 2 的避障指令） |
| 3D Dynamic Simulation | Cesium/Unity | 碰撞风险区域三维动态演示：UAV 轨迹 + 障碍物包围盒 + 危险锥 |

### 3. 依赖技术栈

| 层级 | 技术 |
|------|------|
| 点云处理 | PDAL / Open3D（点云分类、分割、配准） |
| 三维占据栅格 | OctoMap（概率占据地图，分辨率 1~5m） |
| 碰撞检测 | GJK/EPA 算法（凸体碰撞检测）、FCL（Flexible Collision Library） |
| 风险量化 | 概率碰撞模型（考虑 UAV 位置不确定性协方差 + 障碍物测量噪声） |
| 空间分析 | GeoPandas、Shapely、GDAL |
| 运行环境 | Python 3.10+、C++（FCL 绑定）、NumPy |

### 4. 算法流程与模块联系

**步骤一（预处理—多传感器融合与障碍物建模）：**
- 静态障碍物 GIS 图层加载 → 按类型构建三维包围盒（Bounding Box）：
  - 建筑物：LOD1 体块（footprint × height）
  - 通信塔/电力塔：圆柱体 + 顶部球体
  - 桥梁：桥面 box + 桥塔锥体
  - 高大树木：球体树冠 + 锥体树干
  - 山峰：DSM mesh → 简化凸包
- LiDAR 动态点云 → 去噪 → 聚类 → 动态障碍物包围盒
- 三维占据栅格合并：静态 + 动态障碍物统一为 OctoMap

**步骤二（核心计算—碰撞风险评估）：**
- 轨迹采样：对 UAV 计划航线以 0.5s 为间隔采样位置点
- 最小距离计算：每个采样点与 OctoMap 中占据栅格的最小欧氏距离
- 碰撞概率：`P(collision) = ∫ PDF_UAV(x) × P(occupied|x) dx`
- 等效简化：比较 `min_distance` 与 `horizontal_buffer + vertical_buffer` 的加权比值
- 风险等级分类：`high`（d < buffer×0.5）、`medium`（d < buffer）、`low`（d ≥ buffer）

**步骤三（后处理—风险可视化与路径建议）：**
- 风险区域栅格化：对每个 5m×5m 栅格叠加 UAV 轨迹碰撞概率 → 红/黄/绿色标
- 高风险区域生成避障路径建议（调用算法 9 A* 重规划或算法 2 避障指令）
- Cesium 3D 动态演示：UAV 轨迹线 + 障碍物包围盒（半透明着色） + 危险锥渲染

**模块间数据流向：**

```
[GIS 障碍物层 + LiDAR 点云 + UAV 状态 + 计划航线]
  → [障碍物包围盒建模]
    → [OctoMap 三维占据栅格合并]
      → [轨迹采样 + 最小距离计算]
        → [碰撞概率评估 → risk_level 分类]
          → [风险区域 GeoJSON (红/黄/绿)]
          → [避障路径建议]
          → [3D 动态模拟]
  → 上游依赖：算法9(航线规划, planned_trajectory) / 算法5(监控, 实时UAV状态) / 算法2(地形回避, 避障指令验证)
  → 下游消费者：算法10(安全评估, collision_risk因子) / 算法4(资源分配, 禁入区域更新)
```

---

## 算法七：飞行三维空间覆盖分析模型

| 字段 | 内容 |
|------|------|
| **算法名称** | 飞行三维空间覆盖分析模型 |
| **负责人** | 算法工程师 |
| **更新时间** | 2026-05-14 |
| **对应代码** | 全新规划，暂无代码 |

### 1. 一句话简介

对飞行高度、传感器探测范围等进行三维特征建模，结合任务时间限制和区域边界，计算并对比不同飞行方案下的空间覆盖效果，推荐最优覆盖方案。

### 2. 输入与输出

**输入 (Input)：**

| 参数/数据 | 类型 | 说明 |
|-----------|------|------|
| 飞行方案集 | JSON/YAML | 多个 FlightScheme：`scheme_id`、`uav_count`、`formation`（grid/line/spiral/random）、各 UAV `waypoints`、`altitude`、`speed` |
| 传感器参数 | JSON | `sensor_type`（EO/IR/SAR/LiDAR）、`horizontal_fov_deg`、`vertical_fov_deg`、`max_range_m`、`swath_width_m`、`resolution_m` |
| 任务区域边界 | GeoJSON Polygon | 需要覆盖的目标区域边界 |
| DEM | GeoTIFF | 地形高程（用于 LOS 遮挡计算） |
| 时间约束 | JSON | `start_time`、`end_time`、`max_duration_s` |

**输出 (Output)：**

| 结果 | 类型 | 说明 |
|------|------|------|
| `CoverageResult` | JSON | `scheme_id`、`coverage_percentage`（%）、`overlap_percentage`（%）、`gap_areas`（未覆盖区 GeoJSON）、`time_efficiency`（覆盖面积/时间） |
| `CoverageHeatmap` | GeoTIFF | 每个栅格被传感器探测到的次数（0 / 1 / 2+） |
| `ComparisonReport` | JSON | 所有飞行方案覆盖效果排序表：覆盖面积、覆盖百分比、平均重复覆盖率、时间效率 |
| `OptimalScheme` | JSON | 推荐最佳飞行方案及其覆盖参数 |

### 3. 依赖技术栈

| 层级 | 技术 |
|------|------|
| 视锥建模 | 三维几何计算（传感器视锥体 = FOV 棱锥 × max_range） |
| LOS 遮挡 | GDAL viewshed / 自定义射线追踪（基于 DEM） |
| 覆盖计算 | Shapely（geometry 并集/交集）、NumPy |
| 优化 | SciPy.optimize、模拟退火 / 遗传算法（最优航点搜索） |
| 运行环境 | Python 3.10+、GDAL、NumPy、SciPy |

### 4. 算法流程与模块联系

**步骤一（预处理—区域离散化与传感器建模）：**
- 任务区域边界 → 按 resolution_m 离散化为规则栅格（如 100m×100m cells）
- 各高程 cell 关联 DEM 海拔
- 传感器视锥建模：以 UAV 位置为顶点，FOV 为锥角，range 为锥高，定义视线方程
- 考虑传感器安装角度（gimbal pitch offset）

**步骤二（核心计算—覆盖模拟）：**
- 对各飞行方案的每个航点：
  - 计算视锥覆盖区域（地面投影多边形）
  - LOS 遮挡判断：射线检测是否有地形阻断
  - 标记可见栅格：`visited_count[cell] += 1`
- 累积覆盖：`coverage = |visited_cells| / |total_cells|`
- 重复覆盖率：`overlap = |visited_count≥2| / |visited_cells|`
- 未覆盖区：`{cells where visited_count = 0}`

**步骤三（后处理—方案对比与推荐）：**
- 多方案排序：按 (coverage_percentage × 0.6 + time_efficiency × 0.4) 综合评分
- 覆盖热力图：每个栅格标注被探测次数（冷色=0, 暖色=2+）
- 最优方案推荐：输出推荐方案的详细覆盖参数
- 未覆盖区标注供人工审查

**模块间数据流向：**

```
[飞行方案集 + 传感器参数 + 边界 + DEM + 时间约束]
  → [区域离散化 + 视锥建模]
    → [每个方案每个航点 LOS 覆盖模拟]
      → [累积覆盖统计 + 遮挡记录]
        → [覆盖率 + 重复覆盖率 + 未覆盖区]
          → [方案对比排序 + 覆盖热力图 + 最优推荐]
  → 上游依赖：算法9(航线规划, waypoints)
  → 下游消费者：算法9(航线规划, 覆盖优化反馈) / 算法4(资源分配, 飞行方案评估)
```

---

## 算法八：飞行可视域分析模型

| 字段 | 内容 |
|------|------|
| **算法名称** | 飞行可视域分析模型 |
| **负责人** | 算法工程师 |
| **更新时间** | 2026-05-14 |
| **对应代码** | 全新规划，暂无代码 |

### 1. 一句话简介

模拟无人机在复杂环境中的视线覆盖与遮挡风险，包括可视化飞行路径上的可见区域与盲区、评估传感器覆盖范围、识别关键遮挡物并优化任务规划。

### 2. 输入与输出

**输入 (Input)：**

| 参数/数据 | 类型 | 说明 |
|-----------|------|------|
| DSM（数字地表模型） | GeoTIFF（0.5~5m 分辨率） | 包含建筑物、植被、地形的地表高度 |
| 3D 障碍物模型（可选） | OBJ/CityGML/glTF | 精细建筑模型（用于射线追踪精度提升） |
| UAV 视角位姿 | JSON | `x/y/z_m`、`heading_deg`、`gimbal_pitch_deg`、`gimbal_roll_deg`、`sensor_fov_h_deg`、`sensor_fov_v_deg`、`max_visual_range_m` |
| 飞行路径 | JSON | 来自算法 9 的 waypoint 序列（连续航点插值到 0.2s 步长） |
| 关注目标区域 (AOI) | GeoJSON | 需要重点分析可视性的目标区域 |

**输出 (Output)：**

| 结果 | 类型 | 说明 |
|------|------|------|
| `ViewshedResult` | JSON | 每个视角点的 `visible_area`（可见区 GeoJSON polygon）、`occluded_area`（遮挡区 polygon）、`occlusion_ratio`；累积可视域 `cumulative_visible_footprint` |
| `OcclusionRiskReport` | JSON | 盲区统计：`total_blind_duration_s`、`max_blind_span_s`、盲区面积占比；关键遮挡物列表：`obstacle_id`、`position`、`height`、`occlusion_contribution` |
| `ViewshedVisualization` | GeoTIFF/PNG | 绿色=可见区，红色=遮挡区，叠加在 DSM 阴影图上 |
| `SensorOptimizationSuggestion` | JSON | 传感器部署优化建议：加高飞行高度、调整视角、增加中继 UAV |

### 3. 依赖技术栈

| 层级 | 技术 |
|------|------|
| Viewshed 计算 | GDAL `gdal_viewshed` / GRASS GIS `r.viewshed` / 自定义射线追踪 |
| 射线追踪 | Embree / OptiX（GPU 加速，后续） |
| 可视域累积 | 多帧可视域叠加 + 时间衰减权重 |
| 空间分析 | GeoPandas、Shapely、Rasterio |
| 运行环境 | Python 3.10+、GDAL、NumPy |

### 4. 算法流程与模块联系

**步骤一（预处理—DSM 加载与航线采样）：**
- DSM 加载 → 重采样到统一分辨率 → 存储为 NumPy 数组
- 飞行路径插值：waypoints → 0.2s 步长密集采样（覆盖约 3m @ 15m/s）
- 各采样点根据 UAV heading + gimbal 角度 → 计算视线起始点和方向向量
- AOI 栅格化：目标区域 mask 用于重点分析

**步骤二（核心计算—Viewshed 分析）：**
- 每个视角点执行 viewshed 计算：
  - 基于 DSM 的栅格 viewshed：`gdal_viewshed` 以视角点为观测点，计算到 DSM 各栅格是否可视
  - 可见条件：视线路径上所有栅格的 elevation 低于视线高度
  - 输出二值栅格（1=visible, 0=occluded）
- 累积可视域：对所有视角点的 viewshed 做逻辑 OR 叠加 → cumulative visible footprint
- 盲区识别：`occluded_area = total_AOI - cumulative_visible`
- 遮挡物贡献度：每个障碍物造成多少像素的遮挡 → 排序识别关键遮挡物

**步骤三（后处理—报告与优化建议）：**
- 可见区/遮挡区可视化叠加（绿色/红色）
- 盲区统计报告：最长持续遮挡时间、最大盲区面积
- 传感器优化建议：
  - 若盲区 > 30% 且遮挡源为建筑物 → 建议抬高飞行高度
  - 若盲区在边缘 → 建议扩展飞行路径
  - 若遮挡来自单一高物 → 建议该区域加中继 UAV

**模块间数据流向：**

```
[DSM + 3D 障碍物 + UAV 视角位姿 + 飞行路径 + AOI]
  → [航线插值密集采样]
    → [逐点 Viewshed 计算 (射线追踪)]
      → [累积可视域叠加]
        → [盲区识别 + 遮挡物贡献度排序]
          → [Viewshed 可视化 (绿/红)]
          → [遮挡风险报告 + 传感器优化建议]
  → 上游依赖：算法9(航线规划, 飞行路径 waypoints) / 算法7(覆盖分析, DSM 级联)
  → 下游消费者：算法9(航线规划, 盲区避让优化) / 算法6(碰撞风险, 遮挡物作为碰撞候选)
```

---

## 算法九：航线规划与导航分析模型

| 字段 | 内容 |
|------|------|
| **算法名称** | 航线规划与导航分析模型 |
| **负责人** | 算法工程师 |
| **更新时间** | 2026-05-14 |
| **对应代码** | `routing.py` |

### 1. 一句话简介

结合城市三维模型和空域信息，通过智能化算法模拟生成最优航线，综合考虑飞行安全、效率及合规性因素，评估飞行活动合规性与安全性。

### 2. 输入与输出

**输入 (Input)：**

| 参数/数据 | 类型 | 说明 |
|-----------|------|------|
| `cells.csv` | CSV (Pydantic: `GridCell`) | `grid_id`、`height_layer`、`center_x/y/z_m`、`congestion_score`、`em_interference`、`risk_score`、`weather_wind`、`weather_visibility`、`no_fly_flag` |
| `edges.csv` | CSV (Pydantic: `GridEdge`) | `source/target_grid_id`、`source/target_height_layer`、`weight`、`edge_type`、`directed` |
| `RoutePlanRequest` | JSON (Pydantic) | `start_grid_id`、`start_height_layer`、`end_grid_id`、`end_height_layer`、`max_nodes`（默认 10000） |
| 城市 3D 模型（新增） | CityGML / 3D Tiles | 建筑高度、禁飞区/限制区边界、特殊走廊 |
| 空域管制规则（新增） | JSON/YAML | 高度上限、时间限制、廊道宽度、飞行规则（VFR/IFR） |

**输出 (Output)：**

| 结果 | 类型 | 说明 |
|------|------|------|
| `RoutePlan` | JSON | `found`（bool）、`route`（node_id 序列）、`total_cost`、`distance_m`、`risk_cost`、`message` |
| `ComplianceReport`（新增） | JSON | 路径是否经过禁飞区/限制区、是否满足高度限制、是否需要空域申报 |
| `RouteDataFrame` | CSV/JSON | `seq`、`grid_id`、`height_layer`、`node_id` 展开表 |
| 3D Route Visualization（新增） | Cesium/Deck.gl | 路径线叠加在城市 3D 模型上（合规段绿色、风险段黄色、禁入段红色） |

### 3. 依赖技术栈

| 层级 | 技术 |
|------|------|
| 路径规划 | 3D 网格 A*（P0 基线）；后续可升级 D* Lite（动态重规划）、RRT*/PRM（连续空间） |
| 代价函数 | 多因子加权：距离×1.0 + 高度变化×0.4 + 拥堵×120 + EM×60 + 风险×300 |
| 启发式 | 三维欧氏距离 |
| 可视化 | Cesium / deck.gl / Matplotlib 3D |
| 运行环境 | Python 3.10+、heapq、math |

### 4. 算法流程与模块联系

**步骤一（预处理—网格图构建）：**
- `_index_cells()`：cells → dict[node_id → cell attributes]，含坐标、风险、拥堵等
- `_build_adjacency()`：edges → 邻接表 dict[node_id → [(neighbor, weight)]]
  - 无向边自动生成反向链接
- 检查 start/goal 是否在 cells 中
- 从城市 3D 模型提取禁飞区 mask → 标记对应 grid 的 no_fly_flag=1

**步骤二（核心计算—三维 A* 搜索）：**
- Open 堆初始化为 [(0.0, start)]，came_from / g_score / distance_score / risk_score 字典
- 单步代价：`g_next = g[current] + distance·weight·w_dist + |Δz|×0.4 + node_risk[next]`
  - `node_risk = 120×congestion + 60×em + 300×(risk + visibility_penalty + wind_penalty) + no_fly_penalty`
  - `no_fly_penalty`：若 `no_fly_flag==1` → `1,000,000`（回退到其他路径）
- 启发式：`h = w_dist × Euclidean(next, goal)`
- 当 `current == goal` 时回溯路径
- 搜索上限 `max_nodes = 10,000`，超限返回 not found

**步骤三（后处理—合规评估与路径输出）：**
- 路径合法性评估：检查是否通过禁飞区/限制区
- 合规性报告：高度限制 check、廊道 check
- 路径平滑（后续）：B-spline / Minimum Snap 优化轨迹平滑度
- 3D 可视化路径叠加

**模块间数据流向：**

```
[cells + edges + start/goal + 城市3D模型 + 管制规则]
  → [_index_cells + _build_adjacency 构建网格图]
    → [3D A* f(n)=g(n)+h(n) 搜索]
      → [回溯路径 + 合规检查]
        → [RoutePlan JSON + ComplianceReport + 3D Visualization]
  → 上游依赖：算法1(流量预测, congestion_score 作为节点风险) / 算法3(电磁, em_interference 作为节点风险)
  → 下游消费者：算法4(资源分配, A* sub-call) / 算法6(碰撞风险, planned_trajectory) / 算法7(覆盖, waypoints) / 算法8(可视域, 飞行路径)
```

---

## 算法十：低空飞行安全评估分析模型

| 字段 | 内容 |
|------|------|
| **算法名称** | 低空飞行安全评估分析模型 |
| **负责人** | 算法工程师 |
| **更新时间** | 2026-05-14 |
| **对应代码** | `safety.py` |

### 1. 一句话简介

集成气象、地理、飞行器状态等多源数据，通过 CPA/TCPA 冲突检测与网格风险矩阵测算潜在安全隐患；使用机器学习模型分析飞行环境并识别风险事件，自动生成风险提示与风险报告，支持风险点定位标红。

### 2. 输入与输出

**输入 (Input)：**

| 参数/数据 | 类型 | 说明 |
|-----------|------|------|
| `cells.csv` | CSV (Pydantic: `GridCell`) | `grid_id`、`height_layer`、`weather_wind`、`weather_visibility`、`population_density`、`congestion_score`、`em_interference`、`no_fly_flag`、`risk_score` |
| `uavs.csv` | CSV (Pydantic: `UavState`) | `uav_id`、`x/y/z_m`、`vx/vy/vz_mps`、`grid_id`、`height_layer`、`battery_pct` |
| 实时气象数据（新增） | API JSON | 风速/风向、能见度、降水强度、闪电预警（来自气象局 API 或本地传感器） |
| 地理敏感区数据（新增） | GeoJSON | 人口密集区、关键基础设施、学校/医院等人口敏感区 |
| 风险识别模型（新增） | pickle / ONNX | 训练好的 ML 风险分类模型（Random Forest / XGBoost） |

**输出 (Output)：**

| 结果 | 类型 | 说明 |
|------|------|------|
| `SafetyAssessment` | JSON | `overall_risk_score`（0~1）、`risk_level`（normal/attention/high/severe）、`recommended_action`、`conflict_count`、`unsafe_grid_count` |
| `ConflictPairs` | JSON | 每对冲突 UAV：`ownship_id`、`intruder_id`、`tcpa_s`、`horizontal_cpa_m`、`vertical_cpa_m`、`conflict_risk`、`warning_level`、`recommended_action` |
| `GridRiskTable` | JSON | 每个网格：`grid_id`、`height_layer`、`grid_risk_score`、`warning_level`、`risk_level`、`recommended_action` |
| `RiskReport`（新增） | Markdown/PDF | 自动生成的风险分析报告：整体风险评分、高风险清单（位置标红）、风险趋势、处置建议 |
| `RiskVisualization`（新增） | GeoJSON/PNG | 风险点位置标红叠加在地图上 |

**风险等级与处置：**

| risk | risk_level | recommended_action |
|------|-----------|-------------------|
| ≥ 0.85 | severe | 拒绝或改航 |
| ≥ 0.65 | high | 人工审查或延迟 |
| ≥ 0.35 | attention | 监控 |
| < 0.35 | normal | 放行 |

### 3. 依赖技术栈

| 层级 | 技术 |
|------|------|
| 冲突检测 | CPA（最近接近点）/ TCPA（最近接近时间） |
| 风险矩阵 | 加权多因子网格风险评估（6 因子 + no_fly） |
| ML 模型 | scikit-learn（Random Forest / XGBoost）用于气象+地理风险预测 |
| 报告生成 | Jinja2 模板 → Markdown → PDF（WeasyPrint/Pandoc） |
| 运行环境 | Python 3.10+、NumPy、Pandas、scikit-learn |

### 4. 算法流程与模块联系

**步骤一（预处理—多源数据融合）：**
- CSV 数据读取：cells（空域状态） + uavs（飞行器状态）
- 气象数据接入：从气象局 API 获取实时 wind/visibility/precipitation → 更新 cell 的 weather_wind/weather_visibility
- 地理敏感区数据：人口密度映射到 grid、关键基础设施 proximity 计算
- ML 特征构建：融合气象×地理×空域×UAV 的多维特征向量

**步骤二（核心计算—冲突检测与风险矩阵）：**
- **CPA/TCPA 冲突检测**（`detect_conflicts`）：
  - 所有 UAV 对遍历：`r = p_own - p_intru`、`v = v_own - v_intru`
  - TCPA：`-(r·v)/||v||²`，clip 到 [0, lookahead_s]
  - CPA：`|r + v×TCPA|`（分解为水平/垂直分量）
  - 冲突风险：`0.45×h_ratio + 0.35×v_ratio + 0.20×time_ratio`
  
- **网格风险矩阵**（`grid_risk`）：
  - 6 因子加权：`0.20×(wind + visibility + population) + 0.15×(congestion + em) + 0.10×inherited_risk`
  - `no_fly_flag==1` → 直接设为 1.0（最高风险）
  
- **ML 风险预测**（新增）：
  - 训练模型输入：气象 + 地理 + 历史事故数据
  - 输出：每个 grid 的 risk_ml 预测值（0~1）
  - `grid_risk = max(no_fly, 0.7×grid_risk_matrix + 0.3×risk_ml)`

- **整体风险**：`overall = 0.55×max(grid_risk) + 0.45×max(conflict_risk)`

**步骤三（后处理—报告生成与风险标红）：**
- 风险分类：overall ≥ 0.85 → severe / ≥ 0.65 → high / ≥ 0.35 → attention / else → normal
- 风险报告自动生成（Jinja2 模板）：
  - 整体评分 + 等级
  - 高风险点列表（附带位置坐标 → 地图标红）
  - 冲突对详细清单
  - 脆弱网格统计
  - 气象/地理风险因子趋势
- 风险可视化：高风险网格红色填充 → GeoJSON 输出 → 前端叠加

**模块间数据流向：**

```
[cells + uavs + 气象API + 地理敏感区 + ML模型]
  → [多源数据融合 + 特征构建]
    → [CPA/TCPA 冲突检测 (pairwise)] → conflict_risk
    → [网格风险矩阵 (6因子)] + [ML 预测] → grid_risk
      → overall_risk = 0.55×grid + 0.45×conflict
        → [风险等级分类 + 报告生成]
          → [RiskReport (Markdown/PDF) + RiskVisualization (GeoJSON 标红)]
  → 上游依赖：算法1(流量预测, congestion_score + grid occupancy) / 算法3(电磁, em_interference) / 算法5(监控, 实时UAV状态) / 算法6(碰撞风险, collision_risk因子) / 算法9(航线规划, 计划路径的risk评估)
  → 最终输出：提供给空域管理者、UAS 运营商、监管机构
```

---

## 算法间全局数据依赖关系

### 数据流向总图

```md
                              ┌────────────────────────┐
                              │  统一数据总线            │
                              │  cell_key = grid_id    │
                              │  + height_layer        │
                              │  + time_slot           │
                              └────────┬───────────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼              ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
       │ ①流量预测 │  │ ②地形跟随  │  │ ③电磁分析 │  │ ④资源分配 │   │ ⑤飞行监控 │   |⑥碰撞风险  │
       │ Ridge/   │  │ PID/     │  │ FSPL/    │  │ Greedy/   │  │ Kalman/  │  │ OctoMap/│
       │ SGD      │  │ 避障     │  │ SINR     │  │ A*        │  │ 数字孪生  │  │ FCL      │
       └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
            │              │            │               │              │              │
            │ congestion   │ avoidance  │ em_intf       │ flight       │ realtime     │ collision
            │ _score       │ _cmd       │ _level        │ _plan        │ _state       │ _risk
            ▼              │               ▼               │              ▼              │
       ┌──────────┐        │          ┌──────────┐        │         ┌──────────┐         │
       │ ⑦覆盖分析 │        │          │ ⑧可视域   │        │          │ ⑨航线规划 │         │
       │ coverage │        │          │ viewshed │        │         │ A* 3D    │ ◄───────┘
       │ heatmap  │        │          │ analysis │        │         │ route    │
       └────┬─────┘        │          └────┬─────┘        │         └────┬─────┘
            │              │               │               │              │
            │ coverage_    │               │ occlusion_    │              │ waypoints
            │ report       │               │ report        │              │ / route
            └──────┬───────┴───────────────┴───────┬───────┴──────────────┘
                   │                               │
                   ▼                               ▼
            ┌─────────────────────────────────────────────┐
            │          ⑩ 低空飞行安全评估分析模型             │
            │  CPA/TCPA + 网格风险 + ML 预测 + 风险报告      │
            └─────────────────────────────────────────────┘
```

### 关键依赖链

| # | 依赖链 | 说明 |
|---|--------|------|
| 1→4 | 流量预测 → congestion_score → 资源分配 A* 节点代价 | 拥堵分数直接影响路径选择 |
| 1→9 | 流量预测 → congestion_score → 航线规划 node_risk (权重 120) | 拥堵是 A* 节点风险最大权重项 |
| 1→10 | 流量预测 → congestion_score → 安全评估 grid_risk | 拥堵占网格风险矩阵 15% 权重 |
| 3→1 | 电磁分析 → em_interference → 流量预测外生特征 | 电磁干扰作为流量预测的输入特征 |
| 3→9 | 电磁分析 → em_interference → 航线规划 node_risk (权重 60) | 电磁干扰影响路径安全评估 |
| 3→10 | 电磁分析 → em_interference → 安全评估 grid_risk (15% 权重) | 电磁干扰作为网格风险因子 |
| 9→4 | 航线规划 → A* 子调用 → 资源分配 | 资源分配内部调用 A* 计算每对 UAV-Task 路径 |
| 9→6 | 航线规划 → planned_trajectory → 碰撞风险评估 | 计划轨迹作为碰撞检测的基线 |
| 9→7 | 航线规划 → waypoints → 覆盖分析 | 航点作为覆盖模拟的输入 |
| 9→8 | 航线规划 → waypoints → 可视域分析 | 航点作为 viewshed 计算的视角点 |
| 5→10 | 飞行监控 → 实时 UAV 状态 → 安全评估冲突检测 | 实时状态是 CPA/TCPA 的直接输入 |
| 6→10 | 碰撞风险 → collision_risk → 安全评估 | 障碍物碰撞风险作为安全评估的附加因子 |
| 2→6 | 地形回避 → 避障指令 → 碰撞风险再评估 | 避障指令有效性通过碰撞模型验证 |
| 7→9 | 覆盖分析 → 最优航点反馈 → 航线规划优化 | 覆盖率不足区域触发航线重新规划 |
| 8→9 | 可视域分析 → 盲区警告 → 航线规划避开盲区 | 盲区大的区域建议绕飞 |

---

## 后续升级路线

| 算法 | P0 基线 | 后续可替换/增强 |
|------|---------|----------------|
| ① 流量预测 | Ridge + Historical + SGD | Graph WaveNet / AGCRN / PDFormer / DCRNN / STGCN；强化学习（RL）在线策略优化 |
| ② 地形回避 | 级联 PID + 几何避障 | 模型预测控制（MPC）/ LQR / 强化学习（RL）自适应控制 |
| ③ 电磁分析 | FSPL + 链路预算 + SINR | Sionna RT 射线追踪 / QuaDRiGa 信道建模 / ns-3 网络仿真 |
| ④ 资源分配 | 约束贪心 | OR-Tools CP-SAT / VRP / pymoo NSGA-II 多目标优化 |
| ⑤ 飞行监控 | 卡尔曼滤波 + 规则引擎 | LSTM Autoencoder 异常检测 / 深度强化学习辅助决策 |
| ⑥ 碰撞风险 | OctoMap + 几何碰撞 | FCL（灵活碰撞库）/ Bullet Physics 实时仿真 |
| ⑦ 覆盖分析 | 栅格 viewshed + 几何计算 | 多目标进化算法（MOEA）/ 自适应航线优化 |
| ⑧ 可视域 | GDAL viewshed | Embree/OptiX GPU 加速射线追踪 / 概率可视域 |
| ⑨ 航线规划 | 三维网格 A* | D* Lite（动态重规划）/ RRT*（连续空间）/ CBS/ECBS（多机冲突消解） |
| ⑩ 安全评估 | CPA/TCPA + 网格风险矩阵 | NASA DAIDALUS（DAA）/ ICAROUS（地理围栏）/ WellClear 标准 / XGBoost 风险预测 |
