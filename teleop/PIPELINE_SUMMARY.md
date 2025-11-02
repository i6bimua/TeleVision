# TeleVision 全景手部遥操作 Pipeline 详细说明

本文档详细说明了 `teleop_hand_panorama.py` 文件中从 VR 眼镜捕获手部姿态到 IsaacGym 模拟，再到全景图渲染返回的完整处理流程。

![image-20251102110415207](image-20251102110415207.png)

![65320E66-BCE5-43DC-9CBB-DC9E778208D6_1_201_a](65320E66-BCE5-43DC-9CBB-DC9E778208D6_1_201_a.jpeg)

## 目录

1. [系统架构概述](#系统架构概述)
2. [初始化阶段](#初始化阶段)
3. [主循环流程](#主循环流程)
4. [完整闭环流程图](#完整闭环流程图)
5. [关键技术点](#关键技术点)

---

## 系统架构概述

该系统实现了一个完整的 VR 到 IsaacGym 的遥操作系统，包含以下主要组件：

**进程架构**:
- **主进程**: `teleop_hand_panorama.py` - 主循环，包含 Sim 和 VuerTeleop
- **子进程**: `OpenTeleVision` - VR 通信进程，通过 `multiprocessing.Process` 启动

**组件**:
- **VuerTeleop**: 处理 VR 姿态数据，执行重定向
- **Sim**: IsaacGym 模拟器，管理虚拟环境
- **OpenTeleVision**: VR 通信模块，处理图像传输（独立进程）
- **VuerPreprocessor**: 姿态数据预处理，坐标系转换

### 数据流方向

```
                        [进程间边界]
VR眼镜 (WebXR) ←→ [Vuer/WebSocket] ←→ OpenTeleVision (子进程) ←─ [共享内存] ─→ 主进程
                                          ↓                             ↑
                                    事件处理器                      VuerTeleop
                                   (on_hand_move,                 + VuerPreprocessor
                                    on_cam_move)                           ↑
                                          ↓                            Sim
                                  姿态数据写入共享内存                全景图像生成
                                          ↓                             ↓
                                  通过WebSocket发送              写入共享内存
                                    VR姿态数据                          ↓
                                                                  读取全景图
                                                                       ↓
                                                                  返回子进程
                                                                  编码并通过
                                                                  WebSocket发送
```

---

## 初始化阶段

### 1. VuerTeleop 类初始化

**位置**: `teleop_hand_panorama.py:32-63`

#### 1.1 全景图分辨率设置

```python
self.resolution = (720, 1280)        # 原始分辨率
self.panorama_height = 720           # 全景图高度
self.panorama_width = 1440           # 全景图宽度 (2:1 宽高比)
self.img_shape = (720, 1440, 3)      # 图像形状
```

**处理说明**:
- 设置全景图尺寸为 720×1440（2:1 宽高比）
- 确保与后续相机系统输出匹配
- **注意**: 这是在**主进程**中执行的

#### 1.2 共享内存创建

```python
self.shm = shared_memory.SharedMemory(
    create=True, 
    size=np.prod(self.img_shape) * np.uint8().itemsize
)
self.img_array = np.ndarray(
    (self.img_shape[0], self.img_shape[1], 3), 
    dtype=np.uint8, 
    buffer=self.shm.buf
)
```

**处理说明**:
- 主进程创建共享内存缓冲区用于图像数据
- 大小: `720 × 1440 × 3 × 1 byte = 3,110,400 bytes`
- **多进程共享**: 主进程和子进程通过共享内存名访问同一块物理内存
- **零拷贝**: 两个进程直接访问相同的内存页面，无需数据复制
- **底层实现**: 使用操作系统的共享内存机制（POSIX shm_open/SHM）

#### 1.3 OpenTeleVision 初始化

```python
self.tv = OpenTeleVision(
    self.img_shape, 
    self.shm.name, 
    image_queue, 
    toggle_streaming, 
    ngrok=True, 
    panorama_mode=True
)
```

**处理说明**:
- 启动 VR 连接服务器（使用 ngrok 或 SSL）
- 创建 WebSocket 通信通道（Vuer 框架）
- **创建独立子进程**: `Process(target=self.run)` 运行 VR 通信服务
- 在子进程中通过 `shared_memory.SharedMemory(name=shm_name)` 连接到同一共享内存
- 启动异步事件处理（WebSocket 事件处理器）

#### 1.4 重定向配置加载

```python
with Path(config_file_path).open('r') as f:
    cfg = yaml.safe_load(f)
left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
self.left_retargeting = left_retargeting_config.build()
self.right_retargeting = right_retargeting_config.build()
```

**处理说明**:
- 从 `inspire_hand.yml` 加载左右手重定向配置
- 构建重定向器，用于将手部关键点映射到机器人关节角度

---

### 2. Sim 类初始化

**位置**: `teleop_hand_panorama.py:80-225`

#### 2.1 IsaacGym 模拟器初始化

```python
self.gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.dt = 1 / 60                          # 60Hz 时间步
sim_params.substeps = 2                          # 子步数
sim_params.up_axis = gymapi.UP_AXIS_Z           # Z轴向上
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81) # 重力
sim_params.physx.use_gpu = True                  # GPU加速
self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
```

**处理说明**:
- 创建 PhysX 物理引擎实例
- 配置为 60Hz 模拟频率
- 启用 GPU 加速

#### 2.2 场景创建

```python
# 地面
self.gym.add_ground(self.sim, plane_params)

# 桌子 (0.8×0.8×0.1)
table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)
pose.p = gymapi.Vec3(0, 0, 1.2)
table_handle = self.gym.create_actor(self.env, table_asset, pose, 'table', 0)

# 立方体 (0.05×0.05×0.05)
cube_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)
pose.p = gymapi.Vec3(0, 0, 1.25)
cube_handle = self.gym.create_actor(self.env, cube_asset, pose, 'cube', 0)
```

#### 2.3 机器人手部加载

```python
left_asset = self.gym.load_asset(self.sim, asset_root, "inspire_hand/inspire_hand_left.urdf", asset_options)
right_asset = self.gym.load_asset(self.sim, asset_root, "inspire_hand/inspire_hand_right.urdf", asset_options)

# 左手
pose.p = gymapi.Vec3(-0.6, 0, 1.6)
self.left_handle = self.gym.create_actor(self.env, left_asset, pose, 'left', 1, 1)

# 右手
pose.p = gymapi.Vec3(-0.6, 0, 1.6)
self.right_handle = self.gym.create_actor(self.env, right_asset, pose, 'right', 1, 1)
```

**处理说明**:
- 加载 Inspire Hand URDF 模型
- 初始位置设置为 `(-0.6, 0, 1.6)`
- 设置 DOF 驱动模式为位置控制

#### 2.4 根状态张量获取

```python
self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
self.left_root_states = self.root_states[left_idx]
self.right_root_states = self.root_states[right_idx]
```

**处理说明**:
- 获取所有 Actor 的根状态张量（位置、姿态、速度等）
- 用于后续高效更新手部位置

#### 2.5 全景相机系统设置

**位置**: `teleop_hand_panorama.py:226-290`

```python
def _setup_panorama_cameras(self):
    self.face_resolution = 720          # 每个面的分辨率
    self.panorama_height = 720         # 全景图高度
    self.panorama_width = 1440         # 全景图宽度 (2:1)
    
    camera_props = gymapi.CameraProperties()
    camera_props.width = 720
    camera_props.height = 720
    camera_props.horizontal_fov = 90.0  # 90度FOV对应立方体一面
```

**六个相机定义**:

| 相机 | Lookat Offset | 朝向 |
|------|---------------|------|
| F (Front) | `[1, 0, 0]` | +X 轴 |
| R (Right) | `[0, -1, 0]` | +Y 轴 |
| B (Back)  | `[-1, 0, 0]` | -X 轴 |
| L (Left)  | `[0, 1, 0]`  | -Y 轴 |
| U (Up)    | `[0, 0, 1]`  | +Z 轴 |
| D (Down)  | `[0, 0, -1]` | -Z 轴 |

**处理说明**:
- 所有相机位于同一位置 `self.cam_pos = [-0.6, 0, 1.6]`
- 每个相机朝向不同方向，覆盖 360° 视野
- 90° FOV 确保无缝拼接

---

## 主循环流程

主循环在 `teleop_hand_panorama.py:424-436` 执行，每帧包含以下步骤：

### 时序关系总览

系统包含三个独立的时序子系统：

| 子系统 | 更新频率 | 驱动方式 | 限制因素 |
|--------|---------|---------|---------|
| **VR姿态数据** | ~90-120Hz | WebXR 硬件刷新 | VR硬件刷新率 |
| **主循环/IsaacGym** | 60Hz | `sync_frame_time()` | IsaacGym dt=1/60 |
| **全景图传输** | 60Hz | `asyncio.sleep(0.016)` | 主循环输出 |

**关键理解**:
- VR 姿态更新：由 WebXR 硬件驱动，可能比 60Hz 更快
- 主循环读取：在 60Hz 下读取最新共享内存
- 全景图输出：固定在 60Hz，与主循环同步

**循环频率**:
- **IsaacGym**: 60Hz (`sim_params.dt = 1 / 60`)
- **目标**: ~60fps 闭环
- **关键函数**: `self.gym.sync_frame_time(self.sim)` (第339行) - **这是实际决定循环速度的函数**
- **工作方式**: 
  - 无限循环 `while True:` 会尽可能快地执行
  - 但 `sync_frame_time()` 会让主线程**等待**，直到 IsaacGym 的内部时钟到达下一个物理步长时刻
  - 因此虽然循环很忙，但实际帧率被限制到 60Hz
- **如果没有 sync_frame_time**: 循环会以 CPU/GPU 能处理的最快速度运行（可能几百 fps），但 IsaacGym 物理会失同步

**重要区别**:
- **VR姿态更新**: 异步事件驱动，无固定帧率（受硬件限制，通常 90-120Hz）
- **主循环/IsaacGym**: 固定在 60Hz（由 `sync_frame_time` 控制）
- **全景图传输**: 固定在 60Hz（由主循环驱动）

### 步骤 1: 从 VR 眼镜捕获姿态数据

**位置**: `TeleVision_panorama.py`

#### 1.0 WebSocket 通信架构

**Vuer 框架**:
- `OpenTeleVision` 使用 **Vuer 框架** (`from vuer import Vuer`) 建立与 VR 客户端的 WebSocket 连接
- Vuer 是一个基于 WebSocket 的 VR/WebXR 通信框架，支持双向数据传输
- **端口**: 8012 (HTTPS/WSS)
- **协议**: WebSocket (WSS) over HTTPS

**初始化代码**:
```python
# TeleVision_panorama.py:28-31
if ngrok:
    self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
else:
    self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

# 注册事件处理器
self.app.add_handler("HAND_MOVE")(self.on_hand_move)
self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
```

**工作原理**:
1. VR 客户端通过 WebXR API 连接到服务器
2. Vuer 自动处理 WebSocket 连接和事件分发
3. 姿态数据通过预定义的事件类型 (`HAND_MOVE`, `CAMERA_MOVE`) 异步传输
4. 服务器端事件处理器 (`on_hand_move`, `on_cam_move`) 接收数据并写入共享内存

**更新时机**:
- **事件驱动**: 当 VR 用户移动头部/手部时，WebXR 触发事件
- **异步更新**: 事件处理器立即将数据写入共享内存（不受主循环限制）
- **主循环轮询**: 主循环每次调用 `teleoperator.step()` 时读取最新数据

**关键**: VR 姿态矩阵更新**没有固定帧率限制**！
- 由用户的移动速度决定更新频率
- 理论上可以在几毫秒内更新多次
- 实际受限于 WebXR 硬件刷新率（通常 90Hz 或 120Hz）

#### 1.1 头部姿态接收

```python
async def on_cam_move(self, event, session, fps=60):
    self.head_matrix_shared[:] = event.value["camera"]["matrix"]
    self.aspect_shared.value = event.value['camera']['aspect']
```

**数据格式**:
- `head_matrix`: 4×4 变换矩阵 (16 个 float64)
- 存储格式: 列优先 (Fortran order)

**访问接口**:
```python
@property
def head_matrix(self):
    return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")
```

#### 1.2 手部姿态接收

```python
async def on_hand_move(self, event, session, fps=60):
    self.left_hand_shared[:] = event.value["leftHand"]      # 4×4 矩阵
    self.right_hand_shared[:] = event.value["rightHand"]    # 4×4 矩阵
    self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()  # 25×3=75
    self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten() # 25×3=75
```

**数据格式**:
- `left_hand` / `right_hand`: 手腕 4×4 变换矩阵 (16 float64)
- `left_landmarks` / `right_landmarks`: 25 个关键点 × 3 坐标 = 75 float64

**关键点索引**:
- `tip_indices = [4, 9, 14, 19, 24]` - 五个指尖（拇指、食指、中指、无名指、小指）

---

### 步骤 2: 姿态数据预处理

**位置**: `Preprocessor.py:23-52`

#### 2.1 矩阵平滑更新

```python
self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, tv.left_hand.copy())
```

**处理说明**:
- `mat_update()` 实际只是**无效矩阵检查**，检测到奇异矩阵时使用前一帧
- 非平滑；直接采用最新 WebSocket 数据
- `mat_update` 实现：`if det(mat) == 0: return prev_mat else: return mat`

#### 2.2 坐标系转换 (Y-UP → Z-UP)

```python
# 坐标系转换矩阵
grd_yup2grd_zup = np.array([
    [0, 0, -1, 0],   # Y → -Z
    [-1, 0, 0, 0],   # X → -X  
    [0, 1, 0, 0],    # Z → Y
    [0, 0, 0, 1]
])

head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
right_wrist_mat = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
left_wrist_mat = grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
```

**处理说明**:
- VR 系统使用 Y-UP 坐标系
- IsaacGym 使用 Z-UP 坐标系
- 通过相似变换保持旋转性质

#### 2.3 手部坐标系转换

```python
# 手部坐标系对齐矩阵
hand2inspire = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])

rel_left_wrist_mat = left_wrist_mat @ hand2inspire
rel_right_wrist_mat = right_wrist_mat @ hand2inspire
```

**处理说明**:
- 将 VR 手部坐标系对齐到 Inspire Hand 坐标系
- 确保手指方向正确

#### 2.4 相对头部位置计算

```python
rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]
rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]
```

**处理说明**:
- 计算手腕相对于头部的位置
- 使手部位置独立于头部移动

#### 2.5 手指关键点处理

```python
# 转换为齐次坐标
left_fingers = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, 25))])
right_fingers = np.concatenate([tv.right_landmarks.copy().T, np.ones((1, 25))])

# 坐标系转换
left_fingers = grd_yup2grd_zup @ left_fingers
right_fingers = grd_yup2grd_zup @ right_fingers

# 转换为相对于手腕的坐标系
rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers

# 转换为 Inspire Hand 坐标系
rel_left_fingers = (hand2inspire.T @ rel_left_fingers)[0:3, :].T
rel_right_fingers = (hand2inspire.T @ rel_right_fingers)[0:3, :].T
```

**处理说明**:
- 将 25 个关键点转换为齐次坐标 (4×25)
- 转换到 Z-UP 坐标系
- 计算相对于手腕的局部坐标
- 对齐到 Inspire Hand 坐标系

---

### 步骤 3: 构建机器人控制指令

**位置**: `teleop_hand_panorama.py:65-77`

#### 3.1 提取头部旋转矩阵

```python
head_rmat = head_mat[:3, :3]  # 提取 3×3 旋转矩阵
```

**用途**: 后续用于调整相机朝向（虽然全景模式下相机固定）

#### 3.2 构建手部位置姿态

```python
# 位置 = 相对位置 + 固定偏移
left_pose_position = left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6])
right_pose_position = right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6])

# 姿态 = 旋转矩阵转换为四元数
left_pose_quat = rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]
right_pose_quat = rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]

# 拼接: [x, y, z, qx, qy, qz, qw]
left_pose = np.concatenate([left_pose_position, left_pose_quat])
right_pose = np.concatenate([right_pose_position, right_pose_quat])
```

**处理说明**:
- `[-0.6, 0, 1.6]` 是手部在模拟器中的基准位置
- 四元数顺序从 `[w, x, y, z]` 转换为 `[x, y, z, w]`（IsaacGym 格式）

#### 3.3 手指关节重定向

```python
# 提取五个指尖关键点
left_tips = left_hand_mat[tip_indices]  # shape: (5, 3)
right_tips = right_hand_mat[tip_indices]  # shape: (5, 3)

# 重定向到关节角度
left_qpos = self.left_retargeting.retarget(left_tips)
right_qpos = self.right_retargeting.retarget(right_tips)

# 重排顺序 (特定关节映射)
left_qpos = left_qpos[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
right_qpos = right_qpos[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
```

**处理说明**:
- 使用 5 个指尖位置推断 12 个关节角度
- `retarget()` 执行逆运动学或查找表映射
- 重排确保关节顺序与 URDF 定义一致

---

### 步骤 4: 更新 IsaacGym 模拟器

**位置**: `teleop_hand_panorama.py:292-314`

#### 4.1 更新手部根状态（位置和姿态）

```python
self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)
self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
```

**根状态格式** (每只手 13 个元素):
- `[0:3]`: 位置 (x, y, z)
- `[3:7]`: 四元数姿态 (x, y, z, w)
- `[7:10]`: 线速度 (vx, vy, vz)
- `[10:13]`: 角速度 (wx, wy, wz)

**处理说明**:
- 只更新前 7 个元素（位置和姿态）
- 速度和角速度由物理引擎计算

#### 4.2 更新手指关节角度

```python
left_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
left_states['pos'] = left_qpos
self.gym.set_actor_dof_states(self.env, self.left_handle, left_states, gymapi.STATE_POS)

right_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
right_states['pos'] = right_qpos
self.gym.set_actor_dof_states(self.env, self.right_handle, right_states, gymapi.STATE_POS)
```

**处理说明**:
- DOF (Degrees of Freedom) 状态包含位置和速度
- 只设置位置 (`STATE_POS`)，速度由驱动器控制

#### 4.3 物理模拟执行

```python
self.gym.simulate(self.sim)                    # 执行物理步进
self.gym.fetch_results(self.sim, True)        # 获取模拟结果
self.gym.step_graphics(self.sim)               # 更新图形
self.gym.render_all_camera_sensors(self.sim)  # 渲染所有相机
self.gym.refresh_actor_root_state_tensor(self.sim)  # 刷新状态张量
```

**处理说明**:
- `simulate()`: 执行一个时间步的物理计算
- `fetch_results()`: 等待 GPU 计算完成
- `step_graphics()`: 更新可视化状态
- `render_all_camera_sensors()`: 渲染所有相机图像到纹理

#### 4.4 帧同步（关键！）

**位置**: `teleop_hand_panorama.py:338-339`

```python
self.gym.draw_viewer(self.viewer, self.sim, True)
self.gym.sync_frame_time(self.sim)  # ← 这就是决定循环速度的关键！
```

**重要**: `sync_frame_time()` **实际控制循环频率**
- 让主线程等待，直到 IsaacGym 内部时钟到达下一个 `dt` 时刻
- 确保循环按 `sim_params.dt = 1/60` 运行（60Hz）
- 无此调用时，循环会尽可能快地运行，导致物理失同步

---

### 步骤 5: 捕获和转换全景图像

**位置**: `teleop_hand_panorama.py:347-403`

#### 5.1 设置相机位置和朝向

```python
# 所有相机位于固定位置，朝向固定方向
self.gym.set_camera_location(
    self.cube_camera_handles['F'], 
    self.env,
    gymapi.Vec3(*(self.cam_pos)),                    # 位置: [-0.6, 0, 1.6]
    gymapi.Vec3(*(self.cam_pos + self.cam_lookat_offset_F))  # 朝向: [+X]
)
# ... 重复设置其他 5 个相机
```

**注意**: 全景模式下相机不跟随头部旋转，始终保持固定朝向

#### 5.2 捕获六面图像

```python
face_order = ['F', 'R', 'B', 'L', 'U', 'D']
cube_faces = []

for face_name in face_order:
    handle = self.cube_camera_handles[face_name]
    img = self.gym.get_camera_image(self.sim, self.env, handle, gymapi.IMAGE_COLOR)
    # 转换为RGB格式 (移除alpha通道)
    img = img.reshape(img.shape[0], -1, 4)[..., :3]
    cube_faces.append(img)
```

**图像格式**:
- 输入: RGBA, shape `(height, width * height, 4)`
- 输出: RGB, shape `(720, 720, 3)`

#### 5.3 镜像处理

```python
if face_name in ['R', 'B']:
    img = np.fliplr(img)  # 水平翻转
```

**处理说明**:
- Right 和 Back 面需要镜像以匹配标准立方体贴图格式
- 确保后续拼接无缝

#### 5.4 拼接为 Horizon 格式

```python
cube_horizon = np.hstack(cube_faces)  # shape: (720, 4320, 3)
```

**Horizon 布局** (从左到右):
```
[Front] [Right] [Back] [Left] [Up] [Down]
 720px   720px   720px   720px  720px  720px
```

#### 5.5 转换为等距圆柱投影全景图

```python
if PY360CONVERT_AVAILABLE:
    panorama = py360convert.c2e(
        cube_horizon,                  # 输入: horizon 格式立方体贴图
        self.panorama_height,          # 输出高度: 720
        self.panorama_width,           # 输出宽度: 1440
        mode='bilinear',                # 插值方式
        cube_format='horizon'          # 输入格式
    )
    panorama_uint8 = panorama.astype(np.uint8)  # float64 [0-255] → uint8
    return panorama_uint8
```

**处理说明**:
- `c2e()`: Cube to Equirectangular (立方体贴图转等距圆柱投影)
- 双线性插值确保平滑过渡
- 输出 720×1440 全景图 (2:1 宽高比)

**等距圆柱投影特点**:
- 水平方向: 360° 完整覆盖
- 垂直方向: 180° (上下各 90°)
- 顶部和底部有拉伸失真，但中间区域接近透视

---

### 步骤 6: 将图像传输到 VR 眼镜

**位置**: 主循环 `teleop_hand_panorama.py:424-427` + `TeleVision_panorama.py:125-201`

#### 6.1 写入共享内存

```python
# 主循环中
panorama_img = simulator.step(...)
np.copyto(teleoperator.img_array, panorama_img)
```

**处理说明**:
- `np.copyto()` 将全景图复制到共享内存缓冲区
- 用于主进程和 VR 进程间的零拷贝数据传输

#### 6.2 VR 进程读取和编码

```python
# TeleVision_panorama.py:main_panorama()
while True:
    display_image = self.img_array.copy()  # 从共享内存读取
    
    # 转换为PIL图像
    pil_img = Image.fromarray(display_image, 'RGB')
    
    # JPEG编码
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=85)
    img_bytes = buffer.getvalue()
    
    # Base64编码
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    img_data_url = f"data:image/jpeg;base64,{img_base64}"
    
    await asyncio.sleep(0.016)  # ~60fps
```

**处理说明**:
- JPEG 质量 85: 平衡质量和传输速度
- Base64 编码: 便于在 WebSocket 中传输
- Data URL 格式: `data:image/jpeg;base64,{...}`
- **帧率控制**: `await asyncio.sleep(0.016)` 确保 60Hz 图像更新

#### 6.3 通过 WebSocket 发送到 VR 客户端

```python
session.upsert @ Sphere(
    args=[1, 32, 32],                    # 半径1，32×32 细分
    materialType="standard",
    material={"map": img_data_url, "side": 1},  # 纹理贴图，单面渲染
    position=[0, 0, 0],
    rotation=[0, 0.5 * np.pi, 0],        # 绕Y轴旋转90°
    key="skyball"
)
```

**处理说明**:
- `session.upsert @ Sphere()` 通过 Vuer WebSocket 连接将更新发送到 VR 客户端
- `Sphere` 作为天空球 (Skybox) 显示全景图
- 用户位于球心，看到内侧纹理
- 旋转 90° 用于坐标系对齐 (Z-UP → Y-UP)
- `side: 1` 只渲染内侧，避免背面可见

#### 6.4 帧率控制

```python
await asyncio.sleep(0.016)  # ~60fps
```

**处理说明**:
- 目标 60fps，每帧 16.67ms
- 包括图像读取、编码、WebSocket 传输的总时间

---

## 完整闭环流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        VR眼镜 (Meta Quest)                       │
│  ┌──────────────┐          ┌──────────────┐                    │
│  │  头部追踪    │          │   手部追踪   │                    │
│  │  (6DOF)      │          │  (25关键点)  │                    │
│  └──────┬───────┘          └──────┬───────┘                    │
│         │                         │                             │
│         └──────────┬──────────────┘                             │
│                    │ WebSocket/WebRTC                            │
└────────────────────┼─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OpenTeleVision (主进程)                         │
│                                                                 │
│  [1] on_cam_move()                                             │
│      └─> head_matrix_shared[16] ← 头部4×4矩阵                  │
│                                                                 │
│  [2] on_hand_move()                                            │
│      ├─> left_hand_shared[16] ← 左手腕4×4矩阵                  │
│      ├─> right_hand_shared[16] ← 右手腕4×4矩阵                 │
│      ├─> left_landmarks_shared[75] ← 左手25关键点              │
│      └─> right_landmarks_shared[75] ← 右手25关键点            │
│                                                                 │
│  属性访问:                                                      │
│  ├─> tv.head_matrix (4×4)                                      │
│  ├─> tv.left_hand (4×4)                                        │
│  ├─> tv.right_hand (4×4)                                       │
│  ├─> tv.left_landmarks (25×3)                                  │
│  └─> tv.right_landmarks (25×3)                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              VuerPreprocessor.process()                  │
│                                                                 │
│  [3] 矩阵平滑更新                                               │
│      └─> mat_update(previous, current)                          │
│                                                                 │
│  [4] 坐标系转换 (Y-UP → Z-UP)                                  │
│      └─> grd_yup2grd_zup @ matrix @ inv(grd_yup2grd_zup)       │
│                                                                 │
│  [5] 手部坐标系对齐                                             │
│      └─> matrix @ hand2inspire                                  │
│                                                                 │
│  [6] 相对头部位置                                               │
│      └─> wrist_pos - head_pos                                  │
│                                                                 │
│  [7] 手指关键点处理                                             │
│      ├─> 齐次坐标转换                                           │
│      ├─> 坐标系转换                                             │
│      ├─> 相对手腕坐标                                           │
│      └─> Inspire Hand 坐标系对齐                               │
│                                                                 │
│  输出:                                                          │
│  ├─> head_mat (4×4)                                            │
│  ├─> left_wrist_mat (4×4)                                       │
│  ├─> right_wrist_mat (4×4)                                     │
│  ├─> left_hand_mat (25×3)                                       │
│  └─> right_hand_mat (25×3)                                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  VuerTeleop.step()                              │
│                                                                 │
│  [8] 提取头部旋转矩阵                                           │
│      └─> head_rmat = head_mat[:3, :3]                          │
│                                                                 │
│  [9] 构建位置姿态                                               │
│      ├─> position = wrist_pos + offset [-0.6, 0, 1.6]          │
│      └─> quaternion = matrix_to_quat(rotation)[x,y,z,w]         │
│                                                                 │
│  [10] 手指重定向                                                │
│       ├─> tips = hand_mat[tip_indices]  # 5个指尖             │
│       ├─> qpos = retargeting(tips)  # 12个关节角              │
│       └─> qpos = qpos[reorder]  # 重排顺序                     │
│                                                                 │
│  输出:                                                          │
│  ├─> head_rmat (3×3)                                           │
│  ├─> left_pose [x,y,z,qx,qy,qz,qw] (7,)                        │
│  ├─> right_pose [x,y,z,qx,qy,qz,qw] (7,)                       │
│  ├─> left_qpos (12,)                                           │
│  └─> right_qpos (12,)                                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Sim.step()                                   │
│                                                                 │
│  [11] 更新根状态                                                │
│       ├─> left_root_states[0:7] = left_pose                     │
│       ├─> right_root_states[0:7] = right_pose                   │
│       └─> set_actor_root_state_tensor()                         │
│                                                                 │
│  [12] 更新关节角度                                              │
│       ├─> left_states['pos'] = left_qpos                       │
│       ├─> right_states['pos'] = right_qpos                      │
│       └─> set_actor_dof_states()                                │
│                                                                 │
│  [13] 物理模拟                                                  │
│       ├─> simulate()          # 物理步进                        │
│       ├─> fetch_results()     # 等待GPU                        │
│       ├─> step_graphics()     # 更新图形                        │
│       └─> render_all_camera_sensors()  # 渲染相机               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│           _capture_and_convert_panorama()                      │
│                                                                 │
│  [14] 设置六个相机                                              │
│       ├─> F: 看向 +X                                           │
│       ├─> R: 看向 +Y                                           │
│       ├─> B: 看向 -X                                           │
│       ├─> L: 看向 -Y                                           │
│       ├─> U: 看向 +Z                                           │
│       └─> D: 看向 -Z                                           │
│                                                                 │
│  [15] 捕获六面图像                                              │
│       └─> get_camera_image() → (720, 720, 3) × 6                │
│                                                                 │
│  [16] 镜像处理                                                  │
│       └─> np.fliplr(R, B)                                      │
│                                                                 │
│  [17] 拼接Horizon格式                                           │
│       └─> np.hstack() → (720, 4320, 3)                          │
│                                                                 │
│  [18] 转换为全景图                                              │
│       └─> py360convert.c2e() → (720, 1440, 3)                   │
│                                                                 │
│  输出: panorama_uint8 (720, 1440, 3)                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│           主进程 → VR进程: 共享内存                              │
│                                                                 │
│  [19] np.copyto(teleoperator.img_array, panorama_img)          │
│       └─> 写入共享内存 (3,110,400 bytes)                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│            OpenTeleVision.main_panorama()                       │
│                                                                 │
│  [20] 读取共享内存                                              │
│       └─> display_image = self.img_array.copy()                │
│                                                                 │
│  [21] 图像编码                                                  │
│       ├─> PIL Image.fromarray()                                │
│       ├─> JPEG编码 (quality=85)                                 │
│       └─> Base64编码                                            │
│                                                                 │
│  [22] 创建Data URL                                              │
│       └─> "data:image/jpeg;base64,{...}"                       │
│                                                                 │
│  [23] 更新Sphere                                                │
│       └─> session.upsert @ Sphere(material=img_data_url)       │
│                                                                 │
│  [24] Vuer WebSocket传输                                        │
│       └─> 通过session.upsert发送到VR客户端                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        VR眼镜显示                               │
│                                                                 │
│  [25] 接收WebSocket消息                                         │
│  [26] 解析JSON/Sphere数据                                       │
│  [27] 加载纹理到GPU                                             │
│  [28] 渲染Skybox                                                │
│  [29] 用户看到全景图                                            │
│                                                                 │
│  ┌─────────────────────────────┐                                │
│  │     用户移动头部/手部       │                                │
│  │                             │                                │
│  │  → 触发新的姿态事件          │                                │
│  └─────────────────────────────┘                                │
│                                                                 │
│  └───────────────────────────────────────────────────────────┐  │
│   循环回到步骤 [1]，形成闭环                                    │  │
└───────────────────────────────────────────────────────────────┘  │
```

---

## 关键技术点

### 1. 双通道通信架构

系统使用两种通信机制协同工作：

#### 1.1 WebSocket (Vuer) - 跨网络通信

**VR ↔ 服务器**: 通过 Vuer 框架的 WebSocket 连接
- **VR → 服务器**: 姿态数据 (`HAND_MOVE`, `CAMERA_MOVE` 事件)
- **服务器 → VR**: 全景图数据 (`session.upsert @ Sphere`)
- **协议**: WSS over HTTPS (端口 8012)
- **优点**: 实时双向通信，跨网络支持

#### 1.2 共享内存 - 进程间通信 (IPC)

**实现**: `multiprocessing.shared_memory.SharedMemory`

**关键区别**: 共享内存**不是**基于 WebSocket
- WebSocket: 用于 **跨网络** 通信（VR 客户端 ↔ 服务器）
- 共享内存: 用于 **本地进程间** 通信（主进程 ↔ 子进程）

**工作流程**:
1. **主进程**: 创建共享内存 `SharedMemory(create=True)`，获得唯一名称
2. **子进程**: 通过名称连接 `SharedMemory(name=shm_name)` 到同一块物理内存
3. **数据传递**: 
   - 主进程写入: `np.copyto(teleoperator.img_array, panorama_img)`
   - 子进程读取: `display_image = self.img_array.copy()`

**用途**: 主进程 (IsaacGym) ↔ VR进程 (OpenTeleVision) 间的高效数据传递
- **主进程写入**: 渲染的全景图
- **VR进程读取**: 编码并通过 WebSocket 发送到 VR 客户端

**优势**:
- **零拷贝**: 两个进程直接访问同一块物理内存
- **高效**: 无需序列化/反序列化，无需网络开销
- **原子操作**: NumPy 数组操作通常是原子的
- **底层实现**: 使用操作系统共享内存机制（POSIX shm_open/SHM）

**数据大小**:
- 全景图: `720 × 1440 × 3 = 3,110,400 bytes ≈ 3 MB`

### 2. 坐标系转换

#### 2.1 Y-UP → Z-UP 转换

```python
grd_yup2grd_zup = np.array([
    [0,  0, -1,  0],   # Y (VR) → -Z (Gym)
    [-1,  0,  0,  0],   # X (VR) → -X (Gym)
    [0,  1,  0,  0],    # Z (VR) → Y (Gym)
    [0,  0,  0,  1]
])
```

**转换规则**:
- VR坐标系: X=右, Y=上, Z=前 (右手系)
- Gym坐标系: X=?, Y=?, Z=上 (右手系)

#### 2.2 手部坐标系对齐

```python
hand2inspire = np.array([
    [0, -1,  0,  0],   # 
    [0,  0, -1,  0],   # 
    [1,  0,  0,  0],   # 
    [0,  0,  0,  1]
])
```

**用途**: 将VR手部坐标系对齐到机器人手部URDF定义

### 3. 全景图生成算法

#### 3.1 立方体贴图布局

```
        +----+
        | Up |
+----+----+----+----+
|Left|Front|Right|Back|
+----+----+----+----+
        |Down|
        +----+
```

#### 3.2 Horizon格式

水平展开顺序: `[Front, Right, Back, Left, Up, Down]`

#### 3.3 等距圆柱投影

**数学原理**:
- 水平方向: `u = (θ + π) / (2π) * width`
- 垂直方向: `v = (φ + π/2) / π * height`

其中:
- `θ`: 方位角 (azimuth, -π 到 π)
- `φ`: 极角 (polar, -π/2 到 π/2)

### 4. 姿态平滑

**实现**: `mat_update()` (在 `motion_utils.py` 中)

**目的**: 减少噪声和抖动

**方法**: 可能使用指数移动平均或卡尔曼滤波

### 5. 手指重定向

**输入**: 5 个指尖的 3D 位置

**输出**: 12 个关节角度

**方法**:
- 可能使用逆运动学 (IK)
- 或查找表映射
- 或神经网络回归

### 6. 异步通信

**VR通信**: 使用 Vuer 框架的异步事件处理 + WebSocket

**优点**:
- 非阻塞: 主循环不被网络IO阻塞
- 并发: 多个事件可同时处理
- 实时性: 低延迟
- 双向通信: 姿态上行，全景图下行

### 7. 性能优化

#### 7.1 GPU加速
- 物理模拟: PhysX 在 GPU 上运行
- 相机渲染: GPU 纹理渲染

#### 7.2 高效数据传输
- **共享内存**: 进程间零拷贝传递
- **WebSocket**: 跨网络实时双向通信
- **JPEG压缩**: 85质量平衡带宽和质量

#### 7.3 批量操作
- `set_actor_root_state_tensor()`: 一次性更新所有Actor
- `render_all_camera_sensors()`: 批量渲染所有相机

---

## 时间线分析

### 单帧处理时间 (目标: 16.67ms @ 60fps)

| 步骤 | 预估时间 | 说明 |
|------|----------|------|
| VR姿态接收 | ~1ms | WebSocket 事件处理 |
| 姿态预处理 | ~0.5ms | 矩阵运算 |
| 重定向 | ~1ms | IK/查找表 |
| 更新模拟器 | ~2ms | 设置状态 |
| 物理模拟 | ~8ms | GPU计算 |
| 相机渲染 | ~3ms | GPU渲染 |
| 全景转换 | ~1ms | CPU插值 |
| 共享内存写入 | ~0.1ms | 内存复制 |
| VR端编码传输 | ~2ms | JPEG编码+WebSocket |
| **总计** | **~18.6ms** | 略超目标，需优化 |

**瓶颈**: 物理模拟和相机渲染 (GPU)

---

## 总结

本系统实现了一个完整的 VR 到 IsaacGym 遥操作闭环，关键特性包括:

1. **实时性**: 60fps 主循环，低延迟通信
2. **全景视觉**: 360° 沉浸式体验
3. **精确控制**: 25 关键点手部追踪 + 12 关节重定向
4. **高效通信**: 共享内存 + WebSocket
5. **坐标系转换**: 无缝对接 VR (Y-UP) 和模拟器 (Z-UP)

整个系统在单机多进程架构下运行，实现了高质量的远程操作体验。

