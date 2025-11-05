import time
from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import ImageBackground, group, Hands, WebRTCStereoVideoPlane, DefaultScene, Sphere
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore, Value
import numpy as np
import asyncio
from webrtc.zed_server import *
from PIL import Image
import io
import base64
import cv2
from pathlib import Path
from datetime import datetime
import msgpack
import numpy as np
from constants_vuer import grd_yup2grd_zup

class OpenTeleVision:
    def __init__(self, img_shape, shm_name, queue, toggle_streaming, stream_mode="image", cert_file="./cert.pem", key_file="./key.pem", ngrok=False, panorama_mode=True, print_freq=False, meta_shm_name=None):
        self.panorama_mode = panorama_mode
        self.print_freq = print_freq
        
        if panorama_mode:
            # 全景模式：img_shape应该是(高度, 宽度)格式的全景图尺寸
            # 全景图宽高比是2:1
            self.img_shape = img_shape
            self.img_height, self.img_width = img_shape[:2]
        else:
            # 旧的立体相机模式
            self.img_shape = (img_shape[0], 2*img_shape[1], 3)
            self.img_height, self.img_width = img_shape[:2]

        # 参考2D模式：使用queue_len=3，因为降采样后数据量大幅减少，不会导致队列积压
        # 2D模式使用queue_len=3且没有队列积压问题，因为：
        # 1. 使用ImageBackground直接传递numpy数组，Vuer自动编码
        # 2. 图像降采样减少数据量（display_image[::2]）
        # Panorama模式现在也采用降采样（[::2, ::2]），所以可以使用相同的queue_len=3
        # 在Vuer初始化时也禁用所有可能的可视化辅助元素
        # queries参数用于控制URL查询参数，可能影响默认显示
        if ngrok:
            self.app = Vuer(
                host='0.0.0.0', 
                queries=dict(
                    grid=False,
                    showGrid=False,
                    axes=False,
                    showAxes=False,
                    helpers=False,
                ), 
                queue_len=3
            )
        else:
            self.app = Vuer(
                host='0.0.0.0', 
                cert=cert_file, 
                key=key_file, 
                queries=dict(
                    grid=False,
                    showGrid=False,
                    axes=False,
                    showAxes=False,
                    helpers=False,
                ), 
                queue_len=3
            )

        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if stream_mode == "image":
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
            # 帧元数据共享（可选）：int64 frame_id + float64 t0
            self.meta_shm = None
            self.meta_buf = None
            if meta_shm_name is not None:
                try:
                    self.meta_shm = shared_memory.SharedMemory(name=meta_shm_name)
                    self.meta_buf = self.meta_shm.buf
                except Exception as e:
                    print(f"Warning: meta_shm open failed: {e}")
            if panorama_mode:
                self.app.spawn(start=False)(self.main_panorama)
            else:
                self.app.spawn(start=False)(self.main_image)
        elif stream_mode == "webrtc":
            self.app.spawn(start=False)(self.main_webrtc)
        else:
            raise ValueError("stream_mode must be either 'webrtc' or 'image'")

        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        
        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)
        if stream_mode == "webrtc":
            # webrtc server
            if Args.verbose:
                logging.basicConfig(level=logging.DEBUG)
            else:
                logging.basicConfig(level=logging.INFO)
            Args.img_shape = img_shape
            # Args.shm_name = shm_name
            Args.fps = 60

            ssl_context = ssl.SSLContext()
            ssl_context.load_cert_chain(cert_file, key_file)

            app = web.Application()
            cors = aiohttp_cors.setup(app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*",
                )
            })
            rtc = RTC(img_shape, queue, toggle_streaming, 60)
            app.on_shutdown.append(on_shutdown)
            cors.add(app.router.add_get("/", index))
            cors.add(app.router.add_get("/client.js", javascript))
            cors.add(app.router.add_post("/offer", rtc.offer))

            self.webrtc_process = Process(target=web.run_app, args=(app,), kwargs={"host": "0.0.0.0", "port": 8080, "ssl_context": ssl_context})
            self.webrtc_process.daemon = True
            self.webrtc_process.start()
            # web.run_app(app, host="0.0.0.0", port=8080, ssl_context=ssl_context)

        self.process = Process(target=self.run)
        self.process.daemon = True
        self.process.start()

    
    def run(self):
        self.app.run()

    async def on_cam_move(self, event, session):
        """
        处理CAMERA_MOVE事件
        结构与HAND_MOVE类似：event.value直接包含matrix和aspect
        
        注意：CAMERA_MOVE在浏览器打开时就会触发，但真正的VR会话是在点击"ENTER VR"按钮后才开始。
        为了区分这两者，我们采用以下策略：
        1. 记录CAMERA_MOVE事件的时间戳
        2. 当首次收到连续的CAMERA_MOVE事件（说明VR会话已真正启动）时，设置Sphere位置
        3. 或者通过检查matrix的变化来判断是否真正进入VR模式
        """
        # 记录事件接收时间（通信延迟监控）
        event_receive_time = time.time()
        
        try:
            # 调试：检查事件数据结构
            if not hasattr(event, 'value') or event.value is None:
                print("Warning: CAMERA_MOVE event has no value")
                return
            
            # 初始化CAMERA_MOVE事件计数器（用于检测VR会话真正开始）
            if not hasattr(self, '_camera_move_count'):
                self._camera_move_count = 0
                self._camera_move_timestamps = []
                self._vr_session_started = False
                self._last_head_position_print = 0
                # 用于延迟监控：记录最近的事件接收时间
                self._last_event_receive_time = {}
                self._event_processing_start = {}
            
            # 记录事件接收时间（用于计算端到端延迟）
            current_time = time.time()
            self._last_event_receive_time['CAMERA_MOVE'] = event_receive_time
            self._event_processing_start['CAMERA_MOVE'] = current_time
            
            # 仅在Sphere未初始化时进行统计和检测（初始化后跳过以降低延迟）
            sphere_initialized = getattr(self, '_sphere_initialized', False)
            if not sphere_initialized:
                self._camera_move_count += 1
                self._camera_move_timestamps.append(current_time)
                # 只保留最近1秒内的时间戳
                self._camera_move_timestamps = [ts for ts in self._camera_move_timestamps if current_time - ts < 1.0]
                
                # 调试信息：定期打印CAMERA_MOVE事件统计（每50个事件打印一次）
                if self._camera_move_count % 50 == 0:
                    print(f"[调试] CAMERA_MOVE事件统计: 总计数={self._camera_move_count}, 最近1秒内={len(self._camera_move_timestamps)}")
            
            # 调试：第一次打印事件结构（如果需要）
            if not hasattr(self, '_camera_format_logged'):
                print(f"=== CAMERA_MOVE 事件数据结构分析 ===")
                print(f"event.etype: {event.etype}")
                print(f"event.value type: {type(event.value)}")
                if isinstance(event.value, dict):
                    print(f"event.value keys: {list(event.value.keys())}")
                    if "matrix" in event.value:
                        matrix_arr = np.array(event.value["matrix"])
                        print(f"matrix type: {type(event.value['matrix'])}, shape: {matrix_arr.shape if isinstance(event.value['matrix'], (list, np.ndarray)) else 'not array'}, size: {matrix_arr.size if isinstance(event.value['matrix'], (list, np.ndarray)) else 'N/A'}")
                    if "aspect" in event.value:
                        print(f"aspect type: {type(event.value['aspect'])}, value: {event.value['aspect']}")
                print("=" * 50)
                self._camera_format_logged = True
            
            # 实际结构：event.value包含camera字典，camera字典中包含matrix和aspect
            if isinstance(event.value, dict) and "camera" in event.value:
                camera = event.value["camera"]
                
                # 调试：打印camera字典的内容
                if not hasattr(self, '_camera_format_logged'):
                    if isinstance(camera, dict):
                        print(f"event.value['camera'] keys: {list(camera.keys())}")
                        if "matrix" in camera:
                            matrix_arr = np.array(camera["matrix"])
                            print(f"camera['matrix'] type: {type(camera['matrix'])}, shape: {matrix_arr.shape if isinstance(camera['matrix'], (list, np.ndarray)) else 'not array'}, size: {matrix_arr.size if isinstance(camera['matrix'], (list, np.ndarray)) else 'N/A'}")
                        if "aspect" in camera:
                            print(f"camera['aspect'] type: {type(camera['aspect'])}, value: {camera['aspect']}")
                
                if isinstance(camera, dict):
                    matrix = None
                    aspect = None
                    
                    # 提取matrix
                    if "matrix" in camera:
                        matrix_raw = camera["matrix"]
                        matrix = np.array(matrix_raw, dtype=float)
                        
                        # 如果是2D矩阵，flatten
                        if matrix.ndim == 2:
                            matrix = matrix.flatten('F')  # 列优先，匹配4x4矩阵
                        else:
                            matrix = matrix.flatten()
                    
                    # 提取aspect
                    if "aspect" in camera:
                        aspect = float(camera["aspect"])
                    
                    # 赋值到共享内存（记录写入延迟）
                    if matrix is not None:
                        if matrix.size == 16:  # 4x4矩阵
                            t_write_start = time.time()
                            self.head_matrix_shared[:] = matrix
                            t_write_end = time.time()
                            # 记录共享内存写入延迟（通信延迟）
                            if not hasattr(self, '_comm_delays'):
                                self._comm_delays = []
                            self._comm_delays.append({
                                'type': 'comm_write_head_matrix',
                                'delay_ms': (t_write_end - t_write_start) * 1000,
                                'timestamp': t_write_end
                            })
                            
                            # 调试信息：仅在Sphere未初始化时打印头部位置（前10次或每秒一次）
                            # Sphere初始化后，不再打印头部位置以降低延迟
                            sphere_initialized = getattr(self, '_sphere_initialized', False)
                            if not sphere_initialized and (self._camera_move_count <= 10 or (current_time - self._last_head_position_print) >= 1.0):
                                current_head_pos = matrix[12:15]
                                print(f"[调试] 当前头部位置 (vuer坐标系): [{current_head_pos[0]:.4f}, {current_head_pos[1]:.4f}, {current_head_pos[2]:.4f}]")
                                self._last_head_position_print = current_time
                            
                            # 检测是否真正进入VR会话（只设置一次）：
                            # 策略：如果1秒内收到至少5次CAMERA_MOVE事件，且之前还没有初始化过Sphere位置
                            # 注意：Sphere初始化后，不再执行此检测以降低延迟
                            sphere_initialized = getattr(self, '_sphere_initialized', False)
                            if not sphere_initialized and len(self._camera_move_timestamps) >= 5 and not getattr(self, '_vr_session_started', False):
                                # 策略：如果1秒内收到至少5次CAMERA_MOVE事件，且之前还没有初始化过Sphere位置，
                                # 则认为VR会话已真正启动（点击了ENTER VR按钮）
                                # 重要：只设置一次，之后不再更新Sphere位置
                                # 从矩阵中提取头部位置（vuer坐标系，Y-up）
                                # 矩阵是列优先存储的，位置在索引12,13,14（第4列的前3个元素）
                                head_position = matrix[12:15].tolist()
                                
                                # 调试信息：打印检测到VR会话的信息
                                print(f"\n{'='*70}")
                                print(f"[⚠️ 重要] 检测到VR会话已启动！设置Sphere中心位置（仅此一次）")
                                print(f"{'='*70}")
                                print(f"  CAMERA_MOVE事件计数: {self._camera_move_count}")
                                print(f"  最近1秒内事件数: {len(self._camera_move_timestamps)}")
                                print(f"  从4x4矩阵提取位置: matrix[12:15] = [{matrix[12]:.4f}, {matrix[13]:.4f}, {matrix[14]:.4f}]")
                                print(f"  当前Sphere是否已初始化: {sphere_initialized}")
                                print(f"{'='*70}")
                                
                                # 更新Sphere位置为头部位置（只执行一次）
                                try:
                                    # 读取当前的图像数据（如果有）
                                    current_img_data = ""
                                    if hasattr(self, 'img_array'):
                                        try:
                                            from PIL import Image
                                            import io
                                            import base64
                                            display_image = self.img_array.copy()
                                            pil_img = Image.fromarray(display_image, 'RGB')
                                            buffer = io.BytesIO()
                                            pil_img.save(buffer, format='JPEG', quality=60)
                                            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                            current_img_data = f"data:image/jpeg;base64,{img_base64}"
                                        except Exception:
                                            current_img_data = ""
                                    
                                    # 更新Sphere位置
                                    sphere_radius = getattr(self, 'sphere_radius', 40.0)
                                    session.upsert @ Sphere(
                                        args=[sphere_radius, 64, 64],  # 使用较大的半径和更高的细分
                                        materialType="standard",
                                        material={"map": current_img_data, "side": 1},
                                        position=head_position,  # 设置为头部位置
                                        rotation=[0, 0.5 * np.pi, 0],  # 保持原有的旋转
                                        key="skyball"
                                    )
                                    
                                    print(f"\n{'='*60}")
                                    print(f"✓ VR会话已启动 - Sphere中心位置已更新")
                                    print(f"{'='*60}")
                                    print(f"从CAMERA_MOVE事件提取的头部位置 (vuer坐标系, Y-up):")
                                    print(f"  X: {head_position[0]:.4f}")
                                    print(f"  Y: {head_position[1]:.4f}")
                                    print(f"  Z: {head_position[2]:.4f}")
                                    print(f"  [X, Y, Z] = {head_position}")
                                    print(f"\nSphere中心位置已设置为（仅此一次，之后不再更新）:")
                                    print(f"  position = {head_position}")
                                    print(f"  (Sphere的key='skyball')")
                                    print(f"{'='*60}\n")
                                    # 标记为已初始化，确保之后不再更新
                                    self._sphere_initialized = True
                                    self._initial_head_position = head_position
                                    self._vr_session_started = True
                                    print(f"[确认] Sphere位置已固定，后续CAMERA_MOVE事件将不再更新Sphere位置")
                                    # 标记初始化信息已打印，后续不再重复显示
                                    self._init_info_printed = False  # 会在第一次定期打印时设置为True
                                except Exception as e:
                                    print(f"Warning: 更新Sphere位置失败: {e}")
                                    import traceback
                                    traceback.print_exc()
                        else:
                            print(f"Warning: CAMERA_MOVE matrix size mismatch: {matrix.size}, expected 16")
                    
                    if aspect is not None:
                        self.aspect_shared.value = aspect
                else:
                    print(f"Warning: CAMERA_MOVE event.value['camera'] is not a dict, type: {type(camera)}")
            elif isinstance(event.value, dict):
                # 兼容：如果直接包含matrix和aspect（虽然实际结构不是这样）
                if "matrix" in event.value:
                    matrix_raw = event.value["matrix"]
                    matrix = np.array(matrix_raw, dtype=float)
                    if matrix.ndim == 2:
                        matrix = matrix.flatten('F')
                    else:
                        matrix = matrix.flatten()
                    if matrix.size == 16:
                        self.head_matrix_shared[:] = matrix
                        
                        # 调试信息：仅在Sphere未初始化时打印头部位置（前10次或每秒一次）
                        # Sphere初始化后，不再打印头部位置以降低延迟
                        sphere_initialized = getattr(self, '_sphere_initialized', False)
                        if not sphere_initialized and (self._camera_move_count <= 10 or (current_time - self._last_head_position_print) >= 1.0):
                            current_head_pos = matrix[12:15]
                            print(f"[调试] 当前头部位置 (vuer坐标系, 兼容格式): [{current_head_pos[0]:.4f}, {current_head_pos[1]:.4f}, {current_head_pos[2]:.4f}]")
                            self._last_head_position_print = current_time
                        
                        # 同样处理首次进入VR的情况（只设置一次）
                        # 仅在Sphere未初始化时检测VR会话（降低延迟）
                        sphere_initialized = getattr(self, '_sphere_initialized', False)
                        if not sphere_initialized and len(self._camera_move_timestamps) >= 5 and not getattr(self, '_vr_session_started', False):
                            head_position = matrix[12:15].tolist()
                            
                            # 调试信息：打印检测到VR会话的信息
                            print(f"\n{'='*70}")
                            print(f"[⚠️ 重要] 检测到VR会话已启动！设置Sphere中心位置（仅此一次，兼容格式）")
                            print(f"{'='*70}")
                            print(f"  CAMERA_MOVE事件计数: {self._camera_move_count}")
                            print(f"  最近1秒内事件数: {len(self._camera_move_timestamps)}")
                            print(f"  从4x4矩阵提取位置: matrix[12:15] = [{matrix[12]:.4f}, {matrix[13]:.4f}, {matrix[14]:.4f}]")
                            print(f"  当前Sphere是否已初始化: {sphere_initialized}")
                            print(f"{'='*70}")
                            try:
                                current_img_data = ""
                                if hasattr(self, 'img_array'):
                                    try:
                                        from PIL import Image
                                        import io
                                        import base64
                                        display_image = self.img_array.copy()
                                        pil_img = Image.fromarray(display_image, 'RGB')
                                        buffer = io.BytesIO()
                                        pil_img.save(buffer, format='JPEG', quality=60)
                                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                        current_img_data = f"data:image/jpeg;base64,{img_base64}"
                                    except Exception:
                                        current_img_data = ""
                                
                                sphere_radius = getattr(self, 'sphere_radius', 40.0)
                                session.upsert @ Sphere(
                                    args=[sphere_radius, 64, 64],  # 使用较大的半径和更高的细分
                                    materialType="standard",
                                    material={"map": current_img_data, "side": 1},
                                    position=head_position,
                                    rotation=[0, 0.5 * np.pi, 0],
                                    key="skyball"
                                )
                                
                                print(f"\n{'='*60}")
                                print(f"✓ VR会话已启动 - Sphere中心位置已更新（仅此一次）")
                                print(f"{'='*60}")
                                print(f"从CAMERA_MOVE事件提取的头部位置 (vuer坐标系, Y-up):")
                                print(f"  X: {head_position[0]:.4f}")
                                print(f"  Y: {head_position[1]:.4f}")
                                print(f"  Z: {head_position[2]:.4f}")
                                print(f"  [X, Y, Z] = {head_position}")
                                print(f"\nSphere中心位置已设置为（仅此一次，之后不再更新）:")
                                print(f"  position = {head_position}")
                                print(f"  (Sphere的key='skyball')")
                                print(f"{'='*60}\n")
                                # 标记为已初始化，确保之后不再更新
                                self._sphere_initialized = True
                                self._initial_head_position = head_position
                                self._vr_session_started = True
                                print(f"[确认] Sphere位置已固定，后续CAMERA_MOVE事件将不再更新Sphere位置")
                                # 标记初始化信息已打印，后续不再重复显示
                                self._init_info_printed = False  # 会在第一次定期打印时设置为True
                            except Exception as e:
                                print(f"Warning: 更新Sphere位置失败: {e}")
                if "aspect" in event.value:
                    self.aspect_shared.value = float(event.value["aspect"])
            else:
                print(f"Warning: CAMERA_MOVE event.value is not a dict, type: {type(event.value)}")
        except Exception as e:
            print(f"Error in on_cam_move: {e}")
            import traceback
            traceback.print_exc()

    async def on_hand_move(self, event, session):
        # 记录事件接收时间（通信延迟监控）
        event_receive_time = time.time()
        
        try:
            # 调试：检查事件数据结构
            if not hasattr(event, 'value') or event.value is None:
                print("Warning: HAND_MOVE event has no value")
                return
            
            # 初始化延迟监控（如果还没有）
            if not hasattr(self, '_last_event_receive_time'):
                self._last_event_receive_time = {}
                self._event_processing_start = {}
            
            # 记录事件接收时间（用于计算端到端延迟）
            current_time = time.time()
            self._last_event_receive_time['HAND_MOVE'] = event_receive_time
            self._event_processing_start['HAND_MOVE'] = current_time
            
            # 将HAND_MOVE接收时间存储到共享内存，供主机端读取（用于闭环延迟计算）
            # meta_shm结构：前8字节frame_id，接下来8字节t0，接下来8字节hand_move_receive_time，最后8字节websocket_send_complete_time
            if hasattr(self, 'meta_buf') and self.meta_buf is not None:
                try:
                    mv = memoryview(self.meta_buf)
                    # 在meta_shm的16-24字节位置存储HAND_MOVE接收时间（float64）
                    # 使用numpy数组方式正确写入共享内存
                    arr = np.ndarray(1, dtype=np.float64, buffer=mv[16:24])
                    arr[0] = event_receive_time
                except Exception as e:
                    # 如果meta_shm大小不足，记录但不报错
                    pass
            # 同时存储为实例变量，作为备用
            self._last_hand_move_receive_time = event_receive_time
            
            # vuer 新版本API：键名从 leftHand/rightHand 改为 left/right
            # 检查新API格式
            if "left" in event.value and "right" in event.value:
                # 调试：打印数据结构（仅第一次，避免刷屏）
                if not hasattr(self, '_api_format_logged'):
                    print("=== HAND_MOVE 事件数据结构分析 ===")
                    print(f"event.etype: {event.etype}")
                    print(f"event.value type: {type(event.value)}")
                    print(f"event.value keys: {list(event.value.keys())}")
                    
                    # 详细分析 left 数据
                    print(f"\n--- left 数据分析 ---")
                    print(f"left type: {type(event.value['left'])}")
                    if isinstance(event.value['left'], (list, np.ndarray)):
                        left_arr = np.array(event.value['left'])
                        print(f"left shape: {left_arr.shape}, size: {left_arr.size}, ndim: {left_arr.ndim}, dtype: {left_arr.dtype}")
                        print(f"left first 16 elements (wrist matrix): {left_arr[:16]}")
                        if left_arr.size >= 32:
                            print(f"left elements 16-31 (second matrix): {left_arr[16:32]}")
                    elif isinstance(event.value['left'], dict):
                        print(f"left dict keys: {list(event.value['left'].keys())}")
                        for k, v in event.value['left'].items():
                            if isinstance(v, (list, np.ndarray)):
                                v_arr = np.array(v)
                                print(f"  left['{k}']: shape={v_arr.shape}, size={v_arr.size}, ndim={v_arr.ndim}, dtype={v_arr.dtype}")
                    
                    # 详细分析 right 数据
                    print(f"\n--- right 数据分析 ---")
                    print(f"right type: {type(event.value['right'])}")
                    if isinstance(event.value['right'], (list, np.ndarray)):
                        right_arr = np.array(event.value['right'])
                        print(f"right shape: {right_arr.shape}, size: {right_arr.size}, ndim: {right_arr.ndim}, dtype: {right_arr.dtype}")
                        print(f"right first 16 elements (wrist matrix): {right_arr[:16]}")
                        if right_arr.size >= 32:
                            print(f"right elements 16-31 (second matrix): {right_arr[16:32]}")
                    elif isinstance(event.value['right'], dict):
                        print(f"right dict keys: {list(event.value['right'].keys())}")
                        for k, v in event.value['right'].items():
                            if isinstance(v, (list, np.ndarray)):
                                v_arr = np.array(v)
                                print(f"  right['{k}']: shape={v_arr.shape}, size={v_arr.size}, ndim={v_arr.ndim}, dtype={v_arr.dtype}")
                    
                    # 分析状态数据
                    print(f"\n--- State 数据分析 ---")
                    if "leftState" in event.value:
                        left_state = event.value['leftState']
                        print(f"leftState type: {type(left_state)}")
                        if isinstance(left_state, dict):
                            print(f"leftState keys: {list(left_state.keys())}")
                            for k, v in left_state.items():
                                print(f"  leftState['{k}']: {v} (type: {type(v)})")
                    if "rightState" in event.value:
                        right_state = event.value['rightState']
                        print(f"rightState type: {type(right_state)}")
                        if isinstance(right_state, dict):
                            print(f"rightState keys: {list(right_state.keys())}")
                            for k, v in right_state.items():
                                print(f"  rightState['{k}']: {v} (type: {type(v)})")
                    
                    print("\n" + "=" * 50)
                    self._api_format_logged = True
                
                # 处理left/right数据
                # 根据vuer文档：left/right是Float32Array，包含25个4x4矩阵（25*16=400个值）
                # 第一个矩阵（索引0）是手腕（wrist）的变换矩阵
                left_raw = event.value["left"]
                right_raw = event.value["right"]
                
                # 转换为numpy数组
                # 处理msgpack ExtType（二进制数据）
                if isinstance(left_raw, msgpack.ExtType):
                    # ExtType包含二进制数据，需要从buffer中读取
                    buffer_size = len(left_raw.data)
                    # 如果缓冲区太小（小于一个float32），可能是无效数据，尝试直接转换为数组
                    if buffer_size < 4:
                        # 可能是空的或格式不同的数据，尝试直接转换
                        try:
                            left_arr = np.array(left_raw.data, dtype=float)
                            if left_arr.size == 0:
                                # 数据为空，跳过这一帧
                                return
                        except (ValueError, TypeError):
                            # 如果转换失败，跳过这一帧
                            return
                    else:
                        element_size = np.dtype(np.float32).itemsize
                        # 确保缓冲区大小是元素大小的倍数
                        if buffer_size % element_size != 0:
                            # 截断到最接近的倍数
                            buffer_size = (buffer_size // element_size) * element_size
                            if buffer_size == 0:
                                return
                            left_arr = np.frombuffer(left_raw.data[:buffer_size], dtype=np.float32).copy()
                        else:
                            left_arr = np.frombuffer(left_raw.data, dtype=np.float32).copy()
                elif isinstance(left_raw, bytes):
                    # 如果是纯bytes，也使用frombuffer
                    buffer_size = len(left_raw)
                    if buffer_size < 4:
                        return
                    element_size = np.dtype(np.float32).itemsize
                    if buffer_size % element_size != 0:
                        buffer_size = (buffer_size // element_size) * element_size
                        if buffer_size == 0:
                            return
                        left_arr = np.frombuffer(left_raw[:buffer_size], dtype=np.float32).copy()
                    else:
                        left_arr = np.frombuffer(left_raw, dtype=np.float32).copy()
                else:
                    # 尝试直接转换为数组（可能是列表或其他格式）
                    try:
                        left_arr = np.array(left_raw, dtype=float)
                        if left_arr.size == 0:
                            return
                    except (ValueError, TypeError):
                        return
                
                if isinstance(right_raw, msgpack.ExtType):
                    buffer_size = len(right_raw.data)
                    if buffer_size < 4:
                        try:
                            right_arr = np.array(right_raw.data, dtype=float)
                            if right_arr.size == 0:
                                return
                        except (ValueError, TypeError):
                            return
                    else:
                        element_size = np.dtype(np.float32).itemsize
                        if buffer_size % element_size != 0:
                            buffer_size = (buffer_size // element_size) * element_size
                            if buffer_size == 0:
                                return
                            right_arr = np.frombuffer(right_raw.data[:buffer_size], dtype=np.float32).copy()
                        else:
                            right_arr = np.frombuffer(right_raw.data, dtype=np.float32).copy()
                elif isinstance(right_raw, bytes):
                    buffer_size = len(right_raw)
                    if buffer_size < 4:
                        return
                    element_size = np.dtype(np.float32).itemsize
                    if buffer_size % element_size != 0:
                        buffer_size = (buffer_size // element_size) * element_size
                        if buffer_size == 0:
                            return
                        right_arr = np.frombuffer(right_raw[:buffer_size], dtype=np.float32).copy()
                    else:
                        right_arr = np.frombuffer(right_raw, dtype=np.float32).copy()
                else:
                    try:
                        right_arr = np.array(right_raw, dtype=float)
                        if right_arr.size == 0:
                            return
                    except (ValueError, TypeError):
                        return
                
                # 新API格式：left/right是400个元素的数组（25个4x4矩阵）
                # 我们需要提取第一个矩阵（手腕，索引0，前16个元素）
                if not hasattr(self, '_api_format_logged'):
                    print(f"\n--- 矩阵提取逻辑验证 ---")
                    print(f"left_arr.size: {left_arr.size}, right_arr.size: {right_arr.size}")
                
                if left_arr.size == 400:  # 25个4x4矩阵
                    # 提取手腕矩阵（第一个矩阵，前16个元素）
                    left_data = left_arr[:16].copy()
                    if not hasattr(self, '_api_format_logged'):
                        print(f"✓ left: 提取前16个元素作为手腕矩阵 (从400个元素中)")
                        print(f"  提取的矩阵形状: {left_data.shape}, 值: {left_data}")
                elif left_arr.size == 16:  # 直接是单个4x4矩阵
                    left_data = left_arr.copy()
                    if not hasattr(self, '_api_format_logged'):
                        print(f"✓ left: 直接使用16个元素作为矩阵")
                else:
                    print(f"Warning: Unexpected left hand data size: {left_arr.size}, expected 16 or 400")
                    return
                
                if right_arr.size == 400:  # 25个4x4矩阵
                    # 提取手腕矩阵（第一个矩阵，前16个元素）
                    right_data = right_arr[:16].copy()
                    if not hasattr(self, '_api_format_logged'):
                        print(f"✓ right: 提取前16个元素作为手腕矩阵 (从400个元素中)")
                        print(f"  提取的矩阵形状: {right_data.shape}, 值: {right_data}")
                elif right_arr.size == 16:  # 直接是单个4x4矩阵
                    right_data = right_arr.copy()
                    if not hasattr(self, '_api_format_logged'):
                        print(f"✓ right: 直接使用16个元素作为矩阵")
                else:
                    print(f"Warning: Unexpected right hand data size: {right_arr.size}, expected 16 or 400")
                    return
                
                # 赋值到共享内存（必须是16个元素）（记录写入延迟）
                t_write_start = time.time()
                self.left_hand_shared[:] = left_data
                self.right_hand_shared[:] = right_data
                t_write_end = time.time()
                # 记录共享内存写入延迟（通信延迟）
                if not hasattr(self, '_comm_delays'):
                    self._comm_delays = []
                self._comm_delays.append({
                    'type': 'comm_write_hand_data',
                    'delay_ms': (t_write_end - t_write_start) * 1000,
                    'timestamp': t_write_end
                })
                if not hasattr(self, '_api_format_logged'):
                    print(f"✓ 手腕矩阵已写入共享内存")
                
                # 获取landmarks：从25个关节矩阵中提取25个关键点的3D位置
                # 新API格式：left/right包含25个4x4矩阵，每个矩阵的第4列（索引3）前3个元素是位置
                # landmarks需要从所有25个矩阵中提取位置信息
                if left_arr.size == 400:
                    # 提取25个关节的3D位置（每个矩阵的第4列前3个元素）
                    left_landmarks = np.zeros(75, dtype=float)  # 25个点 * 3个坐标
                    for i in range(25):
                        matrix_start = i * 16  # 每个矩阵16个元素
                        # 提取矩阵的第4列（平移部分）：索引12, 13, 14（列优先存储）
                        left_landmarks[i*3:(i+1)*3] = left_arr[matrix_start+12:matrix_start+15]
                    t_write_start = time.time()
                    self.left_landmarks_shared[:] = left_landmarks
                    t_write_end = time.time()
                    # 记录landmarks写入延迟
                    if not hasattr(self, '_comm_delays'):
                        self._comm_delays = []
                    self._comm_delays.append({
                        'type': 'comm_write_landmarks',
                        'delay_ms': (t_write_end - t_write_start) * 1000,
                        'timestamp': t_write_end
                    })
                    if not hasattr(self, '_api_format_logged'):
                        print(f"✓ left landmarks: 从400个元素中提取了25个关节位置 (75个值)")
                        print(f"  前3个关节位置: {left_landmarks[:9].reshape(3, 3)}")
                elif not hasattr(self, '_api_format_logged'):
                    print(f"⚠ left: 数据大小不是400，无法提取landmarks")
                
                if right_arr.size == 400:
                    right_landmarks = np.zeros(75, dtype=float)
                    for i in range(25):
                        matrix_start = i * 16
                        right_landmarks[i*3:(i+1)*3] = right_arr[matrix_start+12:matrix_start+15]
                    t_write_start = time.time()
                    self.right_landmarks_shared[:] = right_landmarks
                    t_write_end = time.time()
                    # 记录landmarks写入延迟
                    if not hasattr(self, '_comm_delays'):
                        self._comm_delays = []
                    self._comm_delays.append({
                        'type': 'comm_write_landmarks',
                        'delay_ms': (t_write_end - t_write_start) * 1000,
                        'timestamp': t_write_end
                    })
                    if not hasattr(self, '_api_format_logged'):
                        print(f"✓ right landmarks: 从400个元素中提取了25个关节位置 (75个值)")
                        print(f"  前3个关节位置: {right_landmarks[:9].reshape(3, 3)}")
                elif not hasattr(self, '_api_format_logged'):
                    print(f"⚠ right: 数据大小不是400，无法提取landmarks")
                
                # 兼容旧格式：直接在event.value中查找landmarks
                if "leftLandmarks" in event.value:
                    landmarks = np.array(event.value["leftLandmarks"], dtype=float).flatten()
                    if len(landmarks) == 75:
                        self.left_landmarks_shared[:] = landmarks
                
                if "rightLandmarks" in event.value:
                    landmarks = np.array(event.value["rightLandmarks"], dtype=float).flatten()
                    if len(landmarks) == 75:
                        self.right_landmarks_shared[:] = landmarks
            # 兼容旧API格式
            elif "leftHand" in event.value and "rightHand" in event.value:
                left_data = np.array(event.value["leftHand"], dtype=float)
                right_data = np.array(event.value["rightHand"], dtype=float)
                
                # 如果是矩阵格式，使用Fortran顺序flatten
                if left_data.ndim == 2:
                    left_data = left_data.flatten('F')
                else:
                    left_data = left_data.flatten()
                    
                if right_data.ndim == 2:
                    right_data = right_data.flatten('F')
                else:
                    right_data = right_data.flatten()
                
                t_write_start = time.time()
                if len(left_data) == 16:
                    self.left_hand_shared[:] = left_data
                if len(right_data) == 16:
                    self.right_hand_shared[:] = right_data
                t_write_end = time.time()
                # 记录共享内存写入延迟（通信延迟）
                if not hasattr(self, '_comm_delays'):
                    self._comm_delays = []
                self._comm_delays.append({
                    'type': 'comm_write_hand_data',
                    'delay_ms': (t_write_end - t_write_start) * 1000,
                    'timestamp': t_write_end
                })
                
                if "leftLandmarks" in event.value and "rightLandmarks" in event.value:
                    left_landmarks = np.array(event.value["leftLandmarks"], dtype=float).flatten()
                    right_landmarks = np.array(event.value["rightLandmarks"], dtype=float).flatten()
                    t_write_start = time.time()
                    if len(left_landmarks) == 75:
                        self.left_landmarks_shared[:] = left_landmarks
                    if len(right_landmarks) == 75:
                        self.right_landmarks_shared[:] = right_landmarks
                    t_write_end = time.time()
                    # 记录landmarks写入延迟
                    self._comm_delays.append({
                        'type': 'comm_write_landmarks',
                        'delay_ms': (t_write_end - t_write_start) * 1000,
                        'timestamp': t_write_end
                    })
            else:
                print(f"Warning: HAND_MOVE event format not recognized. Keys: {event.value.keys() if isinstance(event.value, dict) else 'not a dict'}")
                return
        except Exception as e:
            print(f"Error in on_hand_move: {e}")
            import traceback
            traceback.print_exc()
    
    async def main_webrtc(self, session, fps=60):
        session.set @ DefaultScene(frameloop="always")
        session.upsert @ Hands(stream=True, key="hands", hideLeft=True, hideRight=True)
        session.upsert @ WebRTCStereoVideoPlane(
                src="https://192.168.8.102:8080/offer",
                # iceServer={},
                key="zed",
                aspect=1.33334,
                height = 8,
                position=[0, -2, -0.2],
            )
        while True:
            await asyncio.sleep(1)
    
    async def main_panorama(self, session, fps=60):
        """使用SkyBall显示全景图"""
        try:
            print("=== VR Panorama 初始化开始 ===")
            # 初始化场景和手部追踪
            # 严格按照官方示例：先创建Sphere对象，然后在DefaultScene初始化时传入
            # 注意：添加key以便后续能够更新这个sphere
            # 计算与Isaac Gym固定相机位置匹配的Sphere位置（坐标系转换：Z-up -> Y-up）
            # 为使浏览器视点处于全景球心，Sphere 应置于原点
            # Sphere半径：使用较大的半径（50-100米）以匹配真实世界尺度
            # 如果半径太小（如1米），物体在球面上的投影会显得非常大，导致距离感知失真
            # 例如：1米距离的物体会看起来像0.5米，感觉物体被放大了2倍
            print("创建 Sphere 组件...")
            self.sphere_radius = 40.0  # 25米半径，使物体大小更真实
            sphere = Sphere(
                args=[self.sphere_radius, 64, 64],  # 半径25米，64x64细分提供更平滑的球面
                materialType="standard",
                material={"map": "", "side": 1},
                position=[0, 0, 0],
                rotation=[0, 0.5 * np.pi, 0],  # 绕Y轴旋转90度进行坐标系转换
                key="skyball"
            )
            print("设置 DefaultScene...")
            try:
                # 根据Vuer文档，DefaultScene默认会在bgChildren中添加Grid()
                # 要隐藏红框（Grid）和红线（坐标轴），需要：
                # 1. 设置bgChildren，只包含必要的控件，不包含Grid()，从而隐藏红框
                # 2. 设置show_helper=[]来隐藏所有helper（包括坐标轴），从而隐藏红线
                from vuer.schemas import GrabRender, PointerControls
                session.set @ DefaultScene(
                    sphere,
                    bgChildren=[
                        GrabRender(),  # 保留必要的渲染控件
                        PointerControls(),  # 保留指针控件
                        # 不包含Grid()，从而隐藏红框（地上和天上的）
                    ],
                    show_helper=[],  # 空列表隐藏所有helper元素（包括坐标轴），从而隐藏红线
                )
            except AssertionError as e:
                print(f"❌ WebSocket connection lost during DefaultScene initialization: {e}")
                print("   客户端可能在初始化过程中断开了连接")
                raise
            except Exception as e:
                print(f"❌ Error setting DefaultScene: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # 确保Hands组件在DefaultScene之后立即初始化，以便接收手部跟踪事件
            # 注意：设置 hideLeft=True, hideRight=True 来隐藏手部轮廓显示，但保持数据流接收
            print("初始化 Hands 组件（隐藏手部轮廓）...")
            try:
                session.upsert @ Hands(stream=True, key="hands", hideLeft=True, hideRight=True)
            except AssertionError as e:
                print(f"❌ WebSocket connection lost during Hands initialization: {e}")
                print("   客户端可能在初始化过程中断开了连接")
                raise
            except Exception as e:
                print(f"❌ Error initializing Hands: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            print("✓ Hands component initialized for panorama mode")
            print("✓ VR Panorama 初始化完成，等待客户端连接...")
        except AssertionError as e:
            # WebSocket连接断开，重新抛出但不打印堆栈（这是正常的断开情况）
            print(f"❌ VR Panorama 初始化失败: WebSocket connection lost")
            raise
        except Exception as e:
            print(f"❌ VR Panorama 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        frame_count = 0
        last_frame_id = -1
        start = time.time()  # 用于FPS计算
        # 端到端延迟日志（与latency_log_move.txt格式一致）
        vr_latency_log = Path(__file__).parent.parent / 'latency_log_vr.txt'
        try:
            with open(vr_latency_log, 'w', encoding='utf-8') as f:
                f.write(f"=== VR端延迟监控日志 - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        except Exception:
            pass
        
        # 当前帧的延迟数据结构
        current_frame = {
            'frame_id': None,
            'start_time': None,
            'compute_delays': {},
            'transmit_delays': {},
            't0': None
        }
        
        # 用于带宽延迟测量的历史记录
        bandwidth_history = {
            'transmit_times': [],  # 传输时间列表
            'transmit_sizes': [],  # 传输大小列表
            'frame_intervals': [],  # 帧间隔列表
            'last_send_time': None,  # 上次发送时间
            'expected_interval': 30.0,  # 期望帧间隔（毫秒），对应约33FPS (1000/30 ≈ 33.3)
            'queue_backlog_frames': 0,  # 队列积压的帧数
            'queue_backlog_bytes': 0,  # 队列积压的字节数
        }
        
        while True:
            frame_start = time.time()
            # 仅处理最新帧：读取meta中的frame_id与t0
            curr_frame_id = None
            t0 = None
            if self.meta_buf is not None:
                try:
                    # 前8字节 int64 frame_id
                    mv = memoryview(self.meta_buf)
                    curr_frame_id = int(np.frombuffer(mv[:8], dtype=np.int64, count=1)[0])
                    # 后8字节 float64 t0
                    t0 = float(np.frombuffer(mv[8:16], dtype=np.float64, count=1)[0])
                except Exception:
                    curr_frame_id = None
                    t0 = None

            # 帧跳过逻辑优化：即使frame_id相同，也要检查图像年龄
            # 如果图像年龄过大（>30ms），强制读取以避免延迟累积
            # 降低阈值从50ms到30ms，因为主机端处理约25ms，如果图像年龄>30ms说明已经是旧帧了
            should_skip = False
            force_process_old_frame = False
            if curr_frame_id is not None and curr_frame_id == last_frame_id:
                # 检查图像年龄
                if t0 is not None:
                    image_age_ms = (frame_start - t0) * 1000  # 毫秒
                    # 如果图像年龄大于30ms（主机处理时间约25ms），强制处理以避免延迟累积
                    if image_age_ms > 30:
                        # 图像已过时，强制处理以避免延迟累积
                        force_process_old_frame = True
                        should_skip = False
                    else:
                        # 图像很新，等待新帧
                        await asyncio.sleep(0.005)
                        should_skip = True
                else:
                    # 没有t0信息，等待新帧
                    await asyncio.sleep(0.005)
                    should_skip = True
            
            if should_skip:
                continue

            # 开始新的一帧（或强制处理旧帧）
            current_frame = None  # 初始化为None
            if curr_frame_id is not None:
                # 如果是强制处理旧帧，也创建current_frame以记录诊断信息
                if force_process_old_frame:
                    # 强制处理旧帧，记录诊断信息但不增加frame_count
                    current_frame = {
                        'frame_id': curr_frame_id,
                        'start_time': t0 if t0 is not None else frame_start,
                        'compute_delays': {},
                        'transmit_delays': {},
                        'timestamps': {},  # 初始化 timestamps 字典
                        't0': t0,
                        'force_processed': True  # 标记为强制处理的旧帧
                    }
                else:
                    # 正常新帧
                    frame_count += 1
                    current_frame = {
                        'frame_id': curr_frame_id,
                        'start_time': t0 if t0 is not None else frame_start,
                        'compute_delays': {},
                        'transmit_delays': {},
                        'timestamps': {},  # 初始化 timestamps 字典
                        't0': t0
                    }
                
                # 记录帧跳过情况（用于诊断延迟）
                if last_frame_id >= 0 and not force_process_old_frame:
                    frames_skipped = curr_frame_id - last_frame_id - 1
                    if frames_skipped > 0:
                        current_frame['frames_skipped'] = frames_skipped
                        current_frame['last_frame_id'] = last_frame_id
                        current_frame['current_frame_id'] = curr_frame_id
                        # 计算因跳过帧导致的延迟（每跳过一个帧约30ms）
                        skip_delay_ms = frames_skipped * 30.0
                        if 'bandwidth_delays' not in current_frame:
                            current_frame['bandwidth_delays'] = {}
                        current_frame['bandwidth_delays']['frame_skip_delay'] = skip_delay_ms
            
            # 计算延迟：读取共享内存并复制
            # 注意：即使frame_id相同，如果图像年龄过大，也要读取以避免延迟累积
            t_read_start = time.time()
            display_image = self.img_array.copy()
            t_read_end = time.time()
            
            # 无论是否有current_frame，都计算图像年龄用于诊断
            image_age_ms = None
            if t0 is not None:
                image_age_ms = (t_read_start - t0) * 1000  # 毫秒
            
            if current_frame is not None:
                current_frame['compute_delays']['vr_read_shared_memory'] = (t_read_end - t_read_start) * 1000
                
                # 计算图像年龄（从主机端写入到VR端读取的时间差）
                if image_age_ms is not None:
                    current_frame['image_age_ms'] = image_age_ms
                    # timestamps 字典已在初始化时创建
                    current_frame['timestamps']['image_read_start'] = t_read_start
                    current_frame['timestamps']['image_write_time'] = t0
                    
                    # 如果图像年龄过大（>100ms），说明有延迟
                    if image_age_ms > 100:
                        current_frame['image_age_excessive'] = True
                        # 记录为带宽延迟
                        if 'bandwidth_delays' not in current_frame:
                            current_frame['bandwidth_delays'] = {}
                        current_frame['bandwidth_delays']['image_age_delay'] = image_age_ms - 30  # 减去期望的30ms主机处理时间
            elif image_age_ms is not None and image_age_ms > 100:
                # 即使没有current_frame（frame_id相同且图像很新），如果图像年龄过大也要记录警告
                # 这不应该发生，但如果发生了，说明有严重问题
                print(f"⚠️ 警告: 图像年龄={image_age_ms:.1f}ms，但frame_id未变化，可能存在严重延迟问题")
            
            # 参考2D模式：添加图像降采样以减少传输数据量（关键优化）
            # 2D模式使用 display_image[::2] 只降采样高度，减少50%的数据量但保持宽度
            # 对于panorama模式，我们也只降采样高度，保持宽度不变，减少50%的数据量
            # Sphere渲染时会自动进行纹理插值，降采样后不会明显模糊
            # 这样既减少传输时间避免WebSocket队列积压，又保持较好的画质
            if current_frame is not None:
                t_downsample_start = time.time()
            
            # 降采样：使用抗锯齿的双线性插值降采样，而不是简单跳像素
            # 800x1600 -> 400x1600，减少50%的数据量
            # 使用cv2.INTER_AREA（适合降采样）或cv2.INTER_LINEAR（双线性插值）
            # 这样可以保持更好的画质，减少锯齿，Sphere渲染时也会更平滑
            target_height = display_image.shape[0] // 2
            target_width = display_image.shape[1]
            # INTER_AREA适合降采样，质量更好；INTER_LINEAR更快但质量稍差
            # 使用INTER_AREA可以获得更好的抗锯齿效果
            display_image_downsampled = cv2.resize(
                display_image, 
                (target_width, target_height), 
                interpolation=cv2.INTER_AREA
            )
            
            if current_frame is not None:
                t_downsample_end = time.time()
                current_frame['compute_delays']['vr_image_downsample'] = (t_downsample_end - t_downsample_start) * 1000
                # 记录降采样后的尺寸
                current_frame['image_size_downsampled'] = display_image_downsampled.shape
                current_frame['image_size_original'] = display_image.shape
                current_frame['downsample_ratio'] = 0.5  # 降采样后数据量为原来的50%（只降采样高度）
            
            # 计算延迟：PIL图像转换（使用降采样后的图像）
            t_pil_start = time.time()
            pil_img = Image.fromarray(display_image_downsampled, 'RGB')
            t_pil_end = time.time()
            if current_frame is not None:
                current_frame['compute_delays']['vr_pil_convert'] = (t_pil_end - t_pil_start) * 1000
            
            # 计算延迟：JPEG编码
            # 使用quality=80（与2D模式一致）
            # 降采样后是400x1600，比2D模式的360x1280更高，质量会更好
            t_jpeg_start = time.time()
            buffer = io.BytesIO()
            # 使用quality=80，因为：
            # 1. 2D模式使用ImageBackground自动编码，quality=80
            # 2. Panorama模式降采样后是400x1600，比2D模式的360x1280更高
            # 3. 使用quality=80可以保持高质量画质
            pil_img.save(buffer, format='JPEG', quality=80)
            t_jpeg_end = time.time()
            jpeg_size_bytes = len(buffer.getvalue())  # JPEG编码后的字节数
            
            if current_frame is not None:
                current_frame['compute_delays']['vr_jpeg_encode'] = (t_jpeg_end - t_jpeg_start) * 1000
                # 记录图像大小（用于带宽计算）
                current_frame['image_size_jpeg_bytes'] = jpeg_size_bytes
                current_frame['image_size_jpeg_kb'] = jpeg_size_bytes / 1024.0
            
            # 计算延迟：Base64编码（关键优化：使用更高效的方式）
            # Base64编码会增加33%的数据量，但我们不能跳过它，因为Sphere组件需要data URL
            # 优化：使用更高效的base64编码方式
            t_base64_start = time.time()
            jpeg_bytes = buffer.getvalue()
            # 直接使用bytes进行base64编码，避免额外的字符串操作
            img_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
            img_data_url = f"data:image/jpeg;base64,{img_base64}"
            t_base64_end = time.time()
            base64_size_bytes = len(img_base64)  # Base64编码后的字节数（字符串长度）
            
            if current_frame is not None:
                current_frame['compute_delays']['vr_base64_encode'] = (t_base64_end - t_base64_start) * 1000
                # 记录Base64编码后的大小（用于带宽计算）
                current_frame['image_size_base64_bytes'] = base64_size_bytes
                current_frame['image_size_base64_kb'] = base64_size_bytes / 1024.0
                # Base64比原始数据大约33%：base64_size ≈ jpeg_size * 1.33
                # 实际传输的数据大小（字节数）
                current_frame['transmit_size_bytes'] = base64_size_bytes
                current_frame['transmit_size_kb'] = base64_size_bytes / 1024.0
                
                # 记录JPEG原始大小用于对比
                current_frame['image_size_jpeg_bytes'] = len(jpeg_bytes)
                current_frame['base64_overhead_ratio'] = base64_size_bytes / len(jpeg_bytes) if len(jpeg_bytes) > 0 else 1.0
            
            # 计算延迟：Sphere位置和参数准备
            t_sphere_prep_start = time.time()
            # 更新Sphere贴图，保持已设置的position（如果已初始化）
            # 如果已在on_cam_move中设置了position，使用该position；否则使用默认的[0, 0, 0]
            sphere_position = getattr(self, '_initial_head_position', [0, 0, 0])
            sphere_initialized = getattr(self, '_sphere_initialized', False)
            
            # 调试信息：仅在Sphere未初始化时打印位置信息（每100帧一次）
            # Sphere初始化后，完全跳过这些检查以降低延迟
            if not sphere_initialized:
                if not hasattr(self, '_sphere_update_count'):
                    self._sphere_update_count = 0
                self._sphere_update_count += 1
                if self._sphere_update_count % 100 == 0:
                    sphere_radius = getattr(self, 'sphere_radius', 40.0)
                    print(f"[调试] 第{self._sphere_update_count}帧 - Sphere未初始化，位置: {sphere_position}, 半径: {sphere_radius}")
            
            sphere_radius = getattr(self, 'sphere_radius', 40.0)
            t_sphere_prep_end = time.time()
            if current_frame is not None:
                current_frame['compute_delays']['vr_sphere_preparation'] = (t_sphere_prep_end - t_sphere_prep_start) * 1000
            
            # 传输延迟：WebSocket upsert（包含网络传输延迟）
            # 参考2D模式：不需要stream=True，因为降采样后数据量小，queue_len=3不会积压
            # 2D模式直接传递numpy数组给ImageBackground，由Vuer自动编码，也不使用stream
            t_upsert_start = time.time()
            try:
                session.upsert @ Sphere(
                    args=[sphere_radius, 32, 32],  # 降低细分级别：从64x64到32x32，减少75%的几何数据量
                    materialType="standard",
                    material={"map": img_data_url, "side": 1},
                    position=sphere_position,  # 使用已设置的头部位置，或默认[0, 0, 0]
                    rotation=[0, 0.5 * np.pi, 0],
                    key="skyball"
                )
                t_upsert_end = time.time()
                if current_frame is not None:
                    websocket_send_time = (t_upsert_end - t_upsert_start) * 1000  # 毫秒
                    current_frame['transmit_delays']['vr_websocket_upsert'] = websocket_send_time
                    
                    # 将WebSocket发送完成时间写入共享内存（闭环延迟的终点）
                    # 这是图像真正发送到VR的时间，包括网络传输
                    if hasattr(self, 'meta_buf') and self.meta_buf is not None:
                        try:
                            mv = memoryview(self.meta_buf)
                            # 在meta_shm的24-32字节位置存储WebSocket发送完成时间（float64）
                            # 使用numpy数组方式正确写入共享内存
                            arr = np.ndarray(1, dtype=np.float64, buffer=mv[24:32])
                            arr[0] = t_upsert_end
                        except Exception:
                            pass
                    
                    # 计算带宽延迟（传输延迟）
                    if 'transmit_size_bytes' in current_frame:
                        transmit_size_bytes = current_frame['transmit_size_bytes']
                        transmit_size_kb = current_frame['transmit_size_kb']
                        
                        # 记录到历史中（用于分析带宽限制）
                        bandwidth_history['transmit_times'].append(websocket_send_time)
                        bandwidth_history['transmit_sizes'].append(transmit_size_bytes)
                        # 只保留最近10帧的历史
                        if len(bandwidth_history['transmit_times']) > 10:
                            bandwidth_history['transmit_times'].pop(0)
                            bandwidth_history['transmit_sizes'].pop(0)
                        
                        # 计算帧间隔（用于检测队列阻塞）
                        if bandwidth_history['last_send_time'] is not None:
                            frame_interval = (t_upsert_start - bandwidth_history['last_send_time']) * 1000  # 毫秒
                            bandwidth_history['frame_intervals'].append(frame_interval)
                            if len(bandwidth_history['frame_intervals']) > 10:
                                bandwidth_history['frame_intervals'].pop(0)
                        bandwidth_history['last_send_time'] = t_upsert_start
                        
                        # 如果传输时间>0，估算实际带宽
                        if websocket_send_time > 0:
                            # 方法1：单帧估算带宽（可能不准确，因为WebSocket是异步的）
                            # 估算带宽（Mbps）= (数据大小(字节) * 8) / (传输时间(秒) * 1,000,000)
                            estimated_bandwidth_mbps = (transmit_size_bytes * 8) / (websocket_send_time / 1000.0) / 1_000_000
                            current_frame['bandwidth_estimated_mbps'] = estimated_bandwidth_mbps
                            
                            # 方法2：基于历史数据计算平均传输速率（更准确）
                            if len(bandwidth_history['transmit_times']) >= 3:
                                total_size = sum(bandwidth_history['transmit_sizes'])
                                total_time = sum(bandwidth_history['transmit_times'])
                                if total_time > 0:
                                    avg_transmit_rate_mbps = (total_size * 8) / (total_time / 1000.0) / 1_000_000
                                    current_frame['bandwidth_avg_estimated_mbps'] = avg_transmit_rate_mbps
                                else:
                                    current_frame['bandwidth_avg_estimated_mbps'] = 0
                            else:
                                current_frame['bandwidth_avg_estimated_mbps'] = estimated_bandwidth_mbps
                            
                            # 方法3：基于帧间隔和数据大小计算实际带宽需求
                            if len(bandwidth_history['frame_intervals']) >= 3:
                                avg_frame_interval = sum(bandwidth_history['frame_intervals']) / len(bandwidth_history['frame_intervals'])
                                # 安全获取期望间隔，如果不存在则使用默认值30ms（对应33FPS）
                                expected_interval = bandwidth_history.get('expected_interval', 30.0)
                                
                                if avg_frame_interval > 0:
                                    # 实际带宽需求 = (单帧大小 * 8) / (平均帧间隔(秒))
                                    required_bandwidth_mbps = (transmit_size_bytes * 8) / (avg_frame_interval / 1000.0) / 1_000_000
                                    current_frame['bandwidth_required_mbps'] = required_bandwidth_mbps
                                    
                                    # 检测队列阻塞
                                    if avg_frame_interval > expected_interval * 1.2:  # 如果间隔大于预期的1.2倍，认为有阻塞
                                        # 计算积压的帧数（基于最近的帧间隔）
                                        recent_intervals = bandwidth_history['frame_intervals'][-5:]  # 最近5帧
                                        backlog_frames = 0
                                        for interval in recent_intervals:
                                            if interval > expected_interval:
                                                # 如果这一帧的间隔大于预期，说明积压了额外的帧
                                                extra_frames = (interval - expected_interval) / expected_interval
                                                backlog_frames += extra_frames
                                        
                                        # 估算当前队列中积压的帧数（累计）
                                        backlog_frames = max(0, backlog_frames)
                                        bandwidth_history['queue_backlog_frames'] = backlog_frames
                                        
                                        # 计算积压的字节数（基于最近的平均帧大小）
                                        avg_frame_size = sum(bandwidth_history['transmit_sizes']) / len(bandwidth_history['transmit_sizes'])
                                        backlog_bytes = backlog_frames * avg_frame_size
                                        bandwidth_history['queue_backlog_bytes'] = backlog_bytes
                                        
                                        # 如果平均带宽已计算，估算清空队列需要的时间
                                        if 'bandwidth_avg_estimated_mbps' in current_frame and current_frame['bandwidth_avg_estimated_mbps'] > 0:
                                            available_bandwidth_mbps = current_frame['bandwidth_avg_estimated_mbps']
                                            # 清空队列时间 = 积压字节数 * 8 / 带宽(bps)
                                            queue_drain_time_ms = (backlog_bytes * 8) / (available_bandwidth_mbps * 1_000_000) * 1000
                                            current_frame['queue_backlog_drain_time_ms'] = queue_drain_time_ms
                                        
                                        # 队列阻塞导致的延迟 = 积压帧数 * 期望帧间隔 + 清空队列时间
                                        queue_blocking_delay_ms = backlog_frames * expected_interval
                                        if 'queue_backlog_drain_time_ms' in current_frame:
                                            queue_blocking_delay_ms += current_frame['queue_backlog_drain_time_ms']
                                        
                                        current_frame['queue_backlog_frames'] = backlog_frames
                                        current_frame['queue_backlog_bytes'] = backlog_bytes
                                        current_frame['queue_backlog_kb'] = backlog_bytes / 1024.0
                                        current_frame['queue_blocking_delay_ms'] = queue_blocking_delay_ms
                                        
                                        # 带宽不足导致的延迟（基于帧间隔差异）
                                        size_ratio = transmit_size_bytes / avg_frame_size
                                        bandwidth_delay_from_interval = (avg_frame_interval - expected_interval) * size_ratio
                                        current_frame['transmit_delays']['bandwidth_delay_from_interval'] = bandwidth_delay_from_interval
                                        
                                        # 队列阻塞延迟（更准确）
                                        current_frame['transmit_delays']['queue_blocking_delay'] = queue_blocking_delay_ms
                                    else:
                                        # 没有队列阻塞，重置积压计数
                                        bandwidth_history['queue_backlog_frames'] = 0
                                        bandwidth_history['queue_backlog_bytes'] = 0
                                        current_frame['queue_backlog_frames'] = 0
                                        current_frame['queue_backlog_bytes'] = 0
                                else:
                                    current_frame['bandwidth_required_mbps'] = 0
                            else:
                                current_frame['bandwidth_required_mbps'] = 0
                            
                            # 理论传输时间（假设不同带宽）
                            for test_bandwidth in [5, 10, 20, 50, 100]:  # 测试不同带宽（Mbps）
                                theoretical_time_ms = (transmit_size_bytes * 8) / (test_bandwidth * 1_000_000) * 1000
                                current_frame[f'theoretical_time_{test_bandwidth}mbps_ms'] = theoretical_time_ms
                            
                            # 带宽延迟：基于理论最小传输时间和实际传输时间的差异
                            # 由于WebSocket是异步的，这里只能估算
                            # 使用更保守的估算：假设本地处理时间很小（<1ms），剩余时间主要是网络传输
                            local_overhead_ms = 1.0  # 假设本地序列化等开销1ms
                            estimated_network_time_ms = max(0, websocket_send_time - local_overhead_ms)
                            
                            # 理论最小网络传输时间（假设100Mbps理想带宽）
                            theoretical_min_time_ms = (transmit_size_bytes * 8) / (100 * 1_000_000) * 1000
                            
                            # 带宽延迟 = 实际网络传输时间 - 理论最小传输时间
                            bandwidth_delay_ms = estimated_network_time_ms - theoretical_min_time_ms
                            current_frame['transmit_delays']['bandwidth_delay'] = max(0, bandwidth_delay_ms)
                            
                            # 最终带宽延迟：取基于传输时间、帧间隔、队列阻塞中的最大值
                            final_bandwidth_delay = bandwidth_delay_ms
                            if 'bandwidth_delay_from_interval' in current_frame['transmit_delays']:
                                interval_delay = current_frame['transmit_delays']['bandwidth_delay_from_interval']
                                final_bandwidth_delay = max(final_bandwidth_delay, interval_delay)
                            if 'queue_blocking_delay' in current_frame['transmit_delays']:
                                queue_delay = current_frame['transmit_delays']['queue_blocking_delay']
                                final_bandwidth_delay = max(final_bandwidth_delay, queue_delay)
                            
                            current_frame['transmit_delays']['bandwidth_delay'] = final_bandwidth_delay
                        else:
                            current_frame['bandwidth_estimated_mbps'] = 0
                            current_frame['bandwidth_avg_estimated_mbps'] = 0
                            current_frame['bandwidth_required_mbps'] = 0
                            current_frame['transmit_delays']['bandwidth_delay'] = 0
                    
                    # 写入完整格式的延迟日志
                    if current_frame is not None:
                        frame_end = time.time()
                        total_delay = (frame_end - current_frame['start_time']) * 1000
                        fps = 1000.0 / total_delay if total_delay > 0 else 0
                        
                        try:
                            with open(vr_latency_log, 'a', encoding='utf-8') as f:
                                f.write(f"\n--- 帧 #{frame_count} (frame_id={curr_frame_id}) ---\n")
                                f.write(f"总延迟: {total_delay:.3f} ms, 帧率: {fps:.2f} FPS\n")
                                
                                # 计算并写入通信延迟
                                if hasattr(self, '_comm_delays') and len(self._comm_delays) > 0:
                                    # 获取最近1秒内的通信延迟记录
                                    recent_comm_delays = [d for d in self._comm_delays if frame_end - d['timestamp'] < 1.0]
                                    
                                    # 按类型统计通信延迟
                                    comm_delay_summary = {}
                                    for delay_record in recent_comm_delays:
                                        delay_type = delay_record['type']
                                        if delay_type not in comm_delay_summary:
                                            comm_delay_summary[delay_type] = []
                                        comm_delay_summary[delay_type].append(delay_record['delay_ms'])
                                    
                                    if comm_delay_summary:
                                        f.write("\n通信延迟 (最近1秒内的平均):\n")
                                        for delay_type, delays in comm_delay_summary.items():
                                            avg_delay = sum(delays) / len(delays)
                                            f.write(f"  {delay_type}: {avg_delay:.3f} ms (样本数: {len(delays)})\n")
                                    
                                    # 清理旧的延迟记录（保留最近2秒）
                                    self._comm_delays = [d for d in self._comm_delays if frame_end - d['timestamp'] < 2.0]
                                
                                # 计算端到端延迟（从事件接收到图像发送）
                                if hasattr(self, '_last_event_receive_time'):
                                    end_to_end_delays = []
                                    for event_type, receive_time in self._last_event_receive_time.items():
                                        if receive_time is not None:
                                            e2e_delay = (frame_end - receive_time) * 1000
                                            if 0 < e2e_delay < 1000:  # 合理的延迟范围
                                                end_to_end_delays.append((event_type, e2e_delay))
                                    
                                    if end_to_end_delays:
                                        f.write("\n端到端延迟 (从事件接收到图像发送):\n")
                                        for event_type, e2e_delay in end_to_end_delays:
                                            f.write(f"  {event_type}: {e2e_delay:.3f} ms\n")
                                
                                if t0 is not None:
                                    f.write(f"端到端延迟 (t3-t0): {(frame_end - t0) * 1000:.3f} ms\n")
                                
                                # 计算延迟分类统计
                                compute_total = sum(current_frame['compute_delays'].values())
                                transmit_total = sum(current_frame['transmit_delays'].values())
                                
                                f.write(f"\n计算延迟 (总计: {compute_total:.3f} ms):\n")
                                for name, delay in sorted(current_frame['compute_delays'].items()):
                                    percentage = (delay / compute_total * 100) if compute_total > 0 else 0
                                    f.write(f"  {name}: {delay:.3f} ms ({percentage:.1f}%)\n")
                                
                                # 图像大小信息
                                if 'image_size_jpeg_kb' in current_frame:
                                    f.write(f"\n图像大小信息:\n")
                                    f.write(f"  JPEG压缩后: {current_frame['image_size_jpeg_kb']:.2f} KB ({current_frame['image_size_jpeg_bytes']} 字节)\n")
                                    f.write(f"  Base64编码后: {current_frame['image_size_base64_kb']:.2f} KB ({current_frame['image_size_base64_bytes']} 字节)\n")
                                    f.write(f"  实际传输大小: {current_frame['transmit_size_kb']:.2f} KB\n")
                                
                                f.write(f"\n传输延迟 (总计: {transmit_total:.3f} ms):\n")
                                for name, delay in sorted(current_frame['transmit_delays'].items()):
                                    percentage = (delay / transmit_total * 100) if transmit_total > 0 else 0
                                    f.write(f"  {name}: {delay:.3f} ms ({percentage:.1f}%)\n")
                                
                                # 带宽分析（只要有传输数据就写入，即使带宽估算失败）
                                if 'transmit_size_bytes' in current_frame:
                                    f.write(f"\n带宽分析:\n")
                                    transmit_size_kb = current_frame['transmit_size_kb']
                                    websocket_time = current_frame['transmit_delays'].get('vr_websocket_upsert', 0)
                                    f.write(f"  传输数据: {transmit_size_kb:.2f} KB\n")
                                    f.write(f"  WebSocket发送时间: {websocket_time:.3f} ms\n")
                                    
                                    # 如果有带宽估算，显示估算值
                                    if 'bandwidth_estimated_mbps' in current_frame:
                                        f.write(f"  估算带宽: {current_frame['bandwidth_estimated_mbps']:.2f} Mbps\n")
                                        if 'bandwidth_avg_estimated_mbps' in current_frame:
                                            f.write(f"  平均估算带宽: {current_frame['bandwidth_avg_estimated_mbps']:.2f} Mbps\n")
                                        if 'bandwidth_required_mbps' in current_frame:
                                            f.write(f"  所需带宽: {current_frame['bandwidth_required_mbps']:.2f} Mbps\n")
                                    
                                    # 理论传输时间对比
                                    f.write(f"  理论传输时间 (不同带宽假设):\n")
                                    for test_bandwidth in [5, 10, 20, 50, 100]:
                                        key = f'theoretical_time_{test_bandwidth}mbps_ms'
                                        if key in current_frame:
                                            f.write(f"    {test_bandwidth} Mbps: {current_frame[key]:.3f} ms\n")
                                    
                                    # 带宽延迟（额外延迟）
                                    bandwidth_delay = current_frame['transmit_delays'].get('bandwidth_delay', 0)
                                    
                                    # 显示图像年龄和帧跳过信息
                                    if 'image_age_ms' in current_frame:
                                        image_age = current_frame['image_age_ms']
                                        f.write(f"\n图像年龄分析:\n")
                                        f.write(f"  图像年龄: {image_age:.3f} ms (从主机端写入到VR端读取的时间)\n")
                                        if current_frame.get('image_age_excessive', False):
                                            f.write(f"  ⚠️  警告: 图像年龄过大，说明图像在共享内存中停留时间过长\n")
                                        if 'frames_skipped' in current_frame and current_frame['frames_skipped'] > 0:
                                            f.write(f"  跳过的帧数: {current_frame['frames_skipped']} 帧\n")
                                            f.write(f"  说明: VR端跳过了 {current_frame['frames_skipped']} 帧，可能导致显示延迟\n")
                                    
                                    if 'queue_backlog_frames' in current_frame and current_frame['queue_backlog_frames'] > 0:
                                        f.write(f"\n队列阻塞分析:\n")
                                        f.write(f"  积压帧数: {current_frame['queue_backlog_frames']:.2f} 帧\n")
                                        f.write(f"  积压数据: {current_frame.get('queue_backlog_kb', 0):.2f} KB ({current_frame.get('queue_backlog_bytes', 0)} 字节)\n")
                                        if 'queue_backlog_drain_time_ms' in current_frame:
                                            f.write(f"  清空队列预计时间: {current_frame['queue_backlog_drain_time_ms']:.3f} ms\n")
                                        if 'queue_blocking_delay_ms' in current_frame:
                                            f.write(f"  队列阻塞延迟: {current_frame['queue_blocking_delay_ms']:.3f} ms\n")
                                            
                                            # 队列阻塞的组成部分
                                            if 'bandwidth_delay_from_interval' in current_frame['transmit_delays']:
                                                interval_delay = current_frame['transmit_delays']['bandwidth_delay_from_interval']
                                                f.write(f"     - 帧间隔延迟: {interval_delay:.3f} ms\n")
                                            if 'queue_blocking_delay' in current_frame['transmit_delays']:
                                                queue_delay = current_frame['transmit_delays']['queue_blocking_delay']
                                                f.write(f"     - 队列积压延迟: {queue_delay:.3f} ms\n")
                                    
                                    # 总是写入带宽延迟信息（即使为0也要记录）
                                    if bandwidth_delay > 0:
                                        f.write(f"\n  带宽限制造成的总延迟: {bandwidth_delay:.3f} ms\n")
                                    else:
                                        f.write(f"\n  带宽延迟: 很小（网络带宽充足）\n")
                                
                                f.write("\n")
                        except Exception as e:
                            print(f"Warning: 写入延迟日志失败: {e}")
                    
                    # 更新last_frame_id（只有在正常处理新帧时才更新，强制处理的旧帧不更新）
                    if curr_frame_id is not None:
                        if not (current_frame is not None and current_frame.get('force_processed', False)):
                            last_frame_id = curr_frame_id
            except AssertionError as e:
                # WebSocket连接已断开，退出循环
                print(f"❌ WebSocket connection lost: {e}")
                print("   可能原因：")
                print("   1. 客户端关闭了VR会话")
                print("   2. 网络连接中断")
                print("   3. VR设备进入休眠状态")
                break
            except Exception as e:
                # 其他错误，记录但不退出
                print(f"⚠️  Error updating Sphere (frame {frame_count}): {e}")
                import traceback
                traceback.print_exc()
            
            frame_count += 1
            if self.print_freq and frame_count % 60 == 0:
                print(f'Panorama FPS: {60 / (time.time() - start):.2f}')
            
            # 保持33fps：等待0.03秒（1000/33.33 ≈ 30ms）
            await asyncio.sleep(0.03)  # 33.33 FPS，与2D模式和主机端一致
    
    async def main_image(self, session, fps=60):
        """旧的立体相机显示模式"""
        session.upsert @ Hands(stream=True, key="hands", hideLeft=True, hideRight=True)
        end_time = time.time()
        while True:
            start = time.time()
            display_image = self.img_array

            session.upsert(
            [ImageBackground(
                display_image[::2, :self.img_width],
                format="jpeg",
                quality=80,
                key="left-image",
                interpolate=True,
                aspect=1.66667,
                height = 8,
                position=[0, -1, 3],
                layers=1, 
                alphaSrc="./vinette.jpg"
            ),
            ImageBackground(
                display_image[::2, self.img_width:],
                format="jpeg",
                quality=80,
                key="right-image",
                interpolate=True,
                aspect=1.66667,
                height = 8,
                position=[0, -1, 3],
                layers=2, 
                alphaSrc="./vinette.jpg"
            )],
            to="bgChildren",
            )
            end_time = time.time()
            # 保持33fps：等待0.03秒（1000/33.33 ≈ 30ms）
            await asyncio.sleep(0.03)  # 33.33 FPS

    @property
    def left_hand(self):
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def right_hand(self):
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def left_landmarks(self):
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    
    @property
    def right_landmarks(self):
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        return float(self.aspect_shared.value)

    
if __name__ == "__main__":
    resolution = (720, 1280)
    crop_size_w = 340
    crop_size_h = 270
    resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)
    img_shape = (2 * resolution_cropped[0], resolution_cropped[1], 3)
    img_height, img_width = resolution_cropped[:2]
    shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    shm_name = shm.name
    img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)

    tv = OpenTeleVision(resolution_cropped, cert_file="../cert.pem", key_file="../key.pem")
    while True:
        time.sleep(1)

