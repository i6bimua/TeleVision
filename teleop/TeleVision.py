import time
from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import ImageBackground, group, Hands, WebRTCStereoVideoPlane, DefaultScene
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
import numpy as np
import asyncio
from webrtc.zed_server import *
import msgpack

class OpenTeleVision:
    def __init__(self, img_shape, shm_name, queue, toggle_streaming, stream_mode="image", cert_file="./cert.pem", key_file="./key.pem", ngrok=False, meta_shm_name=None):
        # self.app=Vuer()
        self.img_shape = (img_shape[0], 2*img_shape[1], 3)
        self.img_height, self.img_width = img_shape[:2]

        if ngrok:
            self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if stream_mode == "image":
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
            # 帧元数据共享（可选）：int64 frame_id + float64 t0 + float64 hand_move_receive_time + float64 websocket_send_complete_time
            self.meta_shm = None
            self.meta_buf = None
            if meta_shm_name is not None:
                try:
                    self.meta_shm = shared_memory.SharedMemory(name=meta_shm_name)
                    self.meta_buf = self.meta_shm.buf
                except Exception as e:
                    print(f"Warning: meta_shm open failed: {e}")
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
        """
        # only intercept the ego camera.
        # if event.key != "ego":
        #     return
        try:
            # 调试：检查事件数据结构
            if not hasattr(event, 'value') or event.value is None:
                print("Warning: CAMERA_MOVE event has no value")
                return
            
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
                    
                    # 赋值到共享内存
                    if matrix is not None:
                        if matrix.size == 16:  # 4x4矩阵
                            self.head_matrix_shared[:] = matrix
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
                if "aspect" in event.value:
                    self.aspect_shared.value = float(event.value["aspect"])
            else:
                print(f"Warning: CAMERA_MOVE event.value is not a dict, type: {type(event.value)}")
        except Exception as e:
            print(f"Error in on_cam_move: {e}")
            import traceback
            traceback.print_exc()
        # self.head_matrix = np.array(event.value["camera"]["matrix"]).reshape(4, 4, order="F")
        # print(np.array(event.value["camera"]["matrix"]).reshape(4, 4, order="F"))
        # print("camera moved", event.value["matrix"].shape, event.value["matrix"])

    async def on_hand_move(self, event, session):
        # 记录事件接收时间（用于计算闭环延迟）
        event_receive_time = time.time()
        
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
        
        try:
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
                
                # 检查状态：如果手部没有被检测到，数据可能是空的，这是正常情况
                left_detected = True
                right_detected = True
                if "leftState" in event.value and event.value["leftState"]:
                    # 如果 leftState 有内容，可以检查是否有检测标志
                    # 但目前没有明确的检测标志，所以假设有数据就检测到了
                    pass
                if "rightState" in event.value and event.value["rightState"]:
                    pass
                
                # 转换为numpy数组
                # 处理msgpack ExtType（二进制数据）
                if isinstance(left_raw, msgpack.ExtType):
                    try:
                        # ExtType包含二进制数据，需要从buffer中读取
                        buffer_size = len(left_raw.data)
                        if not hasattr(self, '_exttype_buffer_logged'):
                            print(f"\n--- ExtType 数据处理 ---")
                            print(f"left_raw ExtType code: {left_raw.code}, buffer_size: {buffer_size}")
                            print(f"leftState: {event.value.get('leftState', {})}")
                            print(f"rightState: {event.value.get('rightState', {})}")
                            try:
                                preview = left_raw.data[:20] if buffer_size >= 20 else left_raw.data
                                print(f"left_raw.data type: {type(left_raw.data)}, first 20 bytes: {preview}")
                            except Exception as e:
                                print(f"left_raw.data type: {type(left_raw.data)}, error previewing: {e}")
                            self._exttype_buffer_logged = True
                        
                        if buffer_size < 4:
                            # 缓冲区太小，可能是手部未检测到（正常情况），静默跳过
                            # 只在第一次遇到时记录，避免刷屏
                            if not hasattr(self, '_empty_buffer_logged'):
                                print(f"Info: Hand tracking data not ready yet (buffer_size={buffer_size} bytes), waiting for valid data...")
                                self._empty_buffer_logged = True
                            return
                        else:
                            element_size = np.dtype(np.float32).itemsize
                            if buffer_size % element_size != 0:
                                buffer_size = (buffer_size // element_size) * element_size
                                if buffer_size == 0:
                                    print(f"Warning: left_raw buffer size alignment failed, skipping")
                                    return
                                left_arr = np.frombuffer(left_raw.data[:buffer_size], dtype=np.float32).copy()
                            else:
                                left_arr = np.frombuffer(left_raw.data, dtype=np.float32).copy()
                    except Exception as e:
                        print(f"Error processing left_raw ExtType: {e}")
                        import traceback
                        traceback.print_exc()
                        return
                elif isinstance(left_raw, bytes):
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
                    try:
                        left_arr = np.array(left_raw, dtype=float)
                        if left_arr.size == 0:
                            return
                    except (ValueError, TypeError):
                        return
                
                if isinstance(right_raw, msgpack.ExtType):
                    try:
                        buffer_size = len(right_raw.data)
                        if not hasattr(self, '_exttype_buffer_logged'):
                            print(f"right_raw ExtType code: {right_raw.code}, buffer_size: {buffer_size}")
                            try:
                                preview = right_raw.data[:20] if buffer_size >= 20 else right_raw.data
                                print(f"right_raw.data type: {type(right_raw.data)}, first 20 bytes: {preview}")
                            except Exception as e:
                                print(f"right_raw.data type: {type(right_raw.data)}, error previewing: {e}")
                            self._exttype_buffer_logged = True
                        
                        if buffer_size < 4:
                            # 缓冲区太小，可能是手部未检测到（正常情况），静默跳过
                            # 只在第一次遇到时记录，避免刷屏
                            if not hasattr(self, '_empty_buffer_logged'):
                                print(f"Info: Hand tracking data not ready yet (buffer_size={buffer_size} bytes), waiting for valid data...")
                                self._empty_buffer_logged = True
                            return
                        else:
                            element_size = np.dtype(np.float32).itemsize
                            if buffer_size % element_size != 0:
                                buffer_size = (buffer_size // element_size) * element_size
                                if buffer_size == 0:
                                    print(f"Warning: right_raw buffer size alignment failed, skipping")
                                    return
                                right_arr = np.frombuffer(right_raw.data[:buffer_size], dtype=np.float32).copy()
                            else:
                                right_arr = np.frombuffer(right_raw.data, dtype=np.float32).copy()
                    except Exception as e:
                        print(f"Error processing right_raw ExtType: {e}")
                        import traceback
                        traceback.print_exc()
                        return
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
                if not hasattr(self, '_hand_data_size_logged'):
                    print(f"\n--- 矩阵提取逻辑验证 ---")
                    print(f"left_arr.size: {left_arr.size}, right_arr.size: {right_arr.size}")
                    print(f"left_arr.dtype: {left_arr.dtype}, right_arr.dtype: {right_arr.dtype}")
                    self._hand_data_size_logged = True
                
                if left_arr.size == 400:  # 25个4x4矩阵
                    # 提取手腕矩阵（第一个矩阵，前16个元素）
                    left_data = left_arr[:16].copy()
                elif left_arr.size == 16:  # 直接是单个4x4矩阵
                    left_data = left_arr.copy()
                elif left_arr.size == 0:
                    print(f"Warning: left_arr is empty, skipping this frame")
                    return
                else:
                    print(f"Warning: Unexpected left hand data size: {left_arr.size}, expected 16 or 400")
                    print(f"  left_arr shape: {left_arr.shape}, dtype: {left_arr.dtype}")
                    return
                
                if right_arr.size == 400:  # 25个4x4矩阵
                    # 提取手腕矩阵（第一个矩阵，前16个元素）
                    right_data = right_arr[:16].copy()
                elif right_arr.size == 16:  # 直接是单个4x4矩阵
                    right_data = right_arr.copy()
                elif right_arr.size == 0:
                    print(f"Warning: right_arr is empty, skipping this frame")
                    return
                else:
                    print(f"Warning: Unexpected right hand data size: {right_arr.size}, expected 16 or 400")
                    print(f"  right_arr shape: {right_arr.shape}, dtype: {right_arr.dtype}")
                    return
                
                # 赋值到共享内存（必须是16个元素）
                self.left_hand_shared[:] = left_data
                self.right_hand_shared[:] = right_data
                
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
                    self.left_landmarks_shared[:] = left_landmarks
                
                if right_arr.size == 400:
                    right_landmarks = np.zeros(75, dtype=float)
                    for i in range(25):
                        matrix_start = i * 16
                        right_landmarks[i*3:(i+1)*3] = right_arr[matrix_start+12:matrix_start+15]
                    self.right_landmarks_shared[:] = right_landmarks
                
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
                
                if len(left_data) == 16:
                    self.left_hand_shared[:] = left_data
                if len(right_data) == 16:
                    self.right_hand_shared[:] = right_data
                
                if "leftLandmarks" in event.value and "rightLandmarks" in event.value:
                    left_landmarks = np.array(event.value["leftLandmarks"], dtype=float).flatten()
                    right_landmarks = np.array(event.value["rightLandmarks"], dtype=float).flatten()
                    if len(left_landmarks) == 75:
                        self.left_landmarks_shared[:] = left_landmarks
                    if len(right_landmarks) == 75:
                        self.right_landmarks_shared[:] = right_landmarks
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
    
    async def main_image(self, session, fps=60):
        session.upsert @ Hands(stream=True, key="hands", hideLeft=True, hideRight=True)
        end_time = time.time()
        while True:
            start = time.time()
            # print(end_time - start)
            # aspect = self.aspect_shared.value
            display_image = self.img_array

            # session.upsert(
            # ImageBackground(
            #     # Can scale the images down.
            #     display_image[:self.img_height],
            #     # 'jpg' encoding is significantly faster than 'png'.
            #     format="jpeg",
            #     quality=80,
            #     key="left-image",
            #     interpolate=True,
            #     # fixed=True,
            #     aspect=1.778,
            #     distanceToCamera=2,
            #     position=[0, -0.5, -2],
            #     rotation=[0, 0, 0],
            # ),
            # to="bgChildren",
            # )

            try:
                # 记录WebSocket发送开始时间
                t_websocket_start = time.time()
                
                session.upsert(
                [ImageBackground(
                    # Can scale the images down.
                    display_image[::2, :self.img_width],
                    # display_image[:self.img_height:2, ::2],
                    # 'jpg' encoding is significantly faster than 'png'.
                    format="jpeg",
                    quality=80,
                    key="left-image",
                    interpolate=True,
                    # fixed=True,
                    aspect=1.66667,
                    # distanceToCamera=0.5,
                    height = 8,
                    position=[0, -1, 3],
                    # rotation=[0, 0, 0],
                    layers=1, 
                    alphaSrc="./vinette.jpg"
                ),
                ImageBackground(
                    # Can scale the images down.
                    display_image[::2, self.img_width:],
                    # display_image[self.img_height::2, ::2],
                    # 'jpg' encoding is significantly faster than 'png'.
                    format="jpeg",
                    quality=80,
                    key="right-image",
                    interpolate=True,
                    # fixed=True,
                    aspect=1.66667,
                    # distanceToCamera=0.5,
                    height = 8,
                    position=[0, -1, 3],
                    # rotation=[0, 0, 0],
                    layers=2, 
                    alphaSrc="./vinette.jpg"
                )],
                to="bgChildren",
                )
                
                # 记录WebSocket发送完成时间（闭环延迟的终点）
                t_websocket_end = time.time()
                # 将WebSocket发送完成时间写入共享内存
                if hasattr(self, 'meta_buf') and self.meta_buf is not None:
                    try:
                        mv = memoryview(self.meta_buf)
                        # 在meta_shm的24-32字节位置存储WebSocket发送完成时间（float64）
                        # 使用numpy数组方式正确写入共享内存
                        arr = np.ndarray(1, dtype=np.float64, buffer=mv[24:32])
                        arr[0] = t_websocket_end
                    except Exception:
                        pass
            except AssertionError as e:
                # WebSocket连接已断开，退出循环
                print(f"❌ WebSocket connection lost: {e}")
                print("   退出 main_image 循环")
                break
            except Exception as e:
                # 其他错误，记录但不退出
                print(f"⚠️  Error updating images: {e}")
                import traceback
                traceback.print_exc()
            
            # rest_time = 1/fps - time.time() + start
            end_time = time.time()
            await asyncio.sleep(0.03)

    @property
    def left_hand(self):
        # with self.left_hand_shared.get_lock():
        #     return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def right_hand(self):
        # with self.right_hand_shared.get_lock():
        #     return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def left_landmarks(self):
        # with self.left_landmarks_shared.get_lock():
        #     return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    
    @property
    def right_landmarks(self):
        # with self.right_landmarks_shared.get_lock():
            # return np.array(self.right_landmarks_shared[:]).reshape(25, 3)
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        # with self.head_matrix_shared.get_lock():
        #     return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        # with self.aspect_shared.get_lock():
            # return float(self.aspect_shared.value)
        return float(self.aspect_shared.value)

    
if __name__ == "__main__":
    resolution = (720, 1280)
    crop_size_w = 340  # (resolution[1] - resolution[0]) // 2
    crop_size_h = 270
    resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)  # 450 * 600
    img_shape = (2 * resolution_cropped[0], resolution_cropped[1], 3)  # 900 * 600
    img_height, img_width = resolution_cropped[:2]  # 450 * 600
    shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    shm_name = shm.name
    img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)

    tv = OpenTeleVision(resolution_cropped, cert_file="../cert.pem", key_file="../key.pem")
    while True:
        # print(tv.left_landmarks)
        # print(tv.left_hand)
        # tv.modify_shared_image(random=True)
        time.sleep(1)
