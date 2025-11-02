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

class OpenTeleVision:
    def __init__(self, img_shape, shm_name, queue, toggle_streaming, stream_mode="image", cert_file="./cert.pem", key_file="./key.pem", ngrok=False, panorama_mode=True, print_freq=False):
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

        if ngrok:
            self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if stream_mode == "image":
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
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

    async def on_cam_move(self, event, session, fps=60):
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        try:
            self.left_hand_shared[:] = event.value["leftHand"]
            self.right_hand_shared[:] = event.value["rightHand"]
            self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
            self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
        except: 
            pass
    
    async def main_webrtc(self, session, fps=60):
        session.set @ DefaultScene(frameloop="always")
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
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
        # 初始化场景和手部追踪
        session.set @ DefaultScene(Sphere(
            args=[1, 32, 32],
            materialType="standard",
            material={"map": "", "side": 1},
            position=[0, 0, 0],
            rotation=[0, 0.5 * np.pi, 0],  # 绕Y轴旋转90度进行坐标系转换
        ))
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        
        frame_count = 0
        while True:
            start = time.time()
            display_image = self.img_array.copy()
            
            # 将numpy数组转换为JPEG格式的base64字符串
            pil_img = Image.fromarray(display_image, 'RGB')
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            img_data_url = f"data:image/jpeg;base64,{img_base64}"
            
            # 更新Sphere材质
            session.upsert @ Sphere(
                args=[1, 32, 32],
                materialType="standard",
                material={"map": img_data_url, "side": 1},
                position=[0, 0, 0],
                rotation=[0, 0.5 * np.pi, 0],
                key="skyball"
            )
            
            frame_count += 1
            if self.print_freq and frame_count % 60 == 0:
                print(f'Panorama FPS: {60 / (time.time() - start):.2f}')
            
            await asyncio.sleep(0.016)  # ~60fps
    
    async def main_image(self, session, fps=60):
        """旧的立体相机显示模式"""
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
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
            await asyncio.sleep(0.03)

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

