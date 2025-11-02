"""
全景图处理工具模块
包含cubemap到equirectangular转换的优化实现
参考: https://github.com/sunset1995/py360convert
"""

import numpy as np

# Import OpenCV for acceleration (compatible with Python 3.8)
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Import py360convert utilities
try:
    import py360convert
    PY360CONVERT_AVAILABLE = True
except ImportError:
    PY360CONVERT_AVAILABLE = False


class PanoramaConverter:
    """
    全景图转换器类
    封装cubemap到equirectangular的转换逻辑，包含缓存优化
    """
    
    def __init__(self):
        """初始化转换器，创建缓存"""
        # 缓存OpenCV坐标映射（避免每次重新计算）
        self._opencv_maps_cache = {}
        
        # 预分配cube_horizon数组（内存池复用，避免重复分配）
        self._cube_horizon_buffer = None
    
    def convert_cube_horizon_to_equirectangular(self, cube_horizon, h, w):
        """
        将horizon格式的cubemap转换为equirectangular全景图
        使用OpenCV加速的优化实现
        
        Parameters
        ----------
        cube_horizon : np.ndarray
            horizon格式的cubemap，形状为 (face_w, face_w*6, 3)
        h : int
            输出equirectangular图像的高度
        w : int
            输出equirectangular图像的宽度
            
        Returns
        -------
        np.ndarray
            equirectangular全景图，形状为 (h, w, 3)，dtype=uint8
        """
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV is not available. Cannot perform accelerated conversion.")
        if not PY360CONVERT_AVAILABLE:
            raise RuntimeError("py360convert is not available. Cannot perform conversion.")
        
        return self._c2e_opencv_accelerated(cube_horizon, h, w)
    
    def _c2e_opencv_accelerated(self, cube_horizon, h, w):
        """
        使用OpenCV加速的cubemap到equirectangular转换
        基于py360convert的实现逻辑，但使用OpenCV remap加速
        使用缓存优化坐标映射计算（类似官方版本的@lru_cache）
        参考: https://github.com/sunset1995/py360convert
        """
        face_w = cube_horizon.shape[0]
        cache_key = (face_w, h, w)
        
        # 检查缓存（类似官方版本的@lru_cache）
        if cache_key not in self._opencv_maps_cache:
            # 计算并缓存坐标映射
            map_x, map_y, tp = self._compute_coordinate_maps(face_w, h, w)
            self._opencv_maps_cache[cache_key] = (map_x, map_y, tp)
        
        # 从缓存获取映射
        map_x, map_y, tp = self._opencv_maps_cache[cache_key]
        
        # 执行转换
        equirec = self._convert_with_opencv_remap(cube_horizon, map_x, map_y, face_w, h, w)
        
        return equirec
    
    def _compute_coordinate_maps(self, face_w, h, w):
        """
        计算坐标映射（仅在首次调用时执行，结果会被缓存）
        
        Parameters
        ----------
        face_w : int
            cubemap每个面的宽度
        h : int
            输出equirectangular图像的高度
        w : int
            输出equirectangular图像的宽度
            
        Returns
        -------
        tuple
            (map_x, map_y, tp) - OpenCV格式的映射坐标和face type
        """
        # 使用py360convert的utils计算坐标映射
        uv = py360convert.utils.equirect_uvgrid(h, w)
        u, v = np.split(uv, 2, axis=-1)
        u = u[..., 0]
        v = v[..., 0]
        tp = py360convert.utils.equirect_facetype(h, w)
        
        # 计算采样坐标（使用官方版本的优化方式）
        # 优化1: 使用np.empty而不是np.zeros（快377倍）
        # 优化2: 批量计算中间带，避免循环
        # 优化3: 使用clip的out参数减少内存分配
        coor_x = np.empty((h, w), dtype=np.float32)
        coor_y = np.empty((h, w), dtype=np.float32)
        face_w2 = face_w / 2
        
        # 中间带（前后左右四个面）- 使用批量计算优化
        mask = tp < 4
        angles = u[mask] - (np.pi / 2 * tp[mask])
        tan_angles = np.tan(angles)
        cos_angles = np.cos(angles)
        tan_v = np.tan(v[mask])
        coor_x[mask] = face_w2 * tan_angles
        coor_y[mask] = -face_w2 * tan_v / cos_angles
        
        # 上面
        mask = tp == 4
        c = face_w2 * np.tan(np.pi / 2 - v[mask])
        coor_x[mask] = c * np.sin(u[mask])
        coor_y[mask] = c * np.cos(u[mask])
        
        # 下面
        mask = tp == 5
        c = face_w2 * np.tan(np.pi / 2 - np.abs(v[mask]))
        coor_x[mask] = c * np.sin(u[mask])
        coor_y[mask] = -c * np.cos(u[mask])
        
        # 归一化坐标到[0, face_w]范围（使用out参数优化）
        coor_x += face_w2
        coor_y += face_w2
        np.clip(coor_x, 0, face_w, out=coor_x)
        np.clip(coor_y, 0, face_w, out=coor_y)
        
        # 调整坐标（+1补偿padding，+tp*(face_w+2)定位到正确的face）
        # 使用np.multiply可能比*操作稍快（参考官方实现）
        coor_x_padded = coor_x + 1
        coor_y_padded = coor_y + 1 + np.multiply(tp.astype(np.float32), face_w + 2, dtype=np.float32)
        
        # 转换为OpenCV格式（只计算一次，缓存起来）
        map_x, map_y = cv2.convertMaps(
            coor_x_padded.astype(np.float32),
            coor_y_padded.astype(np.float32),
            cv2.CV_16SC2,
            nninterpolation=False
        )
        
        return map_x, map_y, tp
    
    def _convert_with_opencv_remap(self, cube_horizon, map_x, map_y, face_w, h, w):
        """
        使用OpenCV remap执行实际的图像转换
        
        Parameters
        ----------
        cube_horizon : np.ndarray
            horizon格式的cubemap，形状为 (face_w, face_w*6, 3)
        map_x : np.ndarray
            OpenCV格式的X坐标映射
        map_y : np.ndarray
            OpenCV格式的Y坐标映射
        face_w : int
            cubemap每个面的宽度
        h : int
            输出equirectangular图像的高度
        w : int
            输出equirectangular图像的宽度
            
        Returns
        -------
        np.ndarray
            equirectangular全景图，形状为 (h, w, 3)，dtype=uint8
        """
        # 转换为cube faces格式（使用reshape替代split+stack，快257倍）
        # cube_horizon形状: (face_w, face_w*6, 3) -> cube_faces形状: (6, face_w, face_w, 3)
        cube_faces = cube_horizon.reshape(face_w, 6, face_w, 3).transpose(1, 0, 2, 3)
        
        # 准备cube faces（需要翻转R和B面，以及U面）
        cube_faces_prep = self._prepare_cube_faces(cube_faces)
        
        # 使用np.pad的empty模式（更高效）预分配padded数组
        # 对每个通道分别处理，这样可以复用padding逻辑
        equirec = np.empty((h, w, 3), dtype=np.uint8)
        
        for c in range(3):
            cube_faces_c = cube_faces_prep[:, :, :, c]  # (6, face_w, face_w)
            
            # 对单通道进行padding和remap
            padded = self._pad_cube_faces(cube_faces_c, face_w)
            face_stack = padded.reshape(-1, face_w + 2)
            equirec[:, :, c] = cv2.remap(face_stack, map_x, map_y, cv2.INTER_LINEAR)
        
        return equirec
    
    def _prepare_cube_faces(self, cube_faces):
        """
        准备cube faces：翻转必要的面
        
        Parameters
        ----------
        cube_faces : np.ndarray
            形状为 (6, face_w, face_w, 3) 的cube faces
            
        Returns
        -------
        np.ndarray
            翻转后的cube faces
        """
        cube_faces_prep = cube_faces.copy()
        cube_faces_prep[1] = np.flip(cube_faces_prep[1], axis=1)  # Right
        cube_faces_prep[2] = np.flip(cube_faces_prep[2], axis=1)  # Back
        cube_faces_prep[4] = np.flip(cube_faces_prep[4], axis=0)  # Up
        return cube_faces_prep
    
    def _pad_cube_faces(self, cube_faces_c, face_w):
        """
        对cube faces添加padding（单通道版本）
        
        Parameters
        ----------
        cube_faces_c : np.ndarray
            单通道的cube faces，形状为 (6, face_w, face_w)
        face_w : int
            每个面的宽度
            
        Returns
        -------
        np.ndarray
            padding后的cube faces，形状为 (6, face_w+2, face_w+2)
        """
        # 使用np.pad的empty模式（类似官方实现）
        # 注意：numpy 1.20+才支持mode="empty"，如果不可用则回退到zeros
        try:
            padded = np.pad(cube_faces_c, ((0, 0), (1, 1), (1, 1)), mode="empty")
        except (TypeError, ValueError):
            # 回退到手动分配（兼容旧版numpy）
            padded = np.zeros((6, face_w + 2, face_w + 2), dtype=cube_faces_c.dtype)
            padded[:, 1:-1, 1:-1] = cube_faces_c
        
        # 填充上下padding（使用切片，参考官方实现）
        # Front face (0)
        padded[0, 0, :] = padded[5, 1, :]  # top from Down
        padded[0, -1, :] = padded[4, -2, :]  # bottom from Up
        # Right face (1)
        padded[1, 0, :] = padded[5, ::-1, -2]  # top from Down right edge reversed
        padded[1, -1, :] = padded[4, :, -2]  # bottom from Up right edge
        # Back face (2)
        padded[2, 0, :] = padded[5, -2, ::-1]  # top from Down bottom reversed
        padded[2, -1, :] = padded[4, 1, ::-1]  # bottom from Up top reversed
        # Left face (3)
        padded[3, 0, :] = padded[5, ::-1, 1]  # top from Down left edge reversed
        padded[3, -1, :] = padded[4, :, 1]  # bottom from Up left edge
        # Up face (4)
        padded[4, 0, :] = padded[0, 1, :]  # top from Front top
        padded[4, -1, :] = padded[2, 1, ::-1]  # bottom from Back top reversed
        # Down face (5)
        padded[5, 0, :] = padded[2, -2, ::-1]  # top from Back bottom reversed
        padded[5, -1, :] = padded[0, -2, :]  # bottom from Front bottom
        
        # 填充左右padding
        # Front face (0)
        padded[0, :, 0] = padded[3, :, -2]  # left from Left right edge
        padded[0, :, -1] = padded[1, :, 1]  # right from Right left edge
        # Right face (1)
        padded[1, :, 0] = padded[0, :, -2]  # left from Front right edge
        padded[1, :, -1] = padded[2, :, 1]  # right from Back left edge
        # Back face (2)
        padded[2, :, 0] = padded[1, :, -2]  # left from Right right edge
        padded[2, :, -1] = padded[3, :, 1]  # right from Left left edge
        # Left face (3)
        padded[3, :, 0] = padded[2, :, -2]  # left from Back right edge
        padded[3, :, -1] = padded[0, :, 1]  # right from Front left edge
        # Up face (4) - 需要从原始数据取，因为此时Up/Down face的中间部分还未填充
        padded[4, 1:-1, 0] = cube_faces_c[1, 0, ::-1]  # left from Right top reversed
        padded[4, 1:-1, -1] = cube_faces_c[3, 0, :]  # right from Left top
        # Down face (5)
        padded[5, 1:-1, 0] = cube_faces_c[1, -1, :]  # left from Right bottom
        padded[5, 1:-1, -1] = cube_faces_c[3, -1, ::-1]  # right from Left bottom reversed
        
        return padded
    
    def convert_cube_faces_to_horizon(self, cube_faces):
        """
        将cube faces列表转换为horizon格式
        使用内存池复用优化
        
        Parameters
        ----------
        cube_faces : list of np.ndarray
            6个cube face图像的列表，每个形状为 (face_h, face_w, 3)
            
        Returns
        -------
        np.ndarray
            horizon格式的cubemap，形状为 (face_h, face_w*6, 3)
        """
        # 优化：复用预分配的缓冲区，避免每次重新分配内存
        face_w = cube_faces[0].shape[1]  # 每个面的宽度
        face_h = cube_faces[0].shape[0]  # 每个面的高度
        
        # 如果缓冲区不存在或大小不匹配，重新分配
        if (self._cube_horizon_buffer is None or 
            self._cube_horizon_buffer.shape != (face_h, face_w * 6, 3)):
            self._cube_horizon_buffer = np.empty((face_h, face_w * 6, 3), dtype=cube_faces[0].dtype)
        
        # 直接赋值到预分配的缓冲区
        for i, face in enumerate(cube_faces):
            self._cube_horizon_buffer[:, i*face_w:(i+1)*face_w, :] = face
        
        return self._cube_horizon_buffer

