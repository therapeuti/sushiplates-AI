from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np
import base64
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Set
from contextlib import asynccontextmanager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLO 모델 초기화
models: Dict[str, YOLO] = {}
model_paths = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m_v3.pt",
    "yolov9s": "sushi9or11s.pt",
    "yolo11s": "yolo11s-1010.pt"
}


# 모델 로딩을 위한 컨텍스트 매니저
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 모델 로딩
    for model_name, model_path in model_paths.items():
        try:
            if os.path.exists(model_path):
                models[model_name] = YOLO(model_path)
                logger.info(f"Successfully loaded model: {model_name}")
            else:
                logger.error(f"Model file not found: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
    yield
    # 종료 시 정리 작업
    models.clear()


app = FastAPI(lifespan=lifespan)

# 정적 파일 서빙 설정 - 현재 필요없음
# app.mount("/static", StaticFiles(directory="template"), name="static")


# WebSocket 연결 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected: {id(websocket)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected: {id(websocket)}")

    async def send_inference_results(self, websocket: WebSocket, results: dict):
        try:
            await websocket.send_json(results)
        except Exception as e:
            logger.error(f"Error sending results: {str(e)}")


manager = ConnectionManager()

# 스레드풀 생성
executor = ThreadPoolExecutor(max_workers=8)

def process_image(image_data: str, model_name: str) -> dict:
    try:
        # 이미지 디코딩
        image = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # image_size = len(image)

        if frame is None:
            raise ValueError("Invalid image data")

        model = models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not available")

        # height, width, _ = frame.shape  # 이미지의 높이와 너비 가져오기
        # print(f"Image size (bytes): {image_size}, Width: {width}, Height: {height}")

        results = model.predict(frame, save=False, conf=0.6, verbose=False)

        inference_results = []

        if results:
            for box in results[0].boxes:
                class_id = int(box.cls)
                label = model.names[class_id]
                confidence = box.conf.item()
                coords = box.xyxy[0].cpu().numpy().tolist()

                inference_results.append({
                    "label": label,
                    "confidence": confidence,
                    "coords": coords,
                })

        return {"results": inference_results}

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("template/index-fast.html", "r", encoding="utf-8") as f:
        return f.read()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()

            image_data = data.get('image')
            model_name = data.get('model')

            if not image_data or not model_name:
                await websocket.send_json({"error": "Missing required data"})
                continue

            if model_name not in models:
                await websocket.send_json({"error": f"Model {model_name} not available"})
                continue

            # 이미지 처리를 비동기로 실행

            results = await asyncio.to_thread(process_image, image_data, model_name)

            # results = await process_image(image_data, model_name)
            await manager.send_inference_results(websocket, results)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Unexpected error in websocket connection: {str(e)}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)