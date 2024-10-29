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

# 로깅 설정. INFO 레벨의 로그를 기록
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLO 모델 초기화. {모델 이름: 모델 경로}
models: Dict[str, YOLO] = {}
model_paths = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m_v3.pt",
    "yolov9s": "sushi9or11s.pt",
    "yolo11s": "yolo11s-1010.pt"
}


# 모델 로딩을 위한 비동기 컨텍스트 매니저. 애플리케이션의 생명주기 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 모델 로딩
    for model_name, model_path in model_paths.items():
        try:
            if os.path.exists(model_path):
                models[model_name] = YOLO(model_path)
                logger.info(f"Successfully loaded model: {model_name}") # 모델 로드 성공하면 로그 기록
            else:
                logger.error(f"Model file not found: {model_path}") # 모델 파일이 없으면 오류 로그 기록
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}") # 예외처리로 로딩 되지 않는 모델 오류 로그 기록
    yield # 모델 로딩 완료되면 애플리케이션 시작
    
    models.clear() # 애플리케이션 종료 시 로드된 모델 정리


app = FastAPI(lifespan=lifespan) # 애플리케이션 생성, 생명주기 컨텍스트 매니저 설정(모델 로드)


# WebSocket 연결 관리
class ConnectionManager:
    def __init__(self):  # 활성 웹소켓 연결을 저장할 집합 초기화
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket): # 웹소켓 연결하는 비동기 메서드
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected: {id(websocket)}") # 웹소켓 연결 수락하고, 활성 연결 집합에 추가하고, 클라이언트 ID를 로그로 남김.

    def disconnect(self, websocket: WebSocket): # 웹소켓 연결 해제 메서드
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected: {id(websocket)}") # 웹소켓 해제 시, 활성 연결 집합에서 웹소켓 제거하고, 해제된 클라이언트 ID를 로그로 남김

    async def send_inference_results(self, websocket: WebSocket, results: dict): # 추론 결과를 웹소켓을 통해 클라이언트로 전송하는 비동기 메서드
        try:
            await websocket.send_json(results) # 추론 결과를 JSON 형식으로 전송
        except Exception as e:
            logger.error(f"Error sending results: {str(e)}") # 결과 전송 실패 시 예외 처리 


manager = ConnectionManager() # 인스턴스 생성

# 최대 8개 워커를 가진 스레드풀 생성. 이미지 처리를 위한 비동기 작업 수행에 사용됨.
executor = ThreadPoolExecutor(max_workers=8)


def process_image(image_data: str, model_name: str) -> dict: # 클라이언트로부터 이미지와 모델 이름 정보 받아서 객체 탐지 모델에 입력 하여 결과 반환하는 함수
    try:
        # 이미지 디코딩
        image = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8) # BASE64로 인코딩된 이미지 데이터를 디코딩하여 Numpy 배열로 변환
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR) # Numpy 배열을 OpenCV 형식의 이미지로 변환
        # image_size = len(image) # 전송된 이미지 사이즈 확인 위함

        if frame is None:
            raise ValueError("Invalid image data")  # 이미지가 유효하지 않으면 오류 발생시킴

        model = models.get(model_name) # 전달받은 모델 이름에 해당하는 모델 객체 가져옴. 모델이 없으면 오류 발생시킴.
        if model is None:
            raise ValueError(f"Model {model_name} not available")

        # height, width, _ = frame.shape  # 이미지의 높이와 너비 가져오기
        # print(f"Image size (bytes): {image_size}, Width: {width}, Height: {height}")

        # 모델에 맞는 입력 크기로 이미지 리사이즈
        # input_size = (640, 640)  # 예: YOLOv5는 일반적으로 640x640 입력 크기를 사용
        # resized_frame = cv2.resize(frame, input_size)

        results = model.track(frame, save=False, persist=True, conf=0.6, verbose=False) #track()과 persist=True로 객체 추적

        inference_results = [] # 추론 결과 저장할 리스트 초기화

        if results: # 결과 존재 시 결과 박스에 저장된 객체 id와 라벨, 신뢰도, 좌표를 추출하여 딕셔너리 형태로 inference_results에 저장
            print("결과 있음")
            for box in results[0].boxes:
                track_id = int(box.id) if box.id is not None else None
                class_id = int(box.cls)
                label = model.names[class_id]
                confidence = box.conf.item()
                coords = box.xyxy[0].cpu().numpy().tolist()

                inference_results.append({
                    "track_id": track_id,
                    "label": label,
                    "confidence": confidence,
                    "coords": coords,
                })

        return {"results": inference_results} # 최종 결과 반환

    except Exception as e: # 예외 발생시 로그에 기록하고 오류 메시지 반환
        logger.error(f"Error during inference: {str(e)}")
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse) # 루트 경로에 대한 get 요청 처리하는 비동기 함수
async def get_index():
    with open("template/index-fast.html", "r", encoding="utf-8") as f:  # html 파일을 읽어서 클라이언트에 반환
        return f.read()


@app.websocket("/ws") # 웹소켓 경로 정의. /ws로 접속하면 아래의 비동기 함수 호출
async def websocket_endpoint(websocket: WebSocket): # WebSocket객체를 인자로 받아 웹소켓 연결 처리하는 비동기 함수. 
    await manager.connect(websocket) # connect 메서드 호출하여 웹소켓 연결 수립
    try:
        while True:
            data = await websocket.receive_json() # 클라이언트로부터 json 형식의 데이터 수신

            image_data = data.get('image')
            model_name = data.get('model')  # 수신한 데이터로부터 이미지 데이터와 모델 정보 추출

            if not image_data or not model_name:  # 이미지 데이터나 모델 이름이 없으면 오류 메시지 전송하고 다음 루프로 넘어감
                await websocket.send_json({"error": "Missing required data"})
                continue

            if model_name not in models: # 요청한 모델이 존재하지 않으면 오류 메시지 전송하고 다음 루프로 넘어감
                await websocket.send_json({"error": f"Model {model_name} not available"})
                continue

            print("received image")
            # 이미지 처리를 비동기로 실행
            results = await asyncio.to_thread(process_image, image_data, model_name)
            # send_inference_results 메서드를 호출하여 추론 결과를 클라이언트에 전송
            await manager.send_inference_results(websocket, results)
            print("sended results")

    except WebSocketDisconnect: # 웹소켓 연결 끊어졌을때 예외처리
        manager.disconnect(websocket) # disconnect 메서드 호출
    except Exception as e:
        logger.error(f"Unexpected error in websocket connection: {str(e)}") # 예외 발생시 오류 메시지를 로그에 기록
        manager.disconnect(websocket) # disconnect 메서드 호출


if __name__ == "__main__":
    import uvicorn  # ASGI서버로 FastAPI 애플리케이션 실행하는데 사용

    port = int(os.environ.get('PORT', 8080)) # 환경변수에서 포트 번호 가져오고, 없으면 기본값으로 8080 사용
    uvicorn.run(app, host="0.0.0.0", port=port) # uvicorn을 사용하여 FastAPI 애플리케이션 실행. 모든 IP주소에서 접근 가능하도록 호스트를 0.0.0.0으로 설정. port에서 서버 실행