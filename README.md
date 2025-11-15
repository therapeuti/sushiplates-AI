# 🍣 SushiPlate AI - 실시간 초밥접시 감지 감지 및 계산 시스템

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112.2-green.svg)](https://fastapi.tiangolo.com)
[![YOLO](https://img.shields.io/badge/YOLO-v8%2Fv9%2Fv11-orange.svg)](https://ultralytics.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Run-4285F4.svg)](https://cloud.google.com)

객체 탐지 모델 YOLO 기반 실시간 초밥접시 감지 및 가격 계산 웹 애플리케이션입니다. 카메라를 통해 초밥 접시를 자동으로 인식하고 가격을 계산함으로써 사람이 접시 하나하나 확인하여 계산하는 것보다 훨씬 빠른 시간 안에 계산을 할 수 있습니다.

## 🎯 해결하고자 하는 문제

### 문제점
초밥 식당에서는 손님이 먹은 접시를 직원이 일일이 하나하나 계산해야 하므로 **시간이 오래 걸립니다**.

### 기존 해결책의 한계
RFID를 이용한 자동 계산 시스템도 있지만 **높은 도입 비용**이 발생합니다.

### 우리의 솔루션
**AI 비전 기술**을 활용하여 빠르고, 정확하고, 간편하게 접시를 자동으로 인식하고 계산하는 시스템을 구축합니다.

## ✨ 주요 기능

### 🎯 실시간 객체 감지
- **다중 YOLO 모델 지원**: YOLOv8n/s/m, YOLOv9s, YOLO11s 선택 가능
- **실시간 추적**: 객체 ID 기반 지속적 추적
- **정확도 조절**: 신뢰도 임계값 0.6으로 정밀한 감지
- **FPS 제어**: 1-30 FPS 범위에서 처리 속도 조절

### 🔄 WebSocket 실시간 통신
- **양방향 통신**: 클라이언트-서버 간 실시간 데이터 전송
- **자동 재연결**: 연결 끊김 시 자동 복구
- **상태 모니터링**: 실시간 연결 상태 표시
- **Base64 이미지 전송**: 최적화된 이미지 데이터 전송

### 📊 스마트 분석
- **3초 정밀 감지**: 짧은 시간 내 여러 프레임 분석으로 정확도 향상
- **노이즈 필터링**: 일시적 오감지 제거
- **라벨 일관성 검증**: 최빈값과 신뢰도 기반 최종 판정
- **추적 지속성**: 객체 ID 기반 안정적 추적

### 💰 가격 계산 시스템
- **10가지 접시 타입**: black(₩10,000) ~ yellow-rec(₩1,000)
- **실시간 합계**: 자동 가격 계산 및 표시
- **수동 조정**: 접시 추가/삭제/수정 기능
- **통화 형식**: 원화 표시 및 천 단위 구분자

### 🎨 모던 UI/UX
- **반응형 디자인**: 모바일/태블릿/데스크톱 최적화
- **Material Design**: 직관적이고 깔끔한 인터페이스
- **실시간 피드백**: 로딩 상태, 연결 상태 시각화
- **애니메이션**: 부드러운 상호작용 효과

## 🛠 기술 스택

### Backend
- **FastAPI**: 고성능 비동기 웹 프레임워크
- **Uvicorn**: ASGI 서버
- **WebSocket**: 실시간 양방향 통신
- **ThreadPoolExecutor**: 멀티스레딩 이미지 처리

### AI/ML
- **Ultralytics YOLO**: 최신 객체 감지 모델
  - YOLOv8n/s/m, YOLOv9s, YOLO11s
- **OpenCV**: 컴퓨터 비전 라이브러리
- **PyTorch**: 딥러닝 프레임워크
- **NumPy**: 수치 계산

### Frontend
- **HTML5**: 시맨틱 마크업
- **CSS3**: 모던 스타일링
  - CSS Grid/Flexbox
  - CSS Variables
  - Animations & Transitions
- **JavaScript (ES6+)**: 동적 기능 구현
- **Canvas API**: 실시간 렌더링
- **MediaDevices API**: 웹캠 접근

### Infrastructure
- **Docker**: 컨테이너화
- **Google Cloud Run**: 서버리스 배포
- **Google Cloud Build**: CI/CD 파이프라인
- **Google Cloud Storage**: 모델 파일 저장

### Dependencies
```python
fastapi==0.112.2          # 웹 프레임워크
ultralytics==8.2.91       # YOLO 모델
opencv-python              # 컴퓨터 비전
torch==2.4.1              # 딥러닝
torchvision==0.19.1       # 비전 모델
numpy==1.26.4             # 수치 계산
pillow==10.4.0            # 이미지 처리
uvicorn==0.30.6           # ASGI 서버
websockets==13.0          # WebSocket 지원
```

## 🏗 아키텍처

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   Frontend      │◄──────────────►│   FastAPI       │
│                 │                │   Backend       │
│ • HTML5 Canvas  │                │                 │
│ • getUserMedia  │                │ • WebSocket     │
│ • JavaScript    │                │ • ThreadPool    │
└─────────────────┘                └─────────────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │   YOLO Models   │
                                   │                 │
                                   │ • Object Track  │
                                   │ • Classification│
                                   │ • Confidence    │
                                   └─────────────────┘
```

## 📱 사용법

### 초밥 접시 실시간 추론
1. **모델 선택**: 속도와 정확도에 따라 YOLO 모델 선택
2. **FPS 설정**: 처리 속도 조절 (1-30 FPS), 기본 5 FPS
3. **실시간 감지**: "실시간 감지 시작" 버튼 클릭
4. **3초 정밀 감지**: 더 정확한 결과를 위한 짧은 분석

### 검토
- **추론 결과 검토**: 카메라 화면에서 해당 접시 옆에 추론 결과를 접시 색상으로 표시함으로써 결과를 한 눈에 확인 가능
- **수동 조정**: 감지 결과 검토 후 직접 수정 가능. 쌓인 접시 순서대로 표시되며 각 칸마다 접시 추가/삭제 가능
- **실시간 모니터링**: 연결 상태 및 처리 상태 확인

## 🌐 배포

### Google Cloud Run
프로젝트는 Google Cloud Run에 자동 배포되도록 설정되어 있음.

1. **Cloud Build 설정**
```yaml
# cloudbuild.yaml이 이미 구성됨
# GCS에서 모델 파일 자동 다운로드
# Docker 이미지 빌드 및 배포
```

## 📈 성능

### 처리 속도 및 정확도
- **YOLOv8n**: ~30 FPS (가장 빠르나 정확도가 다소 떨어짐. 접시 색상이 다르거나 일부 접시를 추론하지 못 하는 경우가 있음)
- **YOLOv8s**: ~20 FPS (실시간 추론 시 딜레이가 적으면서 결과 수정이 필요없을 정도로 정확도가 꽤 높음.)
- **YOLOv8m**: ~5 FPS (추론 결과가 정확한 편이나 추론 속도가 다소 느림.)
- **YOLO11s**: ~25 FPS (최신)

### 정확도
- **단일 프레임**: ~85% 정확도
- **3초 정밀 감지**: ~95% 정확도
- **노이즈 필터링**: 오감지 90% 감소


- [Ultralytics](https://ultralytics.com) - YOLO 모델 제공
- [FastAPI](https://fastapi.tiangolo.com) - 훌륭한 웹 프레임워크
- [Google Cloud](https://cloud.google.com) - 클라우드 인프라
