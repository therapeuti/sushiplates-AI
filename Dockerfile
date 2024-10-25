# 잘 동작하던 v2 이미지를 기반으로 사용
FROM gcr.io/sunlit-context-430703-g8/yolo-app:v2

# 작업 디렉토리 설정
WORKDIR /app

# 새로운 소스 코드만 복사
COPY . .

# PORT 환경변수 설정
ENV PORT=8080

# 포트 노출
EXPOSE 8080

# 실행
CMD ["python", "app-fast.py"]