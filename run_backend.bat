@echo off
cd D:\deepl\backend
echo PyTorch CNN 시각화 도구 - 백엔드 서버 실행 중...
echo.
echo 필요한 패키지 설치 중...
pip install -r requirements.txt
echo.
echo 백엔드 서버 실행 중... (종료하려면 Ctrl+C를 누르세요)
python main.py
