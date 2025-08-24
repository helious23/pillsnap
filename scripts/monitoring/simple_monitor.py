#!/usr/bin/env python3
"""
간단한 실시간 학습 모니터링 스크립트
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

class SimpleMonitor:
    def __init__(self):
        self.connected_clients: List[WebSocket] = []
        self.log_file = "/tmp/stage3_bash23_output.log"
        self.last_position = 0
        
    async def broadcast(self, message: str):
        """모든 연결된 클라이언트에게 메시지 전송"""
        if not self.connected_clients:
            return
            
        disconnect_clients = []
        for client in self.connected_clients:
            try:
                await client.send_text(message)
            except:
                disconnect_clients.append(client)
        
        for client in disconnect_clients:
            if client in self.connected_clients:
                self.connected_clients.remove(client)
    
    async def monitor_log(self):
        """로그 파일 모니터링"""
        while True:
            try:
                if Path(self.log_file).exists():
                    with open(self.log_file, 'r') as f:
                        f.seek(self.last_position)
                        new_lines = f.readlines()
                        self.last_position = f.tell()
                        
                        for line in new_lines:
                            if line.strip():
                                kst = timezone(timedelta(hours=9))
                                timestamp = datetime.now(kst).strftime('%H:%M:%S')
                                message = json.dumps({
                                    "timestamp": timestamp,
                                    "message": line.strip()
                                })
                                await self.broadcast(message)
            except Exception as e:
                print(f"Error reading log: {e}")
                
            await asyncio.sleep(1)

monitor = SimpleMonitor()

@app.get("/")
async def get():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Stage 3 Training Monitor</title>
    <style>
        body { 
            font-family: monospace; 
            background: #1a1a1a; 
            color: #0f0; 
            margin: 20px;
        }
        #logs {
            height: 80vh;
            overflow-y: auto;
            border: 1px solid #0f0;
            padding: 10px;
            background: #000;
        }
        .log-line {
            margin: 2px 0;
            white-space: pre-wrap;
        }
        .info { color: #0f0; }
        .error { color: #f00; }
        .warning { color: #ff0; }
    </style>
</head>
<body>
    <h1>🚀 Stage 3 Training Monitor</h1>
    <div id="status">연결 중...</div>
    <div id="logs"></div>
    
    <script>
        const ws = new WebSocket('ws://localhost:8000/ws');
        const logsDiv = document.getElementById('logs');
        const statusDiv = document.getElementById('status');
        
        ws.onopen = () => {
            statusDiv.textContent = '✅ 연결됨';
        };
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const logLine = document.createElement('div');
                logLine.className = 'log-line info';
                logLine.textContent = `[${data.timestamp}] ${data.message}`;
                logsDiv.appendChild(logLine);
                
                // 자동 스크롤
                logsDiv.scrollTop = logsDiv.scrollHeight;
                
                // 로그가 너무 많으면 오래된 것 제거
                if (logsDiv.children.length > 1000) {
                    logsDiv.removeChild(logsDiv.firstChild);
                }
            } catch(e) {
                console.error('Parse error:', e);
            }
        };
        
        ws.onerror = (error) => {
            statusDiv.textContent = '❌ 연결 오류';
            console.error('WebSocket error:', error);
        };
        
        ws.onclose = () => {
            statusDiv.textContent = '⚠️ 연결 끊김';
        };
    </script>
</body>
</html>
    """)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    monitor.connected_clients.append(websocket)
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        monitor.connected_clients.remove(websocket)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor.monitor_log())

if __name__ == "__main__":
    print(f"📝 모니터링 로그 파일: {monitor.log_file}")
    print(f"🌐 웹 인터페이스: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)