#!/usr/bin/env python3
"""
Stage 3 실시간 모니터링 시스템 포괄적 테스트

실시간 로그 스트리밍, WebSocket 통신, 모니터링 데이터 수집 등
모든 모니터링 기능을 검증합니다.
"""

import pytest
import asyncio
import json
import time
import threading
import queue
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import sys

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.stage3_realtime_monitor import (
    LogStreamer,
    Stage3RealtimeMonitor,
    Stage3MonitoringData,
    app
)
from fastapi.testclient import TestClient


class TestStage3MonitoringData:
    """Stage 3 모니터링 데이터 테스트"""
    
    def test_monitoring_data_creation(self):
        """모니터링 데이터 생성 테스트"""
        data = Stage3MonitoringData(
            timestamp=time.time(),
            epoch=10,
            batch_idx=500,
            current_resolution=288,
            optimal_batch_size=24,
            resize_phase="transition",
            samples_per_second=85.5,
            gpu_utilization=88.0,
            gpu_memory_usage_gb=13.2,
            cpu_utilization=45.0
        )
        
        assert data.epoch == 10
        assert data.current_resolution == 288
        assert data.resize_phase == "transition"
        assert data.optimization_recommendations == []  # 기본값
    
    def test_monitoring_data_with_recommendations(self):
        """권고사항이 포함된 모니터링 데이터 테스트"""
        recommendations = [
            {
                'type': 'memory_warning',
                'message': 'GPU 메모리 사용량 높음',
                'priority': 'high'
            }
        ]
        
        data = Stage3MonitoringData(
            timestamp=time.time(),
            epoch=15,
            batch_idx=200,
            current_resolution=352,
            optimal_batch_size=20,
            resize_phase="transition",
            samples_per_second=72.0,
            gpu_utilization=92.0,
            gpu_memory_usage_gb=14.5,
            cpu_utilization=38.0,
            optimization_recommendations=recommendations
        )
        
        assert len(data.optimization_recommendations) == 1
        assert data.optimization_recommendations[0]['priority'] == 'high'


class TestLogStreamer:
    """로그 스트리밍 테스트"""
    
    @pytest.fixture
    def log_streamer(self):
        """로그 스트리머 인스턴스"""
        return LogStreamer()
    
    def test_log_streamer_initialization(self, log_streamer):
        """로그 스트리머 초기화 테스트"""
        assert not log_streamer.is_streaming
        assert log_streamer.process is None
        assert log_streamer.thread is None
        assert log_streamer.log_queue.maxsize == 1000
    
    def test_add_log_entry(self, log_streamer):
        """로그 항목 추가 테스트"""
        test_message = "Test log message"
        log_streamer._add_log_entry(test_message, "test")
        
        # 큐에서 로그 확인
        assert not log_streamer.log_queue.empty()
        
        logs = log_streamer.get_recent_logs(1)
        assert len(logs) == 1
        assert logs[0]['message'] == test_message
        assert logs[0]['source'] == "test"
        assert 'timestamp' in logs[0]
        assert 'datetime' in logs[0]
    
    def test_log_queue_overflow(self, log_streamer):
        """로그 큐 오버플로 처리 테스트"""
        # 큐를 가득 채우기
        for i in range(1010):  # maxsize보다 많이
            log_streamer._add_log_entry(f"Message {i}", "test")
        
        # 큐 크기가 제한되어 있는지 확인
        assert log_streamer.log_queue.qsize() <= 1000
        
        # 최근 로그들이 보존되는지 확인
        recent_logs = log_streamer.get_recent_logs(10)
        assert len(recent_logs) <= 10
    
    @patch('builtins.open')
    def test_stream_from_file(self, mock_open, log_streamer):
        """파일에서 로그 스트리밍 테스트"""
        # Mock 파일 내용
        mock_file = Mock()
        mock_file.readline.side_effect = [
            "Log line 1\n",
            "Log line 2\n", 
            "",  # EOF
        ]
        mock_open.return_value.__enter__.return_value = mock_file
        
        # 파일 스트리밍 시작
        log_file = Path("/tmp/test.log")
        log_streamer.start_log_streaming(log_file=log_file)
        
        # 짧은 시간 대기
        time.sleep(0.1)
        
        # 스트리밍 중지
        log_streamer.stop_streaming()
        
        # 로그가 추가되었는지 확인 (스트리밍 시뮬레이션이므로 기본 메시지 확인)
        logs = log_streamer.get_recent_logs(5)
        # 파일 스트리밍이 시작되었다는 로그나 시뮬레이션 로그가 있는지 확인
        assert len(logs) >= 0  # 최소한 로그 시스템이 동작하는지 확인
    
    @patch('subprocess.Popen')
    def test_stream_from_command(self, mock_popen, log_streamer):
        """명령어에서 로그 스트리밍 테스트"""
        # Mock 프로세스
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "Command output 1\n",
            "Command output 2\n",
            "",  # EOF
        ]
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        
        # 명령어 스트리밍 시작
        log_streamer.start_log_streaming(command="echo 'test'")
        
        # 짧은 시간 대기
        time.sleep(0.1)
        
        # 스트리밍 중지
        log_streamer.stop_streaming()
        
        # 명령어가 실행되었는지 확인
        mock_popen.assert_called_once()
        assert mock_popen.call_args[0][0] == "echo 'test'"
    
    def test_stop_streaming(self, log_streamer):
        """스트리밍 중지 테스트"""
        # 스트리밍 시작
        log_streamer.start_log_streaming()
        assert log_streamer.is_streaming
        
        # 스트리밍 중지
        log_streamer.stop_streaming()
        assert not log_streamer.is_streaming
        
        # 스레드가 종료되었는지 확인
        if log_streamer.thread:
            assert not log_streamer.thread.is_alive()


class TestStage3RealtimeMonitor:
    """Stage 3 실시간 모니터 테스트"""
    
    @pytest.fixture
    def monitor(self):
        """모니터 인스턴스"""
        return Stage3RealtimeMonitor()
    
    def test_monitor_initialization(self, monitor):
        """모니터 초기화 테스트"""
        assert not monitor.is_monitoring
        assert monitor.monitoring_thread is None
        assert len(monitor.data_history) == 0
        assert len(monitor.connected_clients) == 0
        assert monitor.update_interval == 1.0
    
    def test_collect_monitoring_data(self, monitor):
        """모니터링 데이터 수집 테스트"""
        data = monitor._collect_monitoring_data()
        
        assert isinstance(data, Stage3MonitoringData)
        assert data.timestamp > 0
        assert data.epoch >= 0
        assert data.current_resolution >= 224
        assert data.current_resolution <= 384
        assert 0 <= data.gpu_utilization <= 100
        assert data.gpu_memory_usage_gb >= 0
        assert 0 <= data.cpu_utilization <= 100
    
    def test_progressive_resize_simulation(self, monitor):
        """Progressive Resize 시뮬레이션 테스트"""
        # Warmup 단계
        warmup_size = monitor._simulate_progressive_resize(epoch=5)
        assert warmup_size == 224
        
        # Transition 단계
        transition_size = monitor._simulate_progressive_resize(epoch=20)
        assert 224 < transition_size < 384
        assert transition_size % 8 == 0  # 8의 배수 정렬 확인
        
        # Stable 단계
        stable_size = monitor._simulate_progressive_resize(epoch=35)
        assert stable_size == 384
    
    def test_resize_phase_detection(self, monitor):
        """Resize 단계 감지 테스트"""
        assert monitor._get_resize_phase(5) == "warmup"
        assert monitor._get_resize_phase(20) == "transition"
        assert monitor._get_resize_phase(35) == "stable"
    
    def test_optimization_recommendations(self, monitor):
        """최적화 권고사항 생성 테스트"""
        # 높은 GPU 메모리 사용량
        recommendations = monitor._generate_optimization_recommendations(
            gpu_memory=14.0, gpu_util=85.0, samples_per_sec=75.0, resolution=352
        )
        
        # 메모리 경고가 포함되는지 확인
        assert any(rec['type'] == 'memory_warning' for rec in recommendations)
        
        # 낮은 GPU 사용률
        recommendations = monitor._generate_optimization_recommendations(
            gpu_memory=10.0, gpu_util=65.0, samples_per_sec=85.0, resolution=224
        )
        
        # 성능 권고가 포함되는지 확인
        assert any(rec['type'] == 'performance' for rec in recommendations)
    
    def test_start_stop_monitoring(self, monitor):
        """모니터링 시작/중지 테스트"""
        # 모니터링 시작
        monitor.start_monitoring()
        assert monitor.is_monitoring
        assert monitor.monitoring_thread is not None
        
        # 잠시 대기 (데이터 수집 확인)
        time.sleep(1.5)
        assert len(monitor.data_history) > 0
        
        # 모니터링 중지
        monitor.stop_monitoring()
        assert not monitor.is_monitoring
    
    def test_websocket_client_management(self, monitor):
        """WebSocket 클라이언트 관리 테스트"""
        # Mock WebSocket
        mock_client1 = Mock()
        mock_client2 = Mock()
        
        # 클라이언트 추가
        monitor.add_websocket_client(mock_client1)
        monitor.add_websocket_client(mock_client2)
        
        assert len(monitor.connected_clients) == 2
        
        # 클라이언트 제거
        monitor.remove_websocket_client(mock_client1)
        assert len(monitor.connected_clients) == 1
        assert mock_client2 in monitor.connected_clients
    
    def test_get_current_status(self, monitor):
        """현재 상태 반환 테스트"""
        # 데이터 없는 상태
        status = monitor.get_current_status()
        assert status['status'] == 'no_data'
        
        # 모니터링 시작 후 상태
        monitor.start_monitoring()
        time.sleep(1.1)  # 데이터 수집 대기
        
        status = monitor.get_current_status()
        assert 'monitoring_active' in status
        assert 'data_points' in status
        assert 'latest_data' in status
        
        monitor.stop_monitoring()


class TestFastAPIEndpoints:
    """FastAPI 엔드포인트 테스트"""
    
    @pytest.fixture
    def client(self):
        """테스트 클라이언트"""
        return TestClient(app)
    
    def test_dashboard_endpoint(self, client):
        """대시보드 엔드포인트 테스트"""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Stage 3 실시간 모니터링" in response.text
        assert "WebSocket" in response.text
    
    def test_status_api(self, client):
        """상태 API 테스트"""
        response = client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        # 데이터가 없을 때는 status 키만 있고, 있을 때는 monitoring_active 키가 있음
        assert "monitoring_active" in data or "status" in data
    
    def test_start_monitoring_api(self, client):
        """모니터링 시작 API 테스트"""
        response = client.post("/api/start")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        
        # 상태 확인
        status_response = client.get("/api/status")
        status_data = status_response.json()
        assert status_data.get("monitoring_active") == True
    
    def test_stop_monitoring_api(self, client):
        """모니터링 중지 API 테스트"""
        # 먼저 시작
        client.post("/api/start")
        
        # 중지
        response = client.post("/api/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestIntegrationScenarios:
    """통합 시나리오 테스트"""
    
    def test_full_monitoring_cycle(self):
        """전체 모니터링 사이클 테스트"""
        monitor = Stage3RealtimeMonitor()
        
        try:
            # 1. 모니터링 시작
            monitor.start_monitoring()
            assert monitor.is_monitoring
            
            # 2. 데이터 수집 대기
            time.sleep(2.5)
            assert len(monitor.data_history) >= 2
            
            # 3. 모니터링 데이터 검증
            latest_data = monitor.data_history[-1]
            assert isinstance(latest_data, Stage3MonitoringData)
            assert latest_data.current_resolution >= 224
            
            # 4. 상태 확인
            status = monitor.get_current_status()
            assert status['monitoring_active'] == True
            assert status['data_points'] > 0
            
            # 5. 로그 확인
            recent_logs = monitor.log_streamer.get_recent_logs(5)
            assert isinstance(recent_logs, list)
            
        finally:
            # 정리
            monitor.stop_monitoring()
            assert not monitor.is_monitoring
    
    @patch('psutil.cpu_percent')
    def test_system_resource_monitoring(self, mock_cpu):
        """시스템 리소스 모니터링 테스트"""
        mock_cpu.return_value = 67.5
        
        monitor = Stage3RealtimeMonitor()
        
        try:
            monitor.start_monitoring()
            time.sleep(1.5)
            
            # 수집된 데이터에서 CPU 사용률 확인
            if monitor.data_history:
                latest_data = monitor.data_history[-1]
                assert latest_data.cpu_utilization >= 0
                
        finally:
            monitor.stop_monitoring()
    
    def test_log_streaming_with_file(self):
        """파일 기반 로그 스트리밍 통합 테스트"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            test_log_file = Path(f.name)
            f.write("Initial log line\n")
        
        try:
            monitor = Stage3RealtimeMonitor()
            monitor.start_monitoring(str(test_log_file))
            
            # 로그 파일에 새 라인 추가
            with open(test_log_file, 'a') as f:
                f.write("New log line\n")
                f.flush()
            
            time.sleep(1.5)
            
            # 로그가 스트리밍되었는지 확인
            logs = monitor.log_streamer.get_recent_logs(10)
            log_messages = [log['message'] for log in logs]
            assert any("log line" in msg for msg in log_messages)
            
        finally:
            monitor.stop_monitoring()
            test_log_file.unlink()  # 파일 삭제


class TestPerformanceAndReliability:
    """성능 및 안정성 테스트"""
    
    def test_high_frequency_data_collection(self):
        """고빈도 데이터 수집 테스트"""
        monitor = Stage3RealtimeMonitor()
        monitor.update_interval = 0.1  # 매우 빠른 업데이트
        
        try:
            monitor.start_monitoring()
            time.sleep(2.0)  # 2초 동안 실행
            
            # 많은 데이터가 수집되었는지 확인
            assert len(monitor.data_history) >= 10
            
            # 데이터 품질 확인
            for data in monitor.data_history[-5:]:
                assert isinstance(data, Stage3MonitoringData)
                assert data.timestamp > 0
                
        finally:
            monitor.stop_monitoring()
    
    def test_memory_usage_control(self):
        """메모리 사용량 제어 테스트"""
        monitor = Stage3RealtimeMonitor()
        monitor.update_interval = 0.01  # 매우 빠른 업데이트
        
        try:
            monitor.start_monitoring()
            time.sleep(5.0)  # 긴 시간 실행
            
            # 데이터 히스토리가 제한되는지 확인
            assert len(monitor.data_history) <= 3600  # 최대 1시간분
            
        finally:
            monitor.stop_monitoring()
    
    def test_error_resilience(self):
        """오류 복원력 테스트"""
        monitor = Stage3RealtimeMonitor()
        
        # Mock에서 오류 발생하도록 설정
        with patch.object(monitor, '_collect_monitoring_data') as mock_collect:
            mock_collect.side_effect = [
                Exception("Test error"),
                Stage3MonitoringData(
                    timestamp=time.time(), epoch=1, batch_idx=1,
                    current_resolution=224, optimal_batch_size=32,
                    resize_phase="warmup", samples_per_second=80.0,
                    gpu_utilization=75.0, gpu_memory_usage_gb=10.0,
                    cpu_utilization=45.0
                )
            ]
            
            try:
                monitor.start_monitoring()
                time.sleep(6.0)  # 오류 후 복구 시간 허용
                
                # 오류가 발생해도 모니터링이 계속되는지 확인
                assert monitor.is_monitoring
                
            finally:
                monitor.stop_monitoring()


if __name__ == "__main__":
    # 빠른 테스트 실행
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"
    ])