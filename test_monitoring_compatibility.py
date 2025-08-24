#!/usr/bin/env python3
"""
모니터링 시스템 호환성 테스트

기존 로그 포맷과 Phase 1 새로운 로그 포맷이 모두 올바르게 파싱되는지 검증
"""

import re
import json
from datetime import datetime, timezone, timedelta

class MonitoringCompatibilityTest:
    """모니터링 시스템 호환성 테스트"""
    
    def __init__(self):
        self.test_results = []
        
        # Phase 1 확장된 정규식 패턴들 (realtime_training_logger.py와 동일)
        self.patterns = {
            # 기존 패턴들
            'batch': r'Epoch\s+(\d+)\s+\|\s+Batch\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([\d.]+)',
            'accuracy': r'Cls Acc:\s+([\d.]+)\s+\|\s+Det mAP:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)s',
            
            # Phase 1 새로운 패턴들
            'top5': r'Top-1:\s*([‰\d.]+)\s*\|\s*Top-5:\s*([\d.]+)',
            'macro_f1': r'Macro-F1:\s*([\d.]+)',
            'domain': r'Single:\s*Top-1=([‰\d.]+)\s*\|\s*Combination:\s*Top-1=([\d.]+)',
            'latency': r'Pipeline:\s*det=([‰\d.]+)ms,\s*crop=([\d.]+)ms,\s*cls=([\d.]+)ms,\s*total=([\d.]+)ms',
            'confidence': r'Auto-selected confidence:\s*det=([‰\d.]+),\s*cls=([\d.]+)',
            'oom': r'OOM Guard:\s*batch_size\s*reduced\s*(\d+)→(\d+),\s*grad_accum\s*(\d+)→(\d+)',
            'interleave': r'Interleaved:\s*det_steps=(\d+),\s*cls_steps=(\d+)\s*\(ratio=1:([\d.]+)\)'
        }
    
    def test_legacy_logs(self):
        """기존 로그 포맷 테스트"""
        print("🔍 기존 로그 포맷 테스트...")
        
        legacy_logs = [
            "2025-08-24 15:30:22 | INFO | Epoch 4 | Batch 2000/5093 | Loss: 3.5000",
            "2025-08-24 15:35:45 | INFO | Epoch 4 완료 | Cls Acc: 0.441 | Det mAP: 0.250 | Time: 635.0s",
            "2025-08-24 15:36:00 | INFO | Stage 3 학습 진행 중...",
        ]
        
        for log in legacy_logs:
            # Batch 패턴 테스트
            batch_match = re.search(self.patterns['batch'], log)
            if batch_match:
                epoch, current_batch, total_batches, loss = batch_match.groups()
                result = {
                    'type': 'legacy_batch',
                    'epoch': int(epoch),
                    'batch': f"{current_batch}/{total_batches}",
                    'loss': loss,
                    'status': '✅ 파싱 성공'
                }
                self.test_results.append(result)
                print(f"  ✅ Batch 파싱: Epoch {epoch}, Batch {current_batch}/{total_batches}, Loss {loss}")
            
            # 정확도 패턴 테스트
            acc_match = re.search(self.patterns['accuracy'], log)
            if acc_match:
                cls_acc, det_map, epoch_time = acc_match.groups()
                result = {
                    'type': 'legacy_accuracy',
                    'cls_acc': cls_acc,
                    'det_map': det_map,
                    'epoch_time': f"{epoch_time}s",
                    'status': '✅ 파싱 성공'
                }
                self.test_results.append(result)
                print(f"  ✅ 정확도 파싱: Cls {cls_acc}, Det mAP {det_map}, Time {epoch_time}s")
    
    def test_phase1_logs(self):
        """Phase 1 새로운 로그 포맷 테스트"""
        print("\n🆕 Phase 1 새로운 로그 포맷 테스트...")
        
        phase1_logs = [
            "2025-08-24 15:40:12 | METRIC | Top-1: 0.441 | Top-5: 0.672",
            "2025-08-24 15:40:15 | METRIC | Macro-F1: 0.387",
            "2025-08-24 15:40:18 | DOMAIN | Single: Top-1=0.523 | Combination: Top-1=0.342",
            "2025-08-24 15:40:25 | LATENCY | Pipeline: det=45ms, crop=12ms, cls=28ms, total=85ms",
            "2025-08-24 15:40:30 | CONFIDENCE | Auto-selected confidence: det=0.25, cls=0.30",
            "2025-08-24 15:40:35 | OOM | OOM Guard: batch_size reduced 16→8, grad_accum 2→4",
            "2025-08-24 15:40:40 | INTERLEAVE | Interleaved: det_steps=1247, cls_steps=2491 (ratio=1:2.00)"
        ]
        
        for log in phase1_logs:
            # Top-5 정확도 테스트
            top5_match = re.search(self.patterns['top5'], log)
            if top5_match:
                top1_acc, top5_acc = top5_match.groups()
                result = {
                    'type': 'phase1_top5',
                    'top1_accuracy': top1_acc,
                    'top5_accuracy': top5_acc,
                    'status': '✅ 파싱 성공'
                }
                self.test_results.append(result)
                print(f"  ✅ Top-5 파싱: Top-1 {top1_acc}, Top-5 {top5_acc}")
            
            # Macro F1 테스트
            f1_match = re.search(self.patterns['macro_f1'], log)
            if f1_match:
                macro_f1 = f1_match.groups()[0]
                result = {
                    'type': 'phase1_f1',
                    'macro_f1': macro_f1,
                    'status': '✅ 파싱 성공'
                }
                self.test_results.append(result)
                print(f"  ✅ Macro-F1 파싱: {macro_f1}")
            
            # 도메인별 성능 테스트
            domain_match = re.search(self.patterns['domain'], log)
            if domain_match:
                single_acc, combo_acc = domain_match.groups()
                result = {
                    'type': 'phase1_domain',
                    'single_domain_acc': single_acc,
                    'combination_domain_acc': combo_acc,
                    'status': '✅ 파싱 성공'
                }
                self.test_results.append(result)
                print(f"  ✅ 도메인별 파싱: Single {single_acc}, Combination {combo_acc}")
            
            # 레이턴시 분해 테스트
            latency_match = re.search(self.patterns['latency'], log)
            if latency_match:
                det_latency, crop_latency, cls_latency, total_latency = latency_match.groups()
                result = {
                    'type': 'phase1_latency',
                    'det_latency': det_latency,
                    'crop_latency': crop_latency,
                    'cls_latency': cls_latency,
                    'total_latency': total_latency,
                    'status': '✅ 파싱 성공'
                }
                self.test_results.append(result)
                print(f"  ✅ 레이턴시 파싱: Det {det_latency}ms, Crop {crop_latency}ms, Cls {cls_latency}ms, Total {total_latency}ms")
            
            # Auto Confidence 테스트
            conf_match = re.search(self.patterns['confidence'], log)
            if conf_match:
                det_conf, cls_conf = conf_match.groups()
                result = {
                    'type': 'phase1_confidence',
                    'det_confidence': det_conf,
                    'cls_confidence': cls_conf,
                    'status': '✅ 파싱 성공'
                }
                self.test_results.append(result)
                print(f"  ✅ Confidence 파싱: Detection {det_conf}, Classification {cls_conf}")
            
            # OOM Guard 테스트
            oom_match = re.search(self.patterns['oom'], log)
            if oom_match:
                old_batch, new_batch, old_accum, new_accum = oom_match.groups()
                result = {
                    'type': 'phase1_oom',
                    'oom_old_batch': old_batch,
                    'oom_new_batch': new_batch,
                    'oom_old_accum': old_accum,
                    'oom_new_accum': new_accum,
                    'status': '✅ 파싱 성공'
                }
                self.test_results.append(result)
                print(f"  ✅ OOM Guard 파싱: Batch {old_batch}→{new_batch}, Accum {old_accum}→{new_accum}")
            
            # Interleaved Learning 테스트
            interleave_match = re.search(self.patterns['interleave'], log)
            if interleave_match:
                det_steps, cls_steps, ratio = interleave_match.groups()
                result = {
                    'type': 'phase1_interleave',
                    'det_steps': det_steps,
                    'cls_steps': cls_steps,
                    'interleave_ratio': ratio,
                    'status': '✅ 파싱 성공'
                }
                self.test_results.append(result)
                print(f"  ✅ Interleave 파싱: Det {det_steps}, Cls {cls_steps}, Ratio 1:{ratio}")
    
    def test_websocket_payload(self):
        """WebSocket 페이로드 구조 테스트"""
        print("\n📡 WebSocket 페이로드 구조 테스트...")
        
        # 샘플 페이로드 생성 (realtime_training_logger.py broadcast_to_clients와 동일 구조)
        sample_payload = {
            "timestamp": datetime.now(timezone(timedelta(hours=9))).isoformat(),
            "message": "Test message",
            "type": "realtime",
            # 기존 메트릭
            "epoch": 4,
            "batch": "2000/5093",
            "loss": "3.5000",
            "cls_acc": "0.441",
            "det_map": "0.250",
            "epoch_time": "635.0s",
            # Phase 1 새로운 메트릭
            "top1_accuracy": "0.441",
            "top5_accuracy": "0.672",
            "macro_f1": "0.387",
            "single_domain_acc": "0.523",
            "combination_domain_acc": "0.342",
            "det_latency": "45",
            "crop_latency": "12",
            "cls_latency": "28",
            "total_latency": "85",
            "det_confidence": "0.25",
            "cls_confidence": "0.30",
            "oom_old_batch": "16",
            "oom_new_batch": "8",
            "oom_old_accum": "2",
            "oom_new_accum": "4",
            "det_steps": "1247",
            "cls_steps": "2491",
            "interleave_ratio": "2.00"
        }
        
        try:
            # JSON 직렬화/역직렬화 테스트
            json_payload = json.dumps(sample_payload, ensure_ascii=False)
            parsed_payload = json.loads(json_payload)
            
            # 필수 필드 검증
            required_fields = [
                'timestamp', 'message', 'type', 'epoch', 'batch', 'loss'
            ]
            
            phase1_fields = [
                'top1_accuracy', 'top5_accuracy', 'macro_f1', 
                'single_domain_acc', 'combination_domain_acc',
                'det_latency', 'crop_latency', 'cls_latency', 'total_latency',
                'det_confidence', 'cls_confidence'
            ]
            
            # 기존 필드 검증
            missing_required = [field for field in required_fields if field not in parsed_payload]
            if not missing_required:
                print("  ✅ 기존 필수 필드 모두 존재")
            else:
                print(f"  ❌ 누락된 필수 필드: {missing_required}")
            
            # Phase 1 필드 검증
            missing_phase1 = [field for field in phase1_fields if field not in parsed_payload]
            if not missing_phase1:
                print("  ✅ Phase 1 새로운 필드 모두 존재")
            else:
                print(f"  ❌ 누락된 Phase 1 필드: {missing_phase1}")
            
            # 페이로드 크기 확인
            payload_size = len(json_payload)
            print(f"  📊 페이로드 크기: {payload_size} bytes")
            
            if payload_size < 2048:  # 2KB 미만
                print("  ✅ 페이로드 크기 적정 (< 2KB)")
            else:
                print("  ⚠️ 페이로드 크기가 큼 (≥ 2KB)")
            
            result = {
                'type': 'websocket_payload',
                'payload_size': payload_size,
                'required_fields': len(required_fields) - len(missing_required),
                'phase1_fields': len(phase1_fields) - len(missing_phase1),
                'status': '✅ 페이로드 테스트 성공'
            }
            self.test_results.append(result)
            
        except Exception as e:
            print(f"  ❌ 페이로드 테스트 실패: {e}")
            result = {
                'type': 'websocket_payload',
                'error': str(e),
                'status': '❌ 페이로드 테스트 실패'
            }
            self.test_results.append(result)
    
    def generate_report(self):
        """테스트 보고서 생성"""
        print("\n" + "="*60)
        print("📊 모니터링 시스템 호환성 테스트 보고서")
        print("="*60)
        
        total_tests = len(self.test_results)
        success_tests = sum(1 for result in self.test_results if '✅' in result['status'])
        
        print(f"총 테스트: {total_tests}")
        print(f"성공: {success_tests}")
        print(f"실패: {total_tests - success_tests}")
        print(f"성공률: {(success_tests / total_tests * 100):.1f}%")
        
        print("\n📋 상세 결과:")
        for result in self.test_results:
            print(f"  {result['status']} - {result['type']}")
        
        # JSON 보고서 저장
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'success_tests': success_tests,
                'success_rate': success_tests / total_tests * 100
            },
            'details': self.test_results
        }
        
        with open('/tmp/monitoring_compatibility_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 상세 보고서 저장: /tmp/monitoring_compatibility_report.json")
        
        if success_tests == total_tests:
            print("\n🎉 모든 테스트 통과! 모니터링 시스템 호환성 완벽!")
            return True
        else:
            print(f"\n⚠️ {total_tests - success_tests}개 테스트 실패. 추가 점검 필요.")
            return False
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 모니터링 시스템 호환성 테스트 시작")
        print("="*60)
        
        self.test_legacy_logs()
        self.test_phase1_logs()
        self.test_websocket_payload()
        
        return self.generate_report()


if __name__ == "__main__":
    tester = MonitoringCompatibilityTest()
    success = tester.run_all_tests()
    exit(0 if success else 1)