#!/usr/bin/env python3
"""
동적 진행률 계산 테스트

모니터링 시스템이 다양한 에포크/배치 설정을 올바르게 감지하고 진행률을 계산하는지 테스트
"""

import re
import json

class DynamicProgressTest:
    """동적 진행률 계산 테스트"""
    
    def __init__(self):
        self.test_results = []
        
        # 진행률 계산 패턴들 (모니터링 시스템과 동일)
        self.epoch_patterns = [
            r'(?:Starting training for|will run for)\s+(\d+)\s+epochs?',
            r'Total epochs?[:\s]+(\d+)',
            r'Training for\s+(\d+)\s+epochs?',
            r'epochs?[=:\s]+(\d+)',
            r'(?:--epochs?|epochs)\s+(\d+)'
        ]
        
        self.patterns = {
            'batch_total': r'Batch\s+\d+/(\d+)',
            'current_epoch': r'Epoch\s+(\d+)',
            'current_batch': r'Batch\s+(\d+)(?:/\d+)?'
        }
    
    def test_epoch_detection(self):
        """에포크 설정 자동 감지 테스트"""
        print("🔍 에포크 설정 자동 감지 테스트...")
        
        epoch_logs = [
            "Starting training for 50 epochs with Stage 3 configuration",
            "Total epochs: 100 | Stage 4 full training",
            "Starting training for 25 epochs (quick test)",
            "Training will run for 200 epochs",
        ]
        
        for log in epoch_logs:
            detected = False
            total_epochs = None
            
            # 다중 패턴 시도
            for pattern in self.epoch_patterns:
                match = re.search(pattern, log, re.IGNORECASE)
                if match:
                    total_epochs = int(match.group(1))
                    detected = True
                    break
            
            if detected:
                result = {
                    'type': 'epoch_detection',
                    'log': log,
                    'detected_epochs': total_epochs,
                    'status': '✅ 감지 성공'
                }
                self.test_results.append(result)
                print(f"  ✅ 에포크 감지: {total_epochs} (로그: {log[:50]}...)")
            else:
                result = {
                    'type': 'epoch_detection',
                    'log': log,
                    'detected_epochs': None,
                    'status': '❌ 감지 실패'
                }
                self.test_results.append(result)
                print(f"  ❌ 에포크 감지 실패: {log[:50]}...")
    
    def test_batch_detection(self):
        """배치 설정 자동 감지 테스트"""
        print("\n🔍 배치 설정 자동 감지 테스트...")
        
        batch_logs = [
            "Epoch 1 | Batch 100/5093 | Loss: 4.2",
            "Epoch 2 | Batch 2500/5093 | Loss: 3.8", 
            "Epoch 1 | Batch 50/1000 | Loss: 5.1",  # 더 작은 데이터셋
            "Epoch 1 | Batch 1000/25000 | Loss: 3.2",  # 더 큰 데이터셋
        ]
        
        max_batches = 0
        for log in batch_logs:
            match = re.search(self.patterns['batch_total'], log)
            if match:
                total_batches = int(match.group(1))
                if total_batches > max_batches:
                    max_batches = total_batches
                    
                result = {
                    'type': 'batch_detection',
                    'log': log,
                    'detected_batches': total_batches,
                    'max_batches': max_batches,
                    'status': '✅ 감지 성공'
                }
                self.test_results.append(result)
                print(f"  ✅ 배치 감지: {total_batches} (최대: {max_batches})")
    
    def test_progress_calculation(self):
        """진행률 계산 로직 테스트"""
        print("\n🔍 진행률 계산 로직 테스트...")
        
        test_cases = [
            # (총_에포크, 현재_에포크, 총_배치, 현재_배치, 예상_진행률)
            (50, 5, 5093, 2500, 9),    # 5/50 에포크, 배치 50% = 약 9%
            (100, 25, 1000, 500, 24),  # 25/100 에포크, 배치 50% = 약 24%  
            (10, 3, 2000, 1000, 25),   # 3/10 에포크, 배치 50% = 약 25%
            (20, 20, 5000, 5000, 100), # 마지막 에포크, 마지막 배치 = 100%
        ]
        
        for total_epochs, current_epoch, total_batches, current_batch, expected_progress in test_cases:
            # JavaScript와 동일한 계산 로직
            epoch_progress = ((current_epoch - 1) / total_epochs) * 100
            batch_progress = (current_batch / total_batches) * 100
            current_epoch_progress = batch_progress / total_epochs
            overall_progress = round(epoch_progress + current_epoch_progress)
            
            accuracy = abs(overall_progress - expected_progress) <= 2  # ±2% 허용 오차
            
            result = {
                'type': 'progress_calculation',
                'total_epochs': total_epochs,
                'current_epoch': current_epoch,
                'total_batches': total_batches,
                'current_batch': current_batch,
                'calculated_progress': overall_progress,
                'expected_progress': expected_progress,
                'accuracy': accuracy,
                'status': '✅ 계산 정확' if accuracy else '❌ 계산 오차'
            }
            self.test_results.append(result)
            
            status = "✅" if accuracy else "❌"
            print(f"  {status} Epoch {current_epoch}/{total_epochs}, Batch {current_batch}/{total_batches} → {overall_progress}% (예상: {expected_progress}%)")
    
    def test_dynamic_updates(self):
        """동적 업데이트 시나리오 테스트"""
        print("\n🔍 동적 업데이트 시나리오 테스트...")
        
        # Stage 3에서 Stage 4로 전환되는 시나리오 시뮬레이션
        scenarios = [
            {
                'stage': 'Stage 3',
                'logs': [
                    "Starting training for 50 epochs with Stage 3 configuration",
                    "Epoch 1 | Batch 100/5093 | Loss: 4.2",
                    "Epoch 25 | Batch 2500/5093 | Loss: 2.8",
                ],
                'expected_total_epochs': 50,
                'expected_total_batches': 5093
            },
            {
                'stage': 'Stage 4',
                'logs': [
                    "Starting training for 100 epochs with Stage 4 configuration", 
                    "Epoch 1 | Batch 500/25000 | Loss: 3.5",
                    "Epoch 50 | Batch 12500/25000 | Loss: 2.1",
                ],
                'expected_total_epochs': 100,
                'expected_total_batches': 25000
            }
        ]
        
        for scenario in scenarios:
            stage = scenario['stage']
            print(f"\n  📊 {stage} 시나리오 테스트:")
            
            detected_epochs = None
            detected_batches = None
            
            for log in scenario['logs']:
                # 에포크 감지 (다중 패턴) - 첫 번째 감지된 값만 사용
                if detected_epochs is None:
                    for pattern in self.epoch_patterns:
                        epoch_match = re.search(pattern, log, re.IGNORECASE)
                        if epoch_match:
                            detected_epochs = int(epoch_match.group(1))
                            break
                
                # 배치 감지
                batch_match = re.search(self.patterns['batch_total'], log)
                if batch_match:
                    total_batches = int(batch_match.group(1))
                    if not detected_batches or total_batches > detected_batches:
                        detected_batches = total_batches
            
            epochs_correct = detected_epochs == scenario['expected_total_epochs']
            batches_correct = detected_batches == scenario['expected_total_batches']
            
            result = {
                'type': 'dynamic_update',
                'stage': stage,
                'detected_epochs': detected_epochs,
                'expected_epochs': scenario['expected_total_epochs'],
                'detected_batches': detected_batches,
                'expected_batches': scenario['expected_total_batches'],
                'epochs_correct': epochs_correct,
                'batches_correct': batches_correct,
                'status': '✅ 시나리오 성공' if (epochs_correct and batches_correct) else '❌ 시나리오 실패'
            }
            self.test_results.append(result)
            
            epoch_status = "✅" if epochs_correct else "❌"
            batch_status = "✅" if batches_correct else "❌"
            
            print(f"    {epoch_status} 에포크: {detected_epochs} (예상: {scenario['expected_total_epochs']})")
            print(f"    {batch_status} 배치: {detected_batches} (예상: {scenario['expected_total_batches']})")
    
    def generate_report(self):
        """테스트 보고서 생성"""
        print("\n" + "="*60)
        print("📊 동적 진행률 계산 테스트 보고서")
        print("="*60)
        
        total_tests = len(self.test_results)
        success_tests = sum(1 for result in self.test_results if '✅' in result['status'])
        
        print(f"총 테스트: {total_tests}")
        print(f"성공: {success_tests}")
        print(f"실패: {total_tests - success_tests}")
        print(f"성공률: {(success_tests / total_tests * 100):.1f}%")
        
        print("\n📋 테스트 타입별 결과:")
        test_types = {}
        for result in self.test_results:
            test_type = result['type']
            if test_type not in test_types:
                test_types[test_type] = {'total': 0, 'success': 0}
            test_types[test_type]['total'] += 1
            if '✅' in result['status']:
                test_types[test_type]['success'] += 1
        
        for test_type, counts in test_types.items():
            success_rate = (counts['success'] / counts['total'] * 100) if counts['total'] > 0 else 0
            print(f"  {test_type}: {counts['success']}/{counts['total']} ({success_rate:.1f}%)")
        
        # JSON 보고서 저장
        report = {
            'timestamp': '2025-08-24T12:00:00Z',
            'summary': {
                'total_tests': total_tests,
                'success_tests': success_tests,
                'success_rate': success_tests / total_tests * 100
            },
            'test_types': test_types,
            'details': self.test_results
        }
        
        with open('/tmp/dynamic_progress_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 상세 보고서 저장: /tmp/dynamic_progress_report.json")
        
        if success_tests == total_tests:
            print("\n🎉 모든 테스트 통과! 동적 진행률 계산 완벽!")
            return True
        else:
            print(f"\n⚠️ {total_tests - success_tests}개 테스트 실패. 추가 점검 필요.")
            return False
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 동적 진행률 계산 테스트 시작")
        print("="*60)
        
        self.test_epoch_detection()
        self.test_batch_detection()
        self.test_progress_calculation()
        self.test_dynamic_updates()
        
        return self.generate_report()


if __name__ == "__main__":
    tester = DynamicProgressTest()
    success = tester.run_all_tests()
    exit(0 if success else 1)