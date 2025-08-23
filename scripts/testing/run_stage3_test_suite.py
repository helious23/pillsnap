#!/usr/bin/env python3
"""
Stage 3 Classification 프로덕션급 테스트 스위트 실행기

프로덕션 직전 철저한 검증을 위한 종합 테스트 시스템:
- 단위 테스트: Manifest Creator 핵심 기능
- 통합 테스트: Classification Trainer 및 Training Script
- 성능 테스트: 프로덕션 환경 시뮬레이션
- GPU 테스트: 메모리 안정성 및 누수 탐지 (CUDA 사용 가능시)
- 종합 보고서: 프로덕션 준비도 평가
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.core import PillSnapLogger


class Stage3TestSuiteRunner:
    """Stage 3 Classification 테스트 스위트 실행기"""
    
    def __init__(self):
        self.logger = PillSnapLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent
        self.test_results = {}
        self.cuda_available = self._check_cuda_availability()
        
        self.logger.info("Stage 3 Classification 테스트 스위트 실행기 초기화")
        if self.cuda_available:
            self.logger.info("🎮 CUDA 사용 가능 - GPU 테스트 포함")
        else:
            self.logger.warning("💻 CUDA 없음 - CPU 테스트만 실행")
    
    def _check_cuda_availability(self) -> bool:
        """CUDA 사용 가능성 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def run_command(self, command: list, test_name: str, timeout: int = 600) -> dict:
        """명령어 실행 및 결과 수집"""
        self.logger.info(f"🔄 {test_name} 실행 중...")
        
        start_time = time.time()
        
        try:
            # 환경변수 설정
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            env['PILLSNAP_DATA_ROOT'] = '/home/max16/pillsnap_data'
            
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            test_result = {
                'name': test_name,
                'success': result.returncode == 0,
                'duration': duration,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if test_result['success']:
                self.logger.info(f"  ✅ {test_name} 성공 ({duration:.1f}초)")
            else:
                self.logger.error(f"  ❌ {test_name} 실패 ({duration:.1f}초)")
                self.logger.error(f"  Error: {result.stderr[-300:]}")  # 마지막 300자만
            
            return test_result
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"  ⏰ {test_name} 타임아웃 ({timeout}초)")
            return {
                'name': test_name,
                'success': False,
                'duration': timeout,
                'returncode': -1,
                'stdout': '',
                'stderr': f'Test timed out after {timeout} seconds'
            }
        except Exception as e:
            self.logger.error(f"  💥 {test_name} 실행 오류: {e}")
            return {
                'name': test_name,
                'success': False,
                'duration': 0,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e)
            }
    
    def run_unit_tests(self):
        """단위 테스트 실행"""
        self.logger.info("📋 단위 테스트 시작...")
        
        unit_tests = [
            {
                'name': 'Stage 3 Manifest Creator 단위 테스트',
                'command': [
                    'python', '-m', 'pytest',
                    'tests/unit/test_stage3_manifest_creator.py',
                    '-v', '--tb=short'
                ],
                'timeout': 900
            }
        ]
        
        unit_results = []
        for test in unit_tests:
            result = self.run_command(test['command'], test['name'], test['timeout'])
            unit_results.append(result)
        
        self.test_results['unit_tests'] = unit_results
        
        success_count = sum(1 for r in unit_results if r['success'])
        self.logger.info(f"📋 단위 테스트 완료: {success_count}/{len(unit_results)} 성공")
    
    def run_integration_tests(self):
        """통합 테스트 실행"""
        self.logger.info("🔗 통합 테스트 시작...")
        
        integration_tests = [
            {
                'name': 'Classification Trainer 통합 테스트',
                'command': [
                    'python', '-m', 'pytest',
                    'tests/integration/test_stage3_classification_training.py',
                    '-v', '--tb=short'
                ],
                'timeout': 1500
            },
            {
                'name': 'Training Script 통합 테스트',
                'command': [
                    'python', '-m', 'pytest',
                    'tests/integration/test_stage3_training_script.py',
                    '-v', '--tb=short'
                ],
                'timeout': 1200
            }
        ]
        
        integration_results = []
        for test in integration_tests:
            result = self.run_command(test['command'], test['name'], test['timeout'])
            integration_results.append(result)
        
        self.test_results['integration_tests'] = integration_results
        
        success_count = sum(1 for r in integration_results if r['success'])
        self.logger.info(f"🔗 통합 테스트 완료: {success_count}/{len(integration_results)} 성공")
    
    def run_performance_tests(self):
        """성능 테스트 실행"""
        self.logger.info("⚡ 성능 테스트 시작...")
        
        performance_tests = [
            {
                'name': '프로덕션 환경 시뮬레이션',
                'command': [
                    'python', '-m', 'pytest',
                    'tests/performance/test_stage3_production_simulation.py',
                    '-v', '--tb=short', '-m', 'not slow'
                ],
                'timeout': 2100
            }
        ]
        
        performance_results = []
        for test in performance_tests:
            result = self.run_command(test['command'], test['name'], test['timeout'])
            performance_results.append(result)
        
        self.test_results['performance_tests'] = performance_results
        
        success_count = sum(1 for r in performance_results if r['success'])
        self.logger.info(f"⚡ 성능 테스트 완료: {success_count}/{len(performance_results)} 성공")
    
    def run_gpu_tests(self):
        """GPU 테스트 실행 (CUDA 사용 가능시에만)"""
        if not self.cuda_available:
            self.logger.info("🚫 CUDA 없음 - GPU 테스트 건너뜀")
            self.test_results['gpu_tests'] = []
            return
        
        self.logger.info("🎮 GPU 테스트 시작...")
        
        gpu_tests = [
            {
                'name': 'GPU 메모리 안정성 테스트',
                'command': [
                    'python', '-m', 'pytest',
                    'tests/smoke/test_stage3_gpu_memory_stability.py',
                    '-v', '--tb=short', '-m', 'not slow'
                ],
                'timeout': 2700
            }
        ]
        
        gpu_results = []
        for test in gpu_tests:
            result = self.run_command(test['command'], test['name'], test['timeout'])
            gpu_results.append(result)
        
        self.test_results['gpu_tests'] = gpu_results
        
        success_count = sum(1 for r in gpu_results if r['success'])
        self.logger.info(f"🎮 GPU 테스트 완료: {success_count}/{len(gpu_results)} 성공")
    
    def run_long_stability_tests(self):
        """장시간 안정성 테스트 (선택적)"""
        if os.getenv('RUN_LONG_TESTS', 'false').lower() != 'true':
            self.logger.info("⏳ 장시간 테스트 건너뜀 (RUN_LONG_TESTS=true로 설정시 실행)")
            self.test_results['long_stability_tests'] = []
            return
        
        self.logger.info("🕐 장시간 안정성 테스트 시작...")
        
        long_tests = []
        
        # 프로덕션 시뮬레이션 장시간 테스트
        long_tests.append({
            'name': '장시간 프로덕션 안정성',
            'command': [
                'python', '-m', 'pytest',
                'tests/performance/test_stage3_production_simulation.py',
                '-v', '--tb=short', '-m', 'slow'
            ],
            'timeout': 3900
        })
        
        # GPU 장시간 안정성 테스트 (CUDA 사용 가능시)
        if self.cuda_available:
            long_tests.append({
                'name': '장시간 GPU 메모리 안정성',
                'command': [
                    'python', '-m', 'pytest',
                    'tests/smoke/test_stage3_gpu_memory_stability.py',
                    '-v', '--tb=short', '-m', 'slow'
                ],
                'timeout': 3900
            })
        
        long_results = []
        for test in long_tests:
            result = self.run_command(test['command'], test['name'], test['timeout'])
            long_results.append(result)
        
        self.test_results['long_stability_tests'] = long_results
        
        success_count = sum(1 for r in long_results if r['success'])
        self.logger.info(f"🕐 장시간 테스트 완료: {success_count}/{len(long_results)} 성공")
    
    def calculate_production_readiness_score(self) -> float:
        """프로덕션 준비도 점수 계산"""
        category_weights = {
            'unit_tests': 0.25,        # 25% - 기본 기능 검증
            'integration_tests': 0.35, # 35% - 시스템 통합
            'performance_tests': 0.25,  # 25% - 성능 요구사항
            'gpu_tests': 0.15          # 15% - GPU 최적화 (CUDA 없으면 가중치 재배분)
        }
        
        if not self.cuda_available:
            # GPU 테스트 없으면 가중치 재배분
            category_weights['unit_tests'] = 0.30
            category_weights['integration_tests'] = 0.40
            category_weights['performance_tests'] = 0.30
            category_weights['gpu_tests'] = 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for category, weight in category_weights.items():
            if category in self.test_results and self.test_results[category]:
                results = self.test_results[category]
                success_rate = sum(1 for r in results if r['success']) / len(results)
                total_score += success_rate * weight
                total_weight += weight
        
        return (total_score / total_weight) if total_weight > 0 else 0.0
    
    def generate_comprehensive_report(self):
        """종합 테스트 결과 보고서 생성"""
        self.logger.info("📊 종합 테스트 결과 보고서 생성...")
        
        total_tests = 0
        total_success = 0
        total_duration = 0
        
        # 보고서 헤더
        report_lines = [
            "=" * 100,
            "🎯 Stage 3 Classification 프로덕션급 테스트 결과 보고서",
            f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"💻 환경: {'CUDA 사용 가능' if self.cuda_available else 'CPU 전용'}",
            "=" * 100,
            ""
        ]
        
        # 카테고리별 결과
        category_info = {
            'unit_tests': ('📋 단위 테스트', '핵심 컴포넌트 개별 검증'),
            'integration_tests': ('🔗 통합 테스트', '시스템 간 상호작용 검증'),
            'performance_tests': ('⚡ 성능 테스트', '프로덕션 환경 시뮬레이션'),
            'gpu_tests': ('🎮 GPU 테스트', 'GPU 메모리 안정성 검증'),
            'long_stability_tests': ('🕐 장시간 테스트', '장기간 안정성 검증')
        }
        
        for category, results in self.test_results.items():
            if not results:
                continue
                
            category_name, description = category_info.get(category, (category, ''))
            category_success = sum(1 for r in results if r['success'])
            category_duration = sum(r['duration'] for r in results)
            
            total_tests += len(results)
            total_success += category_success
            total_duration += category_duration
            
            success_rate = (category_success / len(results)) * 100
            status_icon = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 50 else "❌"
            
            report_lines.extend([
                f"{status_icon} {category_name}:",
                f"   설명: {description}",
                f"   성공: {category_success}/{len(results)} ({success_rate:.1f}%)",
                f"   소요 시간: {category_duration:.1f}초",
                ""
            ])
            
            for result in results:
                test_status = "✅" if result['success'] else "❌"
                report_lines.append(f"     {test_status} {result['name']} ({result['duration']:.1f}초)")
            
            report_lines.append("")
        
        # 프로덕션 준비도 점수
        production_score = self.calculate_production_readiness_score()
        success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        
        # 준비도 등급 결정
        if production_score >= 0.95:
            readiness_grade = "🏆 우수 (Excellent)"
        elif production_score >= 0.90:
            readiness_grade = "🎯 준비완료 (Production Ready)"
        elif production_score >= 0.80:
            readiness_grade = "⚠️ 주의필요 (Needs Attention)"
        else:
            readiness_grade = "❌ 미준비 (Not Ready)"
        
        # 전체 요약
        report_lines.extend([
            "=" * 100,
            "📊 종합 결과 요약:",
            f"   총 테스트: {total_tests}개",
            f"   성공: {total_success}개",
            f"   실패: {total_tests - total_success}개",
            f"   전체 성공률: {success_rate:.1f}%",
            f"   총 소요 시간: {total_duration:.1f}초 ({total_duration/60:.1f}분)",
            "",
            f"🎯 프로덕션 준비도 점수: {production_score:.1%}",
            f"🏅 준비도 등급: {readiness_grade}",
            "=" * 100
        ])
        
        # 권장사항
        report_lines.extend([
            "",
            "📋 다음 단계 권장사항:",
            ""
        ])
        
        if production_score >= 0.90:
            report_lines.extend([
                "✅ 프로덕션 배포 준비 완료!",
                "",
                "🚀 권장 다음 단계:",
                "   1. ./scripts/train_stage3.sh 실행으로 실제 학습 시작",
                "   2. 학습 진행 모니터링: ./scripts/monitoring/universal_training_monitor.sh --stage 3",
                "   3. 학습 완료 후 Stage 4 Two-Stage 통합 준비",
                ""
            ])
        else:
            report_lines.extend([
                "⚠️ 프로덕션 배포 전 추가 작업 필요",
                "",
                "🔧 우선 해결 항목:"
            ])
            
            for category, results in self.test_results.items():
                if results:
                    failed_tests = [r for r in results if not r['success']]
                    if failed_tests:
                        category_name = category_info.get(category, (category, ''))[0]
                        report_lines.append(f"   - {category_name}: {len(failed_tests)}개 실패 테스트 수정")
            
            report_lines.extend([
                "",
                "🔄 수정 후 재테스트:",
                "   python scripts/testing/run_stage3_test_suite.py",
                ""
            ])
        
        # 콘솔 출력
        for line in report_lines:
            print(line)
        
        # 파일 저장
        report_dir = Path("artifacts/stage3/test_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 텍스트 보고서
        report_file = report_dir / f"stage3_test_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # JSON 상세 보고서
        json_report = {
            'timestamp': datetime.now().isoformat(),
            'environment': {
                'cuda_available': self.cuda_available,
                'project_root': str(self.project_root)
            },
            'test_results': self.test_results,
            'summary': {
                'total_tests': total_tests,
                'total_success': total_success,
                'success_rate': success_rate,
                'total_duration': total_duration,
                'production_score': production_score,
                'production_ready': production_score >= 0.90
            }
        }
        
        json_file = report_dir / f"stage3_test_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📄 텍스트 보고서: {report_file}")
        self.logger.info(f"📄 JSON 보고서: {json_file}")
        
        return production_score >= 0.90
    
    def run_full_test_suite(self):
        """전체 테스트 스위트 실행"""
        self.logger.info("🚀 Stage 3 Classification 테스트 스위트 시작")
        start_time = time.time()
        
        # 1. 단위 테스트
        self.run_unit_tests()
        
        # 2. 통합 테스트
        self.run_integration_tests()
        
        # 3. 성능 테스트
        self.run_performance_tests()
        
        # 4. GPU 테스트 (CUDA 사용 가능시)
        self.run_gpu_tests()
        
        # 5. 장시간 안정성 테스트 (선택적)
        self.run_long_stability_tests()
        
        # 6. 종합 보고서 생성
        success = self.generate_comprehensive_report()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        if success:
            self.logger.info(f"🎉 Stage 3 테스트 스위트 완료 - 프로덕션 준비됨! ({total_duration:.1f}초)")
            return 0
        else:
            self.logger.error(f"💥 Stage 3 테스트 스위트 미완료 - 추가 작업 필요 ({total_duration:.1f}초)")
            return 1


def main():
    """메인 함수"""
    print("🎯 Stage 3 Classification 프로덕션급 테스트 스위트")
    print("=" * 80)
    print("목적: 프로덕션 배포 전 철저한 검증")
    print("범위: 단위/통합/성능/GPU 테스트")
    print("=" * 80)
    print()
    
    runner = Stage3TestSuiteRunner()
    exit_code = runner.run_full_test_suite()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()