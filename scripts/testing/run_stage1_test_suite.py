#!/usr/bin/env python3
"""
Stage 1 데이터 파이프라인 테스트 스위트 실행기

적정 수준의 테스트를 체계적으로 실행:
- 단위 테스트: 핵심 컴포넌트 개별 검증
- 스모크 테스트: 기본 기능 빠른 검증
- 통합 테스트: 시스템 간 상호작용 검증 (샘플 실행)
- 종합 보고서: 테스트 결과 요약
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.core import PillSnapLogger


class Stage1TestSuiteRunner:
    """Stage 1 테스트 스위트 실행기"""
    
    def __init__(self):
        self.logger = PillSnapLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent
        self.test_results = {}
        
        self.logger.info("Stage 1 테스트 스위트 실행기 초기화")
    
    def run_command(self, command: list, test_name: str, timeout: int = 300) -> dict:
        """명령어 실행 및 결과 수집"""
        self.logger.info(f"🔄 {test_name} 실행 중...")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
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
                self.logger.error(f"  Error: {result.stderr[-200:]}")  # 마지막 200자만
            
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
                'name': '샘플링 시스템 단위 테스트',
                'command': [
                    'bash', 'scripts/env/python_executor.sh', '-m', 'pytest',
                    'tests/unit/test_sampling.py::TestStage1SamplingStrategy',
                    '-v', '--tb=short'
                ],
                'timeout': 60
            },
            {
                'name': '의약품 레지스트리 단위 테스트',
                'command': [
                    'bash', 'scripts/env/python_executor.sh', '-m', 'pytest',
                    'tests/unit/test_pharmaceutical_code_registry.py::TestDrugIdentificationRecord',
                    '-v', '--tb=short'
                ],
                'timeout': 60
            }
        ]
        
        unit_results = []
        for test in unit_tests:
            result = self.run_command(test['command'], test['name'], test['timeout'])
            unit_results.append(result)
        
        self.test_results['unit_tests'] = unit_results
        
        success_count = sum(1 for r in unit_results if r['success'])
        self.logger.info(f"📋 단위 테스트 완료: {success_count}/{len(unit_results)} 성공")
    
    def run_smoke_tests(self):
        """스모크 테스트 실행"""
        self.logger.info("🚬 스모크 테스트 시작...")
        
        smoke_tests = [
            {
                'name': '샘플링 기본 기능 스모크',
                'command': [
                    'bash', 'scripts/env/python_executor.sh', '-m', 'pytest',
                    'tests/smoke/test_stage1_data_pipeline_smoke.py::TestStage1DataPipelineSmoke::test_smoke_sampling_basic_functionality',
                    '-v', '--tb=short'
                ],
                'timeout': 120
            },
            {
                'name': '레지스트리 기본 기능 스모크',
                'command': [
                    'bash', 'scripts/env/python_executor.sh', '-m', 'pytest',
                    'tests/smoke/test_stage1_data_pipeline_smoke.py::TestStage1DataPipelineSmoke::test_smoke_registry_basic_functionality',
                    '-v', '--tb=short'
                ],
                'timeout': 60
            },
            {
                'name': 'End-to-End 워크플로우 스모크',
                'command': [
                    'bash', 'scripts/env/python_executor.sh', '-m', 'pytest',
                    'tests/smoke/test_stage1_data_pipeline_smoke.py::TestStage1DataPipelineSmoke::test_smoke_end_to_end_minimal_workflow',
                    '-v', '--tb=short'
                ],
                'timeout': 120
            }
        ]
        
        smoke_results = []
        for test in smoke_tests:
            result = self.run_command(test['command'], test['name'], test['timeout'])
            smoke_results.append(result)
        
        self.test_results['smoke_tests'] = smoke_results
        
        success_count = sum(1 for r in smoke_results if r['success'])
        self.logger.info(f"🚬 스모크 테스트 완료: {success_count}/{len(smoke_results)} 성공")
    
    def run_integration_tests(self):
        """통합 테스트 실행 (샘플만)"""
        self.logger.info("🔗 통합 테스트 시작...")
        
        integration_tests = [
            {
                'name': '샘플링-레지스트리 데이터 일관성',
                'command': [
                    'bash', 'scripts/env/python_executor.sh', '-m', 'pytest',
                    'tests/integration/test_stage1_sampling_registry_integration.py::TestStage1SamplingRegistryIntegration::test_sampling_to_registry_data_consistency',
                    '-v', '--tb=short'
                ],
                'timeout': 180
            }
        ]
        
        integration_results = []
        for test in integration_tests:
            result = self.run_command(test['command'], test['name'], test['timeout'])
            integration_results.append(result)
        
        self.test_results['integration_tests'] = integration_results
        
        success_count = sum(1 for r in integration_results if r['success'])
        self.logger.info(f"🔗 통합 테스트 완료: {success_count}/{len(integration_results)} 성공")
    
    def generate_test_report(self):
        """테스트 결과 종합 보고서 생성"""
        self.logger.info("📊 테스트 결과 보고서 생성...")
        
        total_tests = 0
        total_success = 0
        total_duration = 0
        
        report_lines = [
            "=" * 80,
            f"Stage 1 데이터 파이프라인 테스트 결과 보고서",
            f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            ""
        ]
        
        for category, results in self.test_results.items():
            category_korean = {
                'unit_tests': '📋 단위 테스트',
                'smoke_tests': '🚬 스모크 테스트',
                'integration_tests': '🔗 통합 테스트'
            }.get(category, category)
            
            category_success = sum(1 for r in results if r['success'])
            category_duration = sum(r['duration'] for r in results)
            
            total_tests += len(results)
            total_success += category_success
            total_duration += category_duration
            
            report_lines.extend([
                f"{category_korean} 결과:",
                f"  성공: {category_success}/{len(results)}",
                f"  소요 시간: {category_duration:.1f}초",
                ""
            ])
            
            for result in results:
                status = "✅" if result['success'] else "❌"
                report_lines.append(f"  {status} {result['name']} ({result['duration']:.1f}초)")
            
            report_lines.append("")
        
        # 전체 요약
        success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        overall_status = "✅ 통과" if success_rate >= 80 else "❌ 실패"
        
        report_lines.extend([
            "=" * 80,
            f"📊 전체 요약:",
            f"  총 테스트: {total_tests}개",
            f"  성공: {total_success}개",
            f"  실패: {total_tests - total_success}개",
            f"  성공률: {success_rate:.1f}%",
            f"  총 소요 시간: {total_duration:.1f}초",
            f"  전체 상태: {overall_status}",
            "=" * 80
        ])
        
        # 콘솔 출력
        for line in report_lines:
            print(line)
        
        # 파일 저장
        report_path = Path("artifacts/stage1/test_reports")
        report_path.mkdir(parents=True, exist_ok=True)
        
        report_file = report_path / f"stage1_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"📄 보고서 저장: {report_file}")
        
        return success_rate >= 80
    
    def run_full_test_suite(self):
        """전체 테스트 스위트 실행"""
        self.logger.info("🚀 Stage 1 테스트 스위트 시작")
        start_time = time.time()
        
        # 1. 단위 테스트
        self.run_unit_tests()
        
        # 2. 스모크 테스트
        self.run_smoke_tests()
        
        # 3. 통합 테스트 (샘플)
        self.run_integration_tests()
        
        # 4. 보고서 생성
        success = self.generate_test_report()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        if success:
            self.logger.info(f"🎉 Stage 1 테스트 스위트 완료 ({total_duration:.1f}초)")
            return 0
        else:
            self.logger.error(f"💥 Stage 1 테스트 스위트 실패 ({total_duration:.1f}초)")
            return 1


def main():
    """메인 함수"""
    print("🏥 Stage 1 데이터 파이프라인 테스트 스위트")
    print("=" * 60)
    
    runner = Stage1TestSuiteRunner()
    exit_code = runner.run_full_test_suite()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()