#!/usr/bin/env python
"""
Stage 1 실제 이미지 파이프라인 테스트
- Stage 1 샘플 데이터로 Single/Combo 모드 테스트
- 실제 약품 이미지 처리 검증
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import os

# 환경 설정
os.environ['PILLSNAP_DATA_ROOT'] = '/mnt/data/pillsnap_dataset'

# 모델 import
from src.models.pipeline import create_pillsnap_pipeline
from src.data.sampling import Stage1SamplingStrategy, ProgressiveValidationSampler
from src.utils.core import load_config, PillSnapLogger

logger = PillSnapLogger(__name__)

def test_stage1_pipeline():
    """Stage 1 파이프라인 테스트"""
    
    # Stage 1 샘플 데이터 로드
    config = load_config()
    strategy = Stage1SamplingStrategy()
    data_root = config.get('data', {}).get('root', '/mnt/data/pillsnap_dataset')
    sampler = ProgressiveValidationSampler(data_root, strategy)
    samples = sampler.generate_stage1_sample()
    
    # 첫 번째 이미지 경로 가져오기
    first_k_code = list(samples['samples'].keys())[0]
    first_sample = samples['samples'][first_k_code]
    image_path = first_sample['single_images'][0] if first_sample['single_images'] else first_sample['combo_images'][0]
    
    logger.info(f'테스트 이미지: {image_path}')
    logger.info(f'K-code: {first_k_code}')
    
    # 파이프라인 생성 (테스트용 작은 설정)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = create_pillsnap_pipeline(
        default_mode='single',
        device=device,
        num_classes=50,  # Stage 1용
        detector_input_size=640,
        classifier_input_size=384
    )
    
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('RGB')
    logger.info(f'원본 이미지 크기: {image.size}')
    
    # Single 모드용 전처리 (384x384)
    single_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Combo 모드용 전처리 (640x640)
    combo_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Single 모드 테스트
    logger.info('='*50)
    logger.info('Single 모드 테스트 시작...')
    single_tensor = single_transform(image).unsqueeze(0).to(device)
    logger.info(f'입력 텐서 크기: {single_tensor.shape}')
    
    start_time = time.time()
    single_result = pipeline.predict(single_tensor, mode='single')
    single_time = (time.time() - start_time) * 1000
    
    logger.info(f'Single 모드 완료: {single_time:.2f}ms')
    logger.info(f'결과 개수: {len(single_result)}')
    logger.info(f'타이밍: {single_result.timing}')
    
    # Single 모드 예측 결과
    single_predictions = single_result.get_predictions()
    if single_predictions:
        pred = single_predictions[0]
        logger.info(f'예측 클래스 ID: {pred["class_id"]}')
        logger.info(f'신뢰도: {pred["confidence"]:.4f}')
        logger.info(f'모드: {pred["mode"]}')
    
    # Combo 모드 테스트
    logger.info('='*50)
    logger.info('Combo 모드 테스트 시작...')
    combo_tensor = combo_transform(image).unsqueeze(0).to(device)
    logger.info(f'입력 텐서 크기: {combo_tensor.shape}')
    
    start_time = time.time()
    combo_result = pipeline.predict(combo_tensor, mode='combo')
    combo_time = (time.time() - start_time) * 1000
    
    logger.info(f'Combo 모드 완료: {combo_time:.2f}ms')
    logger.info(f'결과 개수: {len(combo_result)}')
    logger.info(f'타이밍: {combo_result.timing}')
    
    # Combo 모드 예측 결과
    combo_predictions = combo_result.get_predictions()
    if combo_predictions:
        for i, pred in enumerate(combo_predictions):
            logger.info(f'검출 {i+1}: 클래스 ID={pred["class_id"]}, 신뢰도={pred["confidence"]:.4f}')
            if 'bbox' in pred and pred['bbox']:
                logger.info(f'  BBox: {pred["bbox"]}')
            if 'detection_confidence' in pred:
                logger.info(f'  검출 신뢰도: {pred["detection_confidence"]:.4f}')
    else:
        logger.info('Combo 모드에서 검출된 객체 없음')
    
    logger.info('='*50)
    logger.info('✅ Stage 1 실제 이미지 파이프라인 테스트 완료')
    logger.info(f'Single 모드: {single_time:.2f}ms')
    logger.info(f'Combo 모드: {combo_time:.2f}ms')
    
    # 배치 테스트
    logger.info('='*50)
    logger.info('배치 처리 테스트 시작...')
    
    # 여러 이미지 가져오기
    batch_images = []
    batch_k_codes = []
    for k_code, sample_data in list(samples['samples'].items())[:3]:
        img_paths = sample_data['single_images'][:2] if sample_data['single_images'] else sample_data['combo_images'][:2]
        for img_path in img_paths[:2]:
            batch_images.append(img_path)
            batch_k_codes.append(k_code)
            if len(batch_images) >= 3:
                break
        if len(batch_images) >= 3:
            break
    
    # 배치 텐서 생성
    batch_tensors = []
    for img_path in batch_images:
        img = Image.open(img_path).convert('RGB')
        tensor = single_transform(img).to(device)
        batch_tensors.append(tensor)
    
    batch_tensor = torch.stack(batch_tensors)
    logger.info(f'배치 입력 크기: {batch_tensor.shape}')
    
    # 배치 예측
    start_time = time.time()
    batch_results = pipeline.predict_batch(batch_tensor, mode='single', batch_size=2)
    batch_time = (time.time() - start_time) * 1000
    
    logger.info(f'배치 처리 완료: {batch_time:.2f}ms for {len(batch_results)} images')
    logger.info(f'평균 시간: {batch_time/len(batch_results):.2f}ms per image')
    
    for i, (result, k_code) in enumerate(zip(batch_results, batch_k_codes)):
        predictions = result.get_predictions()
        if predictions:
            logger.info(f'이미지 {i+1} (K-{k_code}): 클래스 {predictions[0]["class_id"]}, 신뢰도 {predictions[0]["confidence"]:.4f}')
    
    logger.info('='*50)
    logger.info('✅ 모든 테스트 성공적으로 완료!')
    
    return True

if __name__ == "__main__":
    test_stage1_pipeline()