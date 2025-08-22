#!/usr/bin/env python3
"""
토픽 모델링 결과를 자세히 확인하는 스크립트
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# 결과 디렉토리 설정
result_dir = Path("output/kmeans_20_topics")

print("🔍 토픽 모델링 결과 분석")
print("="*50)

# 1. 토픽 모델 로드
print("\n📊 1. 토픽 모델 기본 정보")
with open(result_dir / "topic_model.pkl", "rb") as f:
    topic_model = pickle.load(f)

with open(result_dir / "topic_assignments.pkl", "rb") as f:
    topics, probs = pickle.load(f)

print(f"총 토픽 개수: {len(topic_model.get_topics())}")
print(f"문서 개수: {len(topics)}")
print(f"확률 계산 여부: {'Yes' if probs is not None else 'No'}")

# 2. 토픽 정보 테이블
print("\n📋 2. 토픽별 문서 개수")
topic_info = topic_model.get_topic_info()
print(topic_info[['Topic', 'Count', 'Name']].head(10))

# 3. 각 토픽의 대표 키워드 (상위 10개)
print("\n🔑 3. 각 토픽의 대표 키워드 (상위 10개)")
print("-" * 80)
for topic_id in range(min(10, len(topic_info))):  # 처음 10개 토픽만
    if topic_id >= 0:  # 유효한 토픽만
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            keywords = [f"{word}({score:.3f})" for word, score in topic_words[:10]]
            print(f"토픽 {topic_id:2d}: {', '.join(keywords)}")

# 4. 토픽 분포 통계
print("\n📈 4. 토픽 분포 통계")
topic_counts = pd.Series(topics).value_counts().sort_index()
print(f"가장 큰 토픽: {topic_counts.idxmax()} ({topic_counts.max()}개 문서)")
print(f"가장 작은 토픽: {topic_counts.idxmin()} ({topic_counts.min()}개 문서)")
print(f"평균 토픽 크기: {topic_counts.mean():.1f}개 문서")
print(f"토픽 크기 표준편차: {topic_counts.std():.1f}")

# 5. 토픽별 상위 문서 샘플 (처음 3개 토픽)
print("\n📝 5. 토픽별 대표 문서 샘플")
print("-" * 80)

# 전처리된 텍스트 로드
with open(result_dir / "texts_clean.json", "r", encoding="utf-8") as f:
    texts_clean = f.read()[:1000] + "..." if len(f.read()) > 1000 else f.read()

# 원본 문서 로드 (처음 몇 개만)
import json
with open("/Users/sujinkkang/Dropbox/Amazing Stories Project/output/processed_stories_by_paragraph.json", "r", encoding="utf-8") as f:
    original_docs = json.load(f)

for topic_id in range(min(3, len(topic_info))):
    if topic_id >= 0:
        # 해당 토픽에 속하는 문서 인덱스 찾기
        topic_docs_idx = [i for i, t in enumerate(topics) if t == topic_id][:3]  # 상위 3개만
        
        print(f"\n🎯 토픽 {topic_id} 대표 문서:")
        for i, doc_idx in enumerate(topic_docs_idx):
            if doc_idx < len(original_docs):
                doc_text = original_docs[doc_idx]["paragraph_text"]
                # 텍스트가 너무 길면 자르기
                if len(doc_text) > 200:
                    doc_text = doc_text[:200] + "..."
                print(f"  {i+1}. {doc_text}")

print("\n✅ 분석 완료!")
print(f"더 자세한 정보는 {result_dir} 디렉토리의 파일들을 확인하세요.")
