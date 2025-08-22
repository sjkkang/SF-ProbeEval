#!/usr/bin/env python3
"""
최종 결과 파일 분석 및 정리 스크립트
- infilled_pairs.csv 파일을 분석하여 구조를 정리
- sent1, sent2, pair, score 형태로 정리
- 실패한 페어 제거 버전과 전체 버전 두 가지 생성
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_and_analyze_results(input_file):
    """결과 파일을 로드하고 기본 분석 수행"""
    print(f"Loading results from: {input_file}")
    
    # CSV 파일 로드
    df = pd.read_csv(input_file)
    
    print(f"\n=== 기본 정보 ===")
    print(f"총 행 수: {len(df):,}")
    print(f"컬럼 수: {len(df.columns)}")
    print(f"컬럼명: {list(df.columns)}")
    
    print(f"\n=== 데이터 타입 ===")
    print(df.dtypes)
    
    print(f"\n=== 결측값 확인 ===")
    print(df.isnull().sum())
    
    print(f"\n=== 처음 5행 샘플 ===")
    print(df.head())
    
    return df

def analyze_scores(df):
    """점수 분포 분석"""
    print(f"\n=== 점수 분석 ===")
    
    # 점수가 있는 행과 없는 행 구분
    if 'similarity_score' in df.columns:
        score_col = 'similarity_score'
    elif 'score' in df.columns:
        score_col = 'score'
    else:
        print("점수 컬럼을 찾을 수 없습니다.")
        return df, 0, 0
    
    valid_scores = df[df[score_col].notna()]
    invalid_scores = df[df[score_col].isna()]
    
    print(f"유효한 점수가 있는 행: {len(valid_scores):,}")
    print(f"점수가 없는 행: {len(invalid_scores):,}")
    
    if len(valid_scores) > 0:
        print(f"\n점수 통계:")
        print(f"평균: {valid_scores[score_col].mean():.3f}")
        print(f"중앙값: {valid_scores[score_col].median():.3f}")
        print(f"표준편차: {valid_scores[score_col].std():.3f}")
        print(f"최솟값: {valid_scores[score_col].min():.3f}")
        print(f"최댓값: {valid_scores[score_col].max():.3f}")
        
        print(f"\n점수 분포:")
        score_ranges = [
            (0.0, 1.0, "매우 낮음 (0.0-1.0)"),
            (1.0, 2.0, "낮음 (1.0-2.0)"),
            (2.0, 3.0, "보통 (2.0-3.0)"),
            (3.0, 4.0, "높음 (3.0-4.0)"),
            (4.0, 5.0, "매우 높음 (4.0-5.0)")
        ]
        
        for min_val, max_val, label in score_ranges:
            count = len(valid_scores[(valid_scores[score_col] >= min_val) & (valid_scores[score_col] < max_val)])
            percentage = count / len(valid_scores) * 100
            print(f"{label}: {count:,}개 ({percentage:.1f}%)")
    
    return valid_scores, invalid_scores, score_col

def restructure_data(df, score_col):
    """데이터를 required 구조로 재구성"""
    print(f"\n=== 데이터 재구성 ===")
    
    # 컬럼명 확인 및 매핑
    column_mapping = {}
    
    # sent1, sent2 컬럼 찾기
    possible_sent1_cols = ['sent1', 'sentence1', 'original_text', 'text1']
    possible_sent2_cols = ['sent2', 'sentence2', 'infilled_text', 'text2']
    
    sent1_col = None
    sent2_col = None
    
    for col in possible_sent1_cols:
        if col in df.columns:
            sent1_col = col
            break
    
    for col in possible_sent2_cols:
        if col in df.columns:
            sent2_col = col
            break
    
    if sent1_col is None or sent2_col is None:
        print("Error: sent1 또는 sent2 컬럼을 찾을 수 없습니다.")
        print(f"사용 가능한 컬럼: {list(df.columns)}")
        return None
    
    print(f"sent1 컬럼: {sent1_col}")
    print(f"sent2 컬럼: {sent2_col}")
    print(f"score 컬럼: {score_col}")
    
    # 새로운 데이터프레임 생성
    new_df = pd.DataFrame()
    
    # sent1, sent2 복사
    new_df['sent1'] = df[sent1_col].copy()
    new_df['sent2'] = df[sent2_col].copy()
    
    # pair 컬럼 생성 (sent1 + [SEP] + sent2)
    new_df['pair'] = new_df['sent1'].astype(str) + ' [SEP] ' + new_df['sent2'].astype(str)
    
    # score 컬럼 복사
    new_df['score'] = df[score_col].copy()
    
    print(f"재구성된 데이터 크기: {len(new_df):,} 행 x {len(new_df.columns)} 열")
    
    return new_df

def save_results(df_all, df_valid, output_dir):
    """결과를 두 가지 버전으로 저장"""
    print(f"\n=== 결과 저장 ===")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 전체 버전 저장 (실패한 페어 포함)
    all_file = os.path.join(output_dir, 'final_pairs_all.csv')
    df_all.to_csv(all_file, index=False, encoding='utf-8')
    print(f"전체 버전 저장: {all_file}")
    print(f"  - 총 {len(df_all):,}개 페어")
    print(f"  - 유효한 점수: {len(df_all[df_all['score'].notna()]):,}개")
    print(f"  - 무효한 점수: {len(df_all[df_all['score'].isna()]):,}개")
    
    # 유효한 페어만 저장 (실패한 페어 제거)
    valid_file = os.path.join(output_dir, 'final_pairs_valid.csv')
    df_valid.to_csv(valid_file, index=False, encoding='utf-8')
    print(f"\n유효한 페어만 저장: {valid_file}")
    print(f"  - 총 {len(df_valid):,}개 페어")
    print(f"  - 평균 점수: {df_valid['score'].mean():.3f}")
    print(f"  - 제거된 페어: {len(df_all) - len(df_valid):,}개")
    
    return all_file, valid_file

def show_samples(df, n=3):
    """샘플 데이터 표시"""
    print(f"\n=== 샘플 데이터 (상위 {n}개) ===")
    
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        print(f"\n--- 샘플 {i+1} ---")
        print(f"sent1: {row['sent1'][:100]}...")
        print(f"sent2: {row['sent2'][:100]}...")
        print(f"score: {row['score']}")
        print(f"pair 길이: {len(row['pair'])} 문자")

def main():
    """메인 함수"""
    # 파일 경로 설정
    input_file = 'output/enhanced_preprocessing_topics_v3/infilled_pairs.csv'
    output_dir = 'output/enhanced_preprocessing_topics_v3/final_analysis'
    
    # 입력 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"Error: 입력 파일을 찾을 수 없습니다: {input_file}")
        return
    
    try:
        # 1. 데이터 로드 및 기본 분석
        df_original = load_and_analyze_results(input_file)
        
        # 2. 점수 분석
        valid_scores_df, invalid_scores_df, score_col = analyze_scores(df_original)
        
        if score_col == 0:  # 점수 컬럼을 찾지 못한 경우
            return
        
        # 3. 데이터 재구성
        df_restructured = restructure_data(df_original, score_col)
        
        if df_restructured is None:
            return
        
        # 4. 유효한 페어만 필터링
        df_valid = df_restructured[df_restructured['score'].notna()].copy()
        
        # 5. 결과 저장
        all_file, valid_file = save_results(df_restructured, df_valid, output_dir)
        
        # 6. 샘플 표시
        if len(df_valid) > 0:
            show_samples(df_valid)
        
        print(f"\n=== 완료 ===")
        print(f"결과 파일이 {output_dir} 폴더에 저장되었습니다.")
        print(f"- 전체 버전: final_pairs_all.csv")
        print(f"- 유효한 페어만: final_pairs_valid.csv")
        
    except Exception as e:
        print(f"Error: 처리 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
