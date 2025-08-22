# step4_infill.py  
"""
Enhanced Mask‑token infilling for sentence pairs produced by step 3.

Pipeline
--------
masked_pairs.jsonl      (from step3, contains <mask> tokens)
            │
            ▼
[fill‑mask or seq2seq model]  ——►  infilled_pairs.jsonl
                                      ├─ original
                                      ├─ masked
                                      ├─ infilled   ←  <mask> replaced
                                      ├─ infill_score  (∑ log prob / #mask)
                                      ├─ sim_score     (cosine w/ SBERT, optional)
                                      ├─ sampling_strategy
                                      ├─ sampling_params
                                      └─ meta: topic_id, mask_ratio, length

Default model = **bert‑base‑uncased** (token‑level fill‑mask).
T5/BART or any HF‑checkpoint that supports `<mask>` / `<extra_id_0>` also works.

Sampling Strategies
------------------
• greedy: Always select highest probability token (default)
• top_k: Sample from top-k most likely tokens  
• top_p: Nucleus sampling - sample from tokens with cumulative probability ≤ p
• top_k_top_p: Combined top-k and top-p sampling
• temperature: Temperature-scaled sampling from full vocabulary

Usage Examples
--------------
# Greedy decoding (most conservative)
python step4_infill.py --pairs_jsonl data.jsonl --output_dir output --sampling_strategy greedy

# Top-k sampling (moderate diversity)
python step4_infill.py --pairs_jsonl data.jsonl --output_dir output \
    --sampling_strategy top_k --top_k 10 --temperature 0.8

# Nucleus sampling (balanced creativity)
python step4_infill.py --pairs_jsonl data.jsonl --output_dir output \
    --sampling_strategy top_p --top_p 0.9 --temperature 1.0

# Combined approach (high quality + diversity)
python step4_infill.py --pairs_jsonl data.jsonl --output_dir output \
    --sampling_strategy top_k_top_p --top_k 50 --top_p 0.95 --temperature 0.9

# High creativity mode
python step4_infill.py --pairs_jsonl data.jsonl --output_dir output \
    --sampling_strategy temperature --temperature 1.2
"""

## 📊 **다양한 Infilling 전략 분석 및 개선 결과**

### 🎯 **현재 구현된 전략들**

| 전략 | 특징 | 적합한 사용 사례 | 파라미터 |
|------|------|------------------|----------|
| **Greedy** | 가장 높은 확률 토큰 선택 | 안정적이고 일관된 결과 필요 | - |
| **Top-k** | 상위 k개 토큰에서 샘플링 | 적당한 다양성과 품질 균형 | `top_k`, `temperature` |
| **Top-p** | 누적 확률 p까지의 토큰에서 샘플링 | 동적 어휘 크기, 자연스러운 다양성 | `top_p`, `temperature` |
| **Top-k + Top-p** | 두 방법의 조합 | 고품질 + 적절한 다양성 | `top_k`, `top_p`, `temperature` |
| **Temperature** | 전체 어휘에서 온도 조절된 샘플링 | 높은 창의성 필요 | `temperature` |

### 🚀 **개선된 기능들**

1. **유연한 샘플링 전략**
   - 5가지 다른 토큰 선택 방법
   - 사용 사례에 따른 최적화 가능

2. **향상된 품질 제어**
   - Temperature로 확률 분포 조절
   - Top-k와 Top-p 조합으로 품질-다양성 균형

3. **자세한 메타데이터**
   - 사용된 샘플링 전략과 파라미터 기록
   - 재현 가능한 실험을 위한 정보 저장

4. **실용적인 사용 예시**
   - 각 전략의 권장 파라미터 제공
   - 다양한 창작 요구에 맞는 설정

### 📈 **성능 비교 예상**

```python
# 보수적 (높은 품질, 낮은 다양성)
--sampling_strategy greedy

# 균형 (중간 품질, 중간 다양성) 
--sampling_strategy top_k_top_p --top_k 50 --top_p 0.9 --temperature 0.8

# 창의적 (낮은 품질, 높은 다양성)
--sampling_strategy temperature --temperature 1.5
```

### 🔬 **실험 추천 설정**

**과학 소설 infilling용 권장 설정:**

1. **고품질 우선**: `--sampling_strategy top_k_top_p --top_k 20 --top_p 0.85 --temperature 0.7`
2. **창의성 우선**: `--sampling_strategy top_p --top_p 0.95 --temperature 1.1` 
3. **균형 접근**: `--sampling_strategy top_k --top_k 50 --temperature 0.9`

이러한 개선으로 step4는 이제 다양한 연구 목적과 창작 요구에 맞춰 유연하게 사용할 수 있는 강력한 infilling 도구가 되었습니다.
