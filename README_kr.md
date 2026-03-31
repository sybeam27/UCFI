# Beyond Uniform Fairness: Node-Aware Multi-Level Fair GNNs

# 하이퍼파라미터 설정 가이드

## 공통 하이퍼파라미터 (Baseline / FnCGNN / FnRGNN / NAFnCGNN / NAFnRGNN 전체 적용)

| 파라미터 | 현재 설정값 | 권장 탐색 범위 | 의미 |
|---|---|---|---|
| `hidden_dim` | 128 | 64, 128, 256 | GNN hidden layer 차원. 클수록 표현력 증가, 과적합 위험 증가 |
| `dropout` | 0.1 | 0.1, 0.3, 0.5 | 과적합 방지. pokec처럼 대규모 그래프에서는 0.3 권장 |
| `lr` | 1e-3 | 1e-4, 5e-4, 1e-3 | 학습률. 낮을수록 안정적이나 수렴 느림 |
| `epochs` | 1000 | 500~2000 | 최대 학습 epoch 수 |
| `patience` | 100 | 50~200 | early stopping 기준. fairness 모델은 100 이상 권장 |
| `seed` | 27 | 임의 정수 | 재현성 고정용. runs 실행 시 seed+i로 자동 증가 |
| `runs` | 5 | 3~10 | 반복 실행 횟수. 평균±표준편차 계산에 사용 |
| `sgc_k` | 2 | 1, 2, 3 | SGC backbone 전용. propagation hop 수 |

---

## 분류 태스크 (Classification)

### Baseline

fairness 제약 없이 순수 task loss만으로 학습하는 참조 모델.

| 파라미터 | 현재 설정값 | 권장 탐색 범위 | 의미 |
|---|---|---|---|
| `weight_decay` | 0.0 | 0.0, 1e-5 | L2 정규화. 분류에서는 0이 일반적 |
| `hidden_dim` | 128 | 64, 128, 256 | — |
| `dropout` | 0.1 | 0.1, 0.3 | — |

val_score 기준: `acc` (정확도 최대화만 고려)

---

### FnCGNN

3단계(Structure / Representation / Output) 공정성 제약을 **모든 노드에 균일하게** 적용하는 분류 모델.

| 파라미터 | 현재 설정값 | 권장 탐색 범위 | 의미 |
|---|---|---|---|
| `weight_decay` | 0.0 | 0.0, 1e-5 | 분류 태스크 기본값 0.0 |
| `lambda_struct` | 0.001 | 0.0001, 0.001, 0.01 | Structure loss 가중치. hidden_dim이 클수록 작게 설정 |
| `lambda_rep` | 0.01 | 0.001, 0.01, 0.05 | Representation loss 가중치. GroupWiseNorm 강도 조절 |
| `lambda_out` | 0.1 | 0.05, 0.1, 0.5 | Output loss 가중치. DP+EO surrogate 강도. 가장 직접적인 공정성 제약 |
| `drop_edge_rate_struct` | 0.1 | 0.05, 0.1, 0.2 | Structure loss 계산 시 edge dropout 비율. 높을수록 perturbation 강도 증가 |
| `val_tradeoff_dp` | 0.3 | 0.1, 0.3, 0.5 | val_score에서 DP 패널티 강도. 높을수록 공정성 우선, 낮을수록 accuracy 우선 |
| `val_tradeoff_eo` | 0.3 | 0.1, 0.3, 0.5 | val_score에서 EO 패널티 강도. val_tradeoff_dp와 동일한 역할 |
| `ablate_struct` | False | True / False | True로 설정 시 Structure loss 비활성화 (ablation study용) |
| `ablate_rep` | False | True / False | True로 설정 시 Representation loss 비활성화 |
| `ablate_out` | False | True / False | True로 설정 시 Output loss 비활성화 |

val_score 기준: `acc - val_tradeoff_dp × |DP| - val_tradeoff_eo × |EO|`

---

### NAFnCGNN

FnCGNN의 3단계 공정성 제약을 **노드 유형별 차등 강도**로 적용하는 분류 모델.
허브/경계 노드에 더 강한 제약, 고립 노드에 완화된 제약을 부여.

**FnCGNN과 동일한 파라미터** (위 표 참고) 외에 아래 노드 분류 파라미터가 추가됨.

| 파라미터 | 현재 설정값 | 권장 탐색 범위 | 의미 |
|---|---|---|---|
| `hub_percentile` | 80 | 70, 80, 90 | degree 상위 몇 %를 허브 노드로 분류할지. 높을수록 허브 노드 수 감소 |
| `isolate_percentile` | 20 | 10, 20, 30 | degree 하위 몇 %를 고립 노드로 분류할지. 낮을수록 고립 노드 수 감소 |
| `boundary_threshold` | 0.3 | 0.2, 0.3, 0.5 | 이웃 중 타 그룹 비율이 이 값 이상이면 경계 노드. 낮을수록 경계 노드 수 증가 |
| `hub_weight` | 2.0 | 1.5, 2.0, 3.0 | 허브 노드의 fairness 개입 강도. 높을수록 허브 노드에 강한 제약 |
| `boundary_weight` | 1.5 | 1.2, 1.5, 2.0 | 경계 노드의 fairness 개입 강도. hub_weight보다 작게 유지 권장 |
| `isolate_weight` | 0.5 | 0.3, 0.5, 0.8 | 고립 노드의 fairness 개입 강도. 1.0보다 작게 설정하여 제약 완화 |

> **노드 유형 우선순위**: 허브 > 경계 > 고립 > 일반 (중복 시 높은 우선순위 적용)

val_score 기준: `acc - val_tradeoff_dp × |DP| - val_tradeoff_eo × |EO|` (FnCGNN과 동일)

---

## 회귀 태스크 (Regression)

### Baseline

| 파라미터 | 현재 설정값 | 권장 탐색 범위 | 의미 |
|---|---|---|---|
| `weight_decay` | 1e-5 | 1e-5, 1e-4 | 회귀에서는 작은 L2 정규화 권장 |
| `hidden_dim` | 128 | 64, 128, 256 | — |
| `dropout` | 0.1 | 0.1, 0.3 | — |

val_score 기준: `-MAE` (MAE 최소화)

---

### FnRGNN

3단계 공정성 제약을 **모든 노드에 균일하게** 적용하는 회귀 모델.

| 파라미터 | 현재 설정값 | 권장 탐색 범위 | 의미 |
|---|---|---|---|
| `weight_decay` | 1e-5 | 1e-5, 1e-4 | 회귀 태스크 기본값 1e-5 |
| `lambda_struct` | 0.001 | 0.0001, 0.001, 0.01 | Structure loss 가중치. MSE 스케일이 타겟 범위에 따라 달라지므로 타겟 정규화 여부 확인 필요 |
| `lambda_rep` | 0.01 | 0.001, 0.01, 0.05 | Representation loss 가중치 |
| `lambda_out` | 0.1 | 0.05, 0.1, 0.5 | Output loss 가중치. mean_pred_gap + bias_gap 강도 조절 |
| `drop_edge_rate_struct` | 0.1 | 0.05, 0.1, 0.2 | Structure loss 계산 시 edge dropout 비율 |
| `val_tradeoff_mae` | 1.0 | 0.5, 1.0, 2.0 | val_score에서 MAE 반영 비중. 높을수록 예측 정확도 우선 |
| `val_tradeoff_bias` | 1.0 | 0.5, 1.0, 2.0 | val_score에서 그룹 간 bias gap 패널티 강도 |
| `val_tradeoff_mean_pred` | 0.5 | 0.1, 0.5, 1.0 | val_score에서 그룹 간 평균 예측값 차이 패널티 강도 |
| `ablate_struct` | False | True / False | Structure loss 비활성화 |
| `ablate_rep` | False | True / False | Representation loss 비활성화 |
| `ablate_out` | False | True / False | Output loss 비활성화 |

val_score 기준: `-(val_tradeoff_mae × MAE + val_tradeoff_bias × bias_gap + val_tradeoff_mean_pred × mean_pred_gap)`

---

### NAFnRGNN

FnRGNN의 3단계 공정성 제약을 **노드 유형별 차등 강도**로 적용하는 회귀 모델.

**FnRGNN과 동일한 파라미터** (위 표 참고) 외에 아래 노드 분류 파라미터가 추가됨.

| 파라미터 | 현재 설정값 | 권장 탐색 범위 | 의미 |
|---|---|---|---|
| `hub_percentile` | 80 | 70, 80, 90 | degree 상위 몇 %를 허브 노드로 분류 |
| `isolate_percentile` | 20 | 10, 20, 30 | degree 하위 몇 %를 고립 노드로 분류 |
| `boundary_threshold` | 0.3 | 0.2, 0.3, 0.5 | 타 그룹 이웃 비율 기준값. 이 값 이상이면 경계 노드 |
| `hub_weight` | 2.0 | 1.5, 2.0, 3.0 | 허브 노드 fairness 개입 강도 |
| `boundary_weight` | 1.5 | 1.2, 1.5, 2.0 | 경계 노드 fairness 개입 강도 |
| `isolate_weight` | 0.5 | 0.3, 0.5, 0.8 | 고립 노드 fairness 개입 강도 (완화) |

val_score 기준: `-(val_tradeoff_mae × MAE + val_tradeoff_bias × bias_gap + val_tradeoff_mean_pred × mean_pred_gap)` (FnRGNN과 동일)

---

## 3단계 공정성 Loss 구조 요약

| 레벨 | Loss 계산 방식 | FnCGNN / FnRGNN | NAFnCGNN / NAFnRGNN |
|---|---|---|---|
| Structure | edge perturbation 후 hidden representation MSE | 모든 노드 균일 평균 | 노드별 가중 평균 (허브↑ 고립↓) |
| Representation | 그룹 간 hidden representation 분포 차이 (GroupWiseNorm) | 단순 평균/분산 차이 | 가중 평균/분산 차이 (WeightedGroupWiseNorm) |
| Output | 그룹 간 예측값 편향 (분류: DP+EO / 회귀: bias_gap+mean_pred_gap) | 단순 그룹 평균 | 노드별 가중 그룹 평균 |

---

## val_score 기준 요약

| 모델 | val_score 수식 | 높을수록 |
|---|---|---|
| Baseline (분류) | `acc` | 정확도 높음 |
| Baseline (회귀) | `-MAE` | 오차 낮음 |
| FnCGNN | `acc - α×\|DP\| - β×\|EO\|` | 정확도 높고 공정성 좋음 |
| FnRGNN | `-(γ×MAE + δ×bias_gap + ε×mean_pred_gap)` | 오차 낮고 공정성 좋음 |
| NAFnCGNN | `acc - α×\|DP\| - β×\|EO\|` | FnCGNN과 동일 기준 |
| NAFnRGNN | `-(γ×MAE + δ×bias_gap + ε×mean_pred_gap)` | FnRGNN과 동일 기준 |