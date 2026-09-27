from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'battery_results'
d = pd.read_csv(ROOT / 'upload/feature_all.csv')
p = pd.read_csv(OUT / 'holdout_predictions.csv')
s = pd.read_csv(OUT / 'model_search.csv')
imp = pd.read_csv(OUT / 'holdout_permutation_importance.csv')
m = json.loads((OUT / 'metrics.json').read_text())

fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
ax[0, 0].hist(d.Lifetime, bins=20, color='#3572a5', edgecolor='white')
ax[0, 0].set(xlabel='Lifetime (source unit)', ylabel='Cells', title='Target distribution')
ax[0, 1].scatter(d.DoD, d.Lifetime, alpha=.7, s=22, c='#347d64')
ax[0, 1].set(xlabel='DoD', ylabel='Lifetime', title='Depth of discharge vs lifetime')
ax[1, 0].scatter(p.actual, p.predicted, c='#c0703a', s=35)
lims = [min(p.actual.min(), p.predicted.min()), max(p.actual.max(), p.predicted.max())]
ax[1, 0].plot(lims, lims, '--', c='gray')
ax[1, 0].set(xlabel='Actual', ylabel='Predicted', title='Unseen groups: final holdout')
top = imp.head(8).iloc[::-1]
ax[1, 1].barh(top.feature, top.importance_MAE, color='#6774a3')
ax[1, 1].set(xlabel='MAE increase after permutation', title='Holdout sensitivity (unstable with 47 cells)')
fig.savefig(OUT / 'eda_and_validation.png', dpi=170)
plt.close(fig)

# Cluster bootstrap quantifies how much a particular set of 12 held-out groups
# affects MAE. It is descriptive and does not repair selection or domain shift.
rng = np.random.default_rng(42)
groups = p.Group.unique()
errors = []
for _ in range(4000):
    sampled = rng.choice(groups, size=len(groups), replace=True)
    errors.append(np.concatenate([p.loc[p.Group.eq(g), 'absolute_error'].to_numpy() for g in sampled]).mean())
lo, hi = np.quantile(errors, [.025, .975])
worst = p.groupby('Group').absolute_error.mean().sort_values(ascending=False)
mae_without_g1 = p.loc[p.Group.ne('G1'), 'absolute_error'].mean()

body = f'''# 배터리 셀 수명 예측 분석

## 데이터와 질문

- 파일: `feature_all.csv`, {len(d)}개 셀, {d.Group.nunique()}개 실험군, 열 {len(d.columns)}개.
- 타깃: `Lifetime` 연속값. **단위와 수명 종료 기준은 CSV만으로 확인되지 않으므로** 오차도 원본 `Lifetime` 단위로 표현한다.
- 식별자 `Cell`, `Group`과 타깃은 입력 피처에서 제외했다. `Group`은 같은 실험 조건의 반복 측정을 묶어 검증에 사용했다.
- 누락값, 중복 셀, 완전 중복 행은 없다. `Lifetime`의 중앙값 {d.Lifetime.median():.3f}, 범위 {d.Lifetime.min():.3f}–{d.Lifetime.max():.3f}; 높은 값의 꼬리가 길다.
- `delta_CV_time_3_0`는 두 CV 시간의 차이, `avg_stress`는 충·방전 스트레스 평균, `multi_stress`는 두 스트레스의 곱으로 수치상 재현된다. 동시 투입 시 정보가 중복될 수 있다.

![분포, DoD 관계, 예측 검증, 변수 민감도](eda_and_validation.png)

## 검증 설계

`GroupShuffleSplit(random_state=42)`으로 {m['train_groups']}개 군 {m['train_n']}개 셀을 탐색에 쓰고, **별도의 {m['test_groups']}개 군 {m['test_n']}개 셀**을 마지막 평가에 사용했다. 훈련 군에만 4분할 `GroupKFold`를 적용하여 MAE가 가장 낮은 조합을 선택했다. 보류군은 튜닝에 사용하지 않았다.

## 피처 엔지니어링과 탐색

| 피처 조합 | 내용 |
|---|---|
| operational | 충·방전 조건, DoD, 초기 용량과 전압 구간, 최초 CV 시간, 스트레스 (초기 3회 변화량 제외) |
| lean | 조건·DoD·초기 용량·CV 시간 0/3·용량 감소·방전 변화량·중간 전압 dQ/dV·평균 스트레스 10개 |
| raw | 식별자와 타깃 제외 모든 수치 29개 |
| engineered | raw + DoD 대비 감소량, CV 시간비, C-rate 비와 합, 용량 구간 차이, DVA 절댓값 합, dQ/dV 및 스트레스 차이 등 11개 |

Ridge, RBF-SVR, ExtraTrees, HistGradientBoosting의 하이퍼파라미터 격자를 조합마다 탐색했다. 상위 결과는 다음과 같다. 점수는 **훈련 군 내부**의 평균 그룹 CV MAE다.

| 피처 | 모델 | CV MAE |
|---|---|---:|
'''
for r in s.head(8).itertuples():
    body += f'| {r.variant} | {r.model} | {r.group_cv_MAE:.3f} |\n'
body += f'''
선택 모델: **lean + RBF-SVR**, `C=10`, `epsilon=1`, `gamma=0.01`; 입력은 표준화했고, 모든 전처리는 학습 fold 내부에서만 적합했다. 이 탐색에서는 복잡한 파생 피처가 MAE를 줄이지 못했다. `operational` 최상위 CV MAE는 {s.loc[s.variant.eq('operational'), 'group_cv_MAE'].min():.3f}로, 초기 3회 자료가 없을 때 참고할 수 있다. 다만 초기 3회 지표의 생성 시점과 실사용 시점이 같아야 한다.

## 마지막 평가: 보지 않은 실험군

| 방법 | MAE | RMSE | R² |
|---|---:|---:|---:|
| 중앙값 기준선 | {m['baseline_test']['MAE']:.3f} | {m['baseline_test']['RMSE']:.3f} | {m['baseline_test']['R2']:.3f} |
| 선택 모델 | **{m['test']['MAE']:.3f}** | {m['test']['RMSE']:.3f} | {m['test']['R2']:.3f} |

훈련 군의 4분할 그룹 CV 재예측 MAE는 {m['train_group_oof']['MAE']:.3f}이고, 같은 셀을 무작위로 나눈 CV MAE는 {m['train_random_cell_cv_MAE']:.3f}다. 후자는 형제 셀이 양쪽에 섞일 수 있어 신규 실험군 성능을 뜻하지 않는다.

보류군 12개를 재표본한 탐색적 군 단위 bootstrap의 MAE 범위(2.5–97.5%)는 **{lo:.2f}–{hi:.2f}**다. 군 수가 작아 정확한 모집단 신뢰구간으로 해석하면 안 된다.

가장 큰 실패는 `G1`: 평균 절대오차 {worst['G1']:.2f}, 해당 군을 제외한 보류 셀의 MAE {mae_without_g1:.2f}. 최고 수명 셀 60.891에 대한 예측은 {p.loc[p.actual.idxmax(), 'predicted']:.2f}였다. 학습 최고 수명은 {m['target_train']['max']:.3f}이며, 고수명 영역은 특히 과소예측 위험이 있다. 아래 결과를 활용할 때 높은 수명값에 대한 범위 외삽으로 간주해야 한다.

보류군 변수 순열 민감도 상위는 `mean_dqdv_dchg_mid_3_0`, `DoD`, `avg_stress`다. 이 지표는 서로 상관된 변수들 사이에서 순위가 흔들리며 **인과적 영향이나 물리적 중요도는 아니다**.

## 다음 데이터 확보 및 적용 조건

1. `Lifetime`의 단위·수명 종료 기준, 3회 변화량 계산 시점, 각 파생 피처의 생성 코드를 확인한다. 예측 시점 이후 정보가 섞이면 이 평가를 다시 해야 한다.
2. 고수명 조건의 독립 실험군을 늘려 `G1` 같은 사례의 과소예측을 점검한다. 같은 군의 반복 셀만 추가하면 신규 조건 일반화 검증에는 도움이 적다.
3. 특정 충·방전 조건을 추천하거나 수명을 보증하는 의사결정에는 독립된 신규 배치 또는 장비 데이터를 더 모아 검증한다.

## 재현

`python battery_lifetime_analysis.py` 다음 `python make_battery_report.py` 실행. 원본 `feature_all.csv`는 실행 파일과 같은 위치의 `upload/` 폴더에 둔다. Python의 pandas, numpy, scipy, scikit-learn, matplotlib, joblib이 필요하다. `model.joblib` 로딩 시 같은 폴더의 `feature_builder.py`가 필요하다. `model_search.csv`에 전체 탐색 결과, `holdout_predictions.csv`에 셀별 예측, `metrics.json`에 평가지표가 있다.
'''
(OUT / 'README.md').write_text(body, encoding='utf-8')
print('report generated',lo,hi)
