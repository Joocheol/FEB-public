# 동반 계산 및 제17장 딥헤징 재현 코드

이 디렉터리는 『금융공학의 이해』의 계산 실습과 제17장 기준 실험을 재현합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python code/ch17_deep_hedging.py
```

빠른 동작 확인은 다음과 같이 합니다.

```bash
python code/ch17_deep_hedging.py --quick
```

기준 환경은 Python 3.13.5, PyTorch 2.10.0 CPU, NumPy 2.3.5입니다. 기본 실행은
난수시드, 거래시점 20개, 훈련·검증·시험 분리를 고정하고 `code/results/`에
새 CSV를 생성합니다. 책에 수록한 고정 결과표는 같은 설정으로 생성한
`ch17_metrics.csv`와 세 정책 비교 CSV입니다. 하드웨어와 저수준 수치연산의
차이로 마지막 소수점은 달라질 수 있으므로, 방향과 허용오차를 함께 확인해야
합니다.

고정 `ch17_learning_curve.csv`는 비용 인식 모형(`model=cost aware`)의 대표
에포크만 보존합니다. 재실행 파일 `ch17_learning_curve_generated.csv`는 비용
없음·비용 인식 두 모형을 25회 간격으로 기록하므로, 자동 대조할 때에는
`model=cost aware`로 거른 뒤 고정 파일의 `epoch`에 맞추어 네 열을 비교합니다.


## 표준 라이브러리 계산 실습

제8장과 제15장의 짧은 검산은 추가 패키지 없이 실행할 수 있습니다.

```bash
python code/ch08_gbm_moments.py
python code/ch08_gbm_moments.py --paths 10000
python code/ch11_barrier_reproduction.py
python code/ch15_mc_convergence.py
```

첫 번째 파일은 기하브라운 운동의 표본평균·표본분산을 이론값과 비교하고,
두 번째 파일은 제11장의 하방소멸 콜 수치예제를 닫힌형 공식으로 재현합니다.
세 번째 파일은 Monte Carlo 표준오차의 로그--로그 기울기가 \(-1/2\)에
가까운지 확인합니다. 난수를 사용하는 제8장·제15장 코드는 시드 2026을 고정합니다.
제8장 코드는 기본값 200,000경로에서 본문의 고정 허용범위를 검사하고, `--paths`로
경로 수를 바꾼 실행에서는 표본값과 이론값을 출력하여 수렴을 비교할 수 있습니다.
나머지 표준 실행도 본문의 기준값이나 허용범위를 벗어나면 실패합니다.
