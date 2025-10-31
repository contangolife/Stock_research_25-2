============================================================
Stock_research_25-2  —  Regime-switch Backtesting Framework
============================================================

개요
----
레짐(시장 상태)에 따라 모멘텀/평균회귀 전략을 스위칭하고,
트레일링 스탑(ATR/퍼센트), 초기 하드스톱, 현실적 체결 모드를 반영한
데일리 백테스트 파이프라인.

핵핵심 특징
---------
- Regime 기반 전략 스위칭 (regime 값에 따라 전략 선택)
- 전략 세트: strat_momentum, strat_mean_reversion, strat_random_walk
- 리스크 관리: ATR/퍼센트 트레일링 스탑 + 초기 하드스톱 + 무장(arm) 조건
- 현실적 체결 모드: execute_price_mode = close | stop | next_open
- 비용 모델: bps 기반 수수료/슬리피지, 리버설 시 턴오버 2x
- 전략/레짐 기여도(Attribution): 레짐 구간별 net 수익을 전략명 기준 집계
- 시각화: 체결/리버설/스탑 포인트, 레짐 셰이딩, 누적수익 곡선

────────────────────────────────────────────────────────────
1) Quick Start
────────────────────────────────────────────────────────────
```python
from regime_backtester import RegimeBacktester, BTConfig

cfg = BTConfig(
    init_capital=50_000_000,
    fee_bps=5,
    slip_bps=0,
    allow_shorts=True
)

rb = RegimeBacktester(config=cfg)

strategies = {
    1: RegimeBacktester.strat_momentum,
    2: RegimeBacktester.strat_momentum,
    3: RegimeBacktester.strat_mean_reversion,
    4: RegimeBacktester.strat_mean_reversion,
}

bt = rb.run_backtest(
    df_out,
    strategies=strategies,
    price_col="Close",
    regime_col="regime",
    start="2015-01-01",
    end="2025-10-20",
    warmup_bars=200,
    exec_shift=1,                  # 일반 신호는 다음 바 체결(룩어헤드 방지)
    k_atr=2,                       # ATR 트레일링 스탑
    trail_pct=None,                # ATR이 우선
    execute_on_trigger_close=True, # 스탑 트리거 당일 처리
    execute_price_mode="stop",     # 스탑 레벨로 체결
    min_gain_to_trail_atr=0.2,     # 무장 조건 1: 이익이 0.2×ATR 이상
    arm_after_bars=21,             # 무장 조건 2: 진입 후 21바 경과
    initial_k_atr=2.0,             # 초기 하드스톱: 진입가 ± 2×ATR
    hard_stop_use_entry_atr=True   # 초기 ATR 고정
)

rb.plot_backtest_with_trades_rainbow(bt, df_out, ticker="SPY")
rb.plot_cum_return_after(bt, title="Cumulative Return (net) vs Buy&Hold")
```

────────────────────────────────────────────────────────────
2) Data Requirements
────────────────────────────────────────────────────────────
- df_out: pandas DataFrame, DatetimeIndex
- 필수 컬럼: Close
- 권장 컬럼: High, Low (스탑 트리거 정확도 향상을 위해 사실상 필수)
- regime 컬럼: 정수 {1,2,3,4,...} 또는 문자열 {Momentum, MeanReversion, RandomWalk}
- 참고: High/Low가 없으면 내부적으로 High=Low=Close로 간주하여
        종가가 스탑 레벨을 통과할 때만 트리거됨

────────────────────────────────────────────────────────────
3) Core API 개요
────────────────────────────────────────────────────────────
```python
# BTConfig (핵심 일부)
from dataclasses import dataclass

@dataclass
class BTConfig:
    init_capital: float = 1_000_000.0
    fee_bps: float = 5.0
    slip_bps: float = 0.0
    max_leverage: float = 1.0
    allow_shorts: bool = True
    # 전략 파라미터 예
    mom_lookback: int = 20
    mr_z_window: int = 60
    mr_smooth: int = 10

# run_backtest 주요 인자(발췌)
rb.run_backtest(
    df_out,
    strategies: Dict[int, Callable],
    price_col="Close",
    regime_col="regime",
    start=None,
    end=None,
    warmup_bars=100,
    exec_shift=0,
    k_atr=None,                   # ATR 스탑 강도(k)
    trail_pct=0.05,               # 퍼센트 스탑(ATR보다 우선순위 낮음)
    execute_on_trigger_close=False,
    execute_price_mode="close",   # "close" | "stop" | "next_open"
    min_gain_to_trail_atr=0.0,
    arm_after_bars=0,
    initial_k_atr=None,
    hard_stop_use_entry_atr=True,
    ...
)
```

체결/스탑 로직 요약
- 트리거 판정
  · 롱 보유: Low <= stop_level
  · 숏 보유: High >= stop_level
- 체결가 모드(execute_price_mode)
  · "close"     : 그날 종가로 청산
  · "stop"      : 계산된 스탑 레벨로 청산
  · "next_open" : 다음 날 시가로 청산
- 트리거 시점 처리
  · execute_on_trigger_close=True → 트리거 당일 PnL/체결 반영
  · 일반 진입/청산 신호는 exec_shift 만큼 지연 체결

PnL 반영(스탑 체결 바)
- gross[t] = prev_pos * ((exec_px / prev_close) - 1)
- net[t]   = gross[t] - cost[t]
- cost = |Δpos| × (fee_bps + slip_bps) / 10000
- 리버설(+1 → -1)은 |Δpos|=2로 비용 2배

────────────────────────────────────────────────────────────
4) Strategies
────────────────────────────────────────────────────────────
- strat_momentum
  · 듀얼 룩백 누적수익 + 변동성 정규화(z-score)
  · 히스테리시스 밴드, 장기 SMA 동의(trend filter), EWM smoothing
- strat_mean_reversion
  · RSI 기반 진입/청산 밴드(+ pad)
  · 장기 평균 대비 z-편차 과도 시 거래 금지(트렌드 블록)
- strat_random_walk
  · 포지션 0 (테스트/비교용)

────────────────────────────────────────────────────────────
5) Trailing Stop 상세
────────────────────────────────────────────────────────────
- ATR 스탑
  · 롱: stop = peak - k*ATR (단조 증가)
  · 숏: stop = trough + k*ATR (단조 감소)
- 퍼센트 스탑
  · 롱: stop = peak * (1 - pct)
  · 숏: stop = trough * (1 + pct)
- 무장(arm) 논리
  · min_gain_to_trail_atr: 진입가 대비 이익이 X×ATR 이상이면 무장
  · arm_after_bars: 진입 후 N바 경과하면 무장
  · 둘 중 하나라도 충족하면 활성화
- 초기 하드스톱
  · initial_k_atr 지정 시, 진입가 ± (k × ATR_at_entry)
  · hard_stop_use_entry_atr=True면 진입 시 ATR 고정

────────────────────────────────────────────────────────────
6) Strategy Attribution(기여도)
────────────────────────────────────────────────────────────
```python
from regime_backtester import strategy_attribution

# 예: regime1_momentum, regime3_mean_reversion 등 라벨 자동 생성
name_map = RegimeBacktester.make_regime_name_map(strategies)

summary, parts = rb.strategy_attribution(
    bt,
    df_out,
    regime_col="regime",
    strategies=strategies,
    name_map=name_map
)

print(summary)   # TotalRet, CAGR, Sharpe(ann.), MaxDD, DaysActive, Contribution(%)
# parts: 일별 전략별 net 기여 시리즈 (같은 전략명은 자동 합산)
```
────────────────────────────────────────────────────────────
7) 시각화
────────────────────────────────────────────────────────────
```python
rb.plot_backtest_with_trades_rainbow(bt, df_out, ticker="SPY")
# 가격, 레짐 셰이딩, 엔트리/청산/리버설, 스탑 포인트 표시

rb.plot_cum_return_after(bt, title="Cumulative Return (net) vs Buy&Hold")
# 순/총(그로스) 누적수익, 바앤홀 비교, 초기/최종 자본 출력
```

────────────────────────────────────────────────────────────
8) 권장 비용 세팅(참고)
────────────────────────────────────────────────────────────
- 미국 대형 ETF/선물: fee_bps 0~1, slip_bps 1~2
- KOSPI 200 대형주/ETF: fee_bps 1~2, slip_bps 2~4
- 리버설은 턴오버=2 → 비용 2배
- (선택) 스탑/리버설/갭 추가 bps 같은 현실적 비용 모델 확장 가능

────────────────────────────────────────────────────────────
9) 프로젝트 구조(예시)
────────────────────────────────────────────────────────────
```
Stock_research_25-2/
├─ regime_backtester.py     # 핵심 클래스/전략/스탑/시각화/어트리뷰션
├─ notebooks/
│  └─ demo.ipynb            # 사용 예제
├─ data/
│  └─ sample.csv            # 예시 데이터(선택)
├─ README.md
└─ requirements.txt
```

requirements.txt 예시
```
pandas
numpy
matplotlib
```

────────────────────────────────────────────────────────────
10) 재현(Repro) 절차(예시)
────────────────────────────────────────────────────────────
1. df_out 준비 (필수: Close, 권장: High/Low, 레짐: regime)
2. strategies 매핑 정의 (regime id → 전략 함수)
3. run_backtest 실행
4. plot_* 함수로 시각화
5. strategy_attribution 실행으로 기여도 분석

────────────────────────────────────────────────────────────
11) Open Questions(채워 주세요)
────────────────────────────────────────────────────────────
- 레짐 산출 방식: regime 컬럼 생성 방법(모델/룰/레이블링)
- 데이터 소스/주기: 종목/지수/ETF, 시간대/영업일, 결측치 처리
- 대표 성능 지표: 백테스트 결과(CAGR/Sharpe/MaxDD) 스냅샷(표/숫자)
- 라이선스: 저장소/코드 라이선스(예: MIT)
- 기여 규칙: PR/이슈 템플릿
- 추가 비용 모델: 스탑/리버설/갭 추가 bps 적용 여부와 기본값
