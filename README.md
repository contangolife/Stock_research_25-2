Stock_research_25-2

Regime-switch Backtesting Framework
레짐(시장 상태)에 따라 모멘텀 / 평균회귀 전략을 스위칭하고, 트레일링 스탑(ATR/퍼센트), 초기 하드스탑, 현실적 체결 모드를 반영한 데일리 백테스트 파이프라인.

Key Features

🔀 Regime 기반 스위칭: regime ∈ {…}마다 다른 전략 매핑

📈 전략 세트: strat_momentum, strat_mean_reversion, strat_random_walk

🛡️ 리스크 관리: ATR/퍼센트 트레일링 스탑 + 초기 하드스탑 + 무장(arm) 조건

⚙️ 현실적 체결 모드: execute_price_mode ∈ {"close","stop","next_open"}

💸 비용 모델: bps 기반 수수료/슬리피지 + 리버설 시 턴오버 2× 반영

📊 전략/레짐 기여도(Attribution): 레짐 구간별 net 수익을 전략명 기준으로 집계

🖼️ 시각화: 체결/리버설/스탑 포인트 및 레짐 셰이딩, 누적수익 그래프

1) Quick Start
from regime_backtester import RegimeBacktester, BTConfig

cfg = BTConfig(
    init_capital=50_000_000,
    fee_bps=5, slip_bps=0, allow_shorts=True
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
    start="2015-01-01", end="2025-10-20",
    warmup_bars=200,
    exec_shift=1,              # 일반 신호는 다음 바 체결(룩어헤드 방지)
    k_atr=2,                   # ATR 트레일링 스탑
    trail_pct=None,            # (ATR가 우선)
    execute_on_trigger_close=True,  # 스탑 트리거 당일 반영
    execute_price_mode="stop",      # 스탑 레벨로 체결
    min_gain_to_trail_atr=0.2,      # 무장 조건 1: 이익이 0.2×ATR 이상
    arm_after_bars=21,              # 무장 조건 2: 진입 후 21바 경과
    initial_k_atr=2.0,              # 초기 하드스톱: 진입가 ± 2×ATR
    hard_stop_use_entry_atr=True,   # 초기 ATR 고정
)

rb.plot_backtest_with_trades_rainbow(bt, df_out, ticker="SPY")
rb.plot_cum_return_after(bt, title="Cumulative Return (net) vs Buy&Hold")

2) Data Requirements

df_out (pandas DataFrame, DatetimeIndex)

필수: Close

권장: High, Low (스탑 트리거 정확도를 위해 필수에 가깝습니다)

레짐: regime (정수 {1,2,3,4,...} 또는 문자열 {"Momentum","MeanReversion","RandomWalk"})

High/Low가 없으면 내부적으로 High=Low=Close로 간주되어, 종가가 스탑 레벨을 넘을 때만 트리거됩니다.

3) Core API
3.1 Config
@dataclass
class BTConfig:
    init_capital: float = 1_000_000.0
    fee_bps: float = 5.0
    slip_bps: float = 0.0
    max_leverage: float = 1.0
    allow_shorts: bool = True
    # 전략 파라미터(예: 모멘텀/평균회귀용)
    mom_lookback: int = 20
    mr_z_window: int = 60
    mr_smooth: int = 10

3.2 Run Backtest
rb.run_backtest(
    df_out,
    *,
    strategies: Dict[int, Callable],
    price_col: str = "Close",
    regime_col: str = "regime",
    start=None, end=None,
    warmup_bars=100,
    exec_shift=0,
    k_atr: float = None,
    trail_pct: float = 0.05,
    execute_on_trigger_close=False,
    execute_price_mode: str = "close",   # "close" | "stop" | "next_open"
    min_gain_to_trail_atr: float = 0.0,
    arm_after_bars: int = 0,
    initial_k_atr: float = None,
    hard_stop_use_entry_atr: bool = True,
    ...
) -> Dict[str, pd.DataFrame]


체결/스탑 로직 핵심

트리거 판정:

롱 보유: Low <= stop_level → 트리거

숏 보유: High >= stop_level → 트리거

체결가(execute_price_mode)

"close": 그날 종가로 청산

"stop": 계산된 스탑 레벨로 청산

"next_open": 다음 날 시가로 청산

트리거 시점 처리

execute_on_trigger_close=True → 트리거 당일 PnL/체결 로그 반영

일반 신호는 exec_shift만큼 지연 체결(룩어헤드 방지)

PnL 반영

스탑 체결 바에서
gross[t] = prev_pos × ((exec_px / prev_close) - 1)
net[t] = gross[t] - cost[t]

cost = |Δpos| × (fee_bps + slip_bps)/1e4

리버설(+1→−1)은 |Δpos|=2로 비용이 2×.

4) Strategies
4.1 Momentum (strat_momentum)

듀얼 룩백 누적수익 + 변동성 정규화(z-score) + 히스테리시스 밴드

장기 SMA 동의(trend filter)

EWM smoothing 옵션

cfg.mom_* 파라미터로 조정

4.2 Mean Reversion (strat_mean_reversion)

RSI(진입/청산 밴드 + pad) 기반

장기평균 대비 z-편차가 큰 트렌드 블록 시 거래 금지

EWM smoothing 옵션

cfg.mr_* 파라미터로 조정

4.3 Random Walk (strat_random_walk)

포지션 0(테스트용)

5) Trailing Stop
5.1 ATR Stop (권장)

롱: stop = peak - k*ATR (단조 증가)

숏: stop = trough + k*ATR (단조 감소)

5.2 Percent Stop

롱: stop = peak × (1 - pct)

숏: stop = trough × (1 + pct)

5.3 “무장(arm)” 논리

min_gain_to_trail_atr: 진입가 대비 이익이 X×ATR 이상이면 무장

arm_after_bars: 진입 후 N바 경과하면 무장
→ 둘 중 하나라도 충족하면 활성화

5.4 초기 하드스탑

initial_k_atr 지정 시, 진입가 ±(k×ATR_at_entry)로 초기 손절선 부여

hard_stop_use_entry_atr=True면 진입 시 ATR 고정

6) Strategy Attribution

전략/레짐별 net 수익 기여를 집계합니다.

from regime_backtester import strategy_attribution

name_map = RegimeBacktester.make_regime_name_map(strategies)  # regime1_momentum 형식
summary, parts = rb.strategy_attribution(
    bt, df_out,
    regime_col="regime",
    strategies=strategies,
    name_map=name_map
)

print(summary)   # 전략별 TotalRet, CAGR, Sharpe, MaxDD, DaysActive, Contribution(%)
# parts: 일별 전략별 net 기여 시리즈


같은 이름(예: 레짐1·2 모두 Momentum)은 자동 병합.

Unattributed 컬럼은 이론상 0(진단용).

7) Visualization
rb.plot_backtest_with_trades_rainbow(bt, df_out, ticker="SPY")
# 가격 + 레짐 셰이딩 + 엔트리/청산/리버설 + 스탑 포인트

rb.plot_cum_return_after(bt, title="Cumulative Return (net) vs Buy&Hold")
# 순/총(그로스) 누적수익, 바앤홀 비교, 초기/최종 자본 표기

8) Recommended Cost Settings

미국 대형 ETF/선물: fee_bps=0~1, slip_bps=1~2

KOSPI 200 대형주/ETF: fee_bps=1~2, slip_bps=2~4

리버설은 턴오버=2 → 비용 2×

(선택) 스탑/리버설/갭 추가 bps 옵션을 추후 확장 가능

9) Project Structure (예시)
Stock_research_25-2/
├─ regime_backtester.py     # 핵심 클래스/전략/스탑/시각화/어트리뷰션
├─ notebooks/
│  └─ demo.ipynb            # 사용 예제 노트북
├─ data/
│  └─ sample.csv            # 예시 데이터(옵션)
├─ README.md
└─ requirements.txt


requirements.txt

pandas
numpy
matplotlib

10) Repro Steps (예시)

df_out 준비 (Close, 권장: High, Low, regime)

strategies 매핑 정의

rb.run_backtest(...) 실행

plot_*로 시각화

필요 시 strategy_attribution(...)으로 기여도 분석

11) Open Questions (보충 필요)

 레짐 산출 방식: regime 컬럼을 어떻게 생성했는지(모델/룰/레이블링) 간략 설명

 데이터 소스/주기: 종목/지수/ETF, TZ/영업일, 결측치 처리 규칙

 대표 성능 지표: 백테스트 결과(CAGR/Sharpe/MaxDD 등) 스냅샷(표/이미지)

 라이선스: 저장소/코드 라이선스 (예: MIT)

 기여 규칙: PR/이슈 템플릿 여부

 추가 비용 모델: 스탑/리버설/갭 추가 bps 적용 여부/기본값

필요한 항목 알려주시면 해당 섹션에 바로 채워서 완성 README로 드릴게요.
