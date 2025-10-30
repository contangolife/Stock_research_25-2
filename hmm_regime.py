# -*- coding: utf-8 -*-
# hmm_regime.py
# Flexible HMM Regime Analysis Tool (Class Version, same behavior as run_hmm)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, List, Tuple
from matplotlib.lines import Line2D


# ---------------- Utils ----------------
def _to_series_1d(x, name=None, index=None):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            raise ValueError(f"Expected 1D Series, got DataFrame with shape {x.shape}")
    arr = np.asarray(x).ravel()
    return pd.Series(arr, index=index if index is not None else getattr(x, "index", None), name=name)

def ema(s, span):
    return _to_series_1d(s).ewm(span=span, adjust=False).mean()

def macd_signal(close, fast=12, slow=26, signal=9):
    close = _to_series_1d(close)
    macd = ema(close, fast) - ema(close, slow)
    sig  = ema(macd, signal)
    return macd.astype(float), sig.astype(float)

def rsi_wilder(close, period=14):
    close = _to_series_1d(close).astype(float)
    delta = close.diff()
    up    = delta.clip(lower=0.0)
    down  = -delta.clip(upper=0.0)
    roll_up   = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50).astype(float)

def true_range(high, low, close):
    high, low, close = map(_to_series_1d, [high, low, close])
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return pd.Series(tr, index=close.index).astype(float)

def atr(series_tr, window=100):
    return _to_series_1d(series_tr).rolling(window, min_periods=max(1, window // 2)).mean().astype(float)

def rolling_autocorr_mean(ret1, window=60, max_lag=5):
    ret1 = _to_series_1d(ret1).astype(float)
    min_p = max(2, window // 2)
    cols = [ret1.rolling(window, min_periods=min_p).corr(ret1.shift(lag)) for lag in range(1, max_lag + 1)]
    if not cols:
        return pd.Series(np.nan, index=ret1.index)
    return pd.concat(cols, axis=1).mean(axis=1, skipna=True).astype(float)

def safe_lag_autocorr(ret, lag=1, mask=None, eps=1e-12):
    ret, r_lag = _to_series_1d(ret), _to_series_1d(ret).shift(lag)
    if mask is not None:
        mask_aligned = pd.Series(mask, index=ret.index).astype(bool)
        sel = mask_aligned & mask_aligned.shift(lag).fillna(False)
        x, y = ret[sel], r_lag[sel]
    else:
        valid = ret.notna() & r_lag.notna()
        if valid.sum() < 5:
            return 0.0
        x, y = ret[valid], r_lag[valid]
    if x.empty or y.empty:
        return 0.0
    x, y = x.dropna(), y.dropna()
    n = min(len(x), len(y))
    if n < 5 or x.var() < eps or y.var() < eps:
        return 0.0
    try:
        corr = np.corrcoef(x.iloc[:n], y.iloc[:n])[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0
    except:
        return 0.0

def fit_best_hmm(X, n_states=4, n_trials=10, covariance_type='full', random_seed=42, max_iter=1000):
    best_model, best_score = None, -np.inf
    rng = np.random.RandomState(random_seed)
    for _ in range(n_trials):
        seed = rng.randint(0, 10_000)
        try:
            model = GaussianHMM(n_components=n_states, covariance_type=covariance_type,
                                n_iter=max_iter, random_state=seed, tol=1e-3)
            model.fit(X)
            current_score = model.score(X)
            if np.isfinite(current_score) and current_score > best_score:
                best_score, best_model = current_score, model
        except Exception:
            continue
    return best_model, best_score

# ---------------- Labeling (Numeric by Trend) ----------------
def label_states(df, states, n_states):
    info = []
    ret = np.log(df['Close']).diff()
    has_ac1_5 = 'AC1_5' in df.columns

    for k in range(n_states):
        mask = (states == k)
        if np.sum(mask) < 10:
            info.append({'state': k, 'trend_score': -np.inf})
            continue
        info.append({
            'state': k,
            'trend_score': float((df['Close'] - df['MAtrend'])[mask].mean()),
            'mom_score': float(safe_lag_autocorr(ret, lag=1, mask=mask)),  # Diagnostic
            'macd_score': float(df['MACDdiff'][mask].mean()),             # Diagnostic
            'ac1_5_score': float(df['AC1_5'][mask].mean()) if has_ac1_5 else np.nan
        })
    info_df = pd.DataFrame(info).dropna(subset=['trend_score'])
    info_df = info_df.sort_values('trend_score', ascending=False).reset_index(drop=True)

    mapping = {int(row['state']): i + 1 for i, row in info_df.iterrows() if row['trend_score'] > -np.inf}
    all_predicted = np.unique(states)
    for k in all_predicted:
        if k not in mapping:
            mapping[k] = 99  # Unknown/sparse state label

    return mapping, info_df

# ---------------- Plotting (Auto regime count) ----------------
def plot_regimes_banner(
    df, ticker, title_note="Numeric Labels by Trend",
    figsize=(16, 3.6), line_lw=1.6, shade_alpha=0.10, plot_to_file: Optional[str] = None,
    *, ax=None, mode: str = "spans+price", legend_loc="upper left"
):
    if 'regime' not in df.columns:
        raise ValueError("df에 'regime' 컬럼이 없습니다.")

    recent = df.copy()
    recent['regime'] = recent['regime'].astype(int)

    unique_regs = sorted(set(recent['regime']) - {99})
    if 99 in set(recent['regime']):
        unique_regs.append(99)

    base_colors = ['red','orange','gold','green','blue','indigo','violet',
                   'grey','black','pink','brown','cyan','olive','teal','navy',
                   'firebrick','darkorange','darkgoldenrod','forestgreen','royalblue']
    palette = {r: ('lightgray' if r==99 else base_colors[i % len(base_colors)])
               for i, r in enumerate(unique_regs)}

    recent['_block'] = (recent['regime'] != recent['regime'].shift()).cumsum()

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True

    # spans first (bottom layer)
    for _, g in recent.groupby('_block'):
        r = int(g['regime'].iloc[0])
        ax.axvspan(g.index[0], g.index[-1], color=palette.get(r, 'lightgray'),
                   alpha=shade_alpha, linewidth=0, zorder=0)

    if mode == "spans+price":
        plotted = set()
        for (_, _b), g in recent.groupby(['regime', '_block']):
            r = int(g['regime'].iloc[0])
            color = palette.get(r, 'lightgray')
            label = f"Regime {r}" if r != 99 else "Unknown"
            if r not in plotted:
                ax.plot(g.index, g['Close'], color=color, lw=line_lw, label=label, zorder=1)
                plotted.add(r)
            else:
                ax.plot(g.index, g['Close'], color=color, lw=line_lw, zorder=1)

        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)),
                       key=lambda i: (labels[i] == "Unknown",
                                      int(labels[i].split()[-1]) if labels[i] != "Unknown" else 10**9))
        if handles:
            ax.legend([handles[i] for i in order], [labels[i] for i in order],
                      loc=legend_loc, frameon=False, ncol=max(1, len(unique_regs)//2))
    else:
        proxies = [Line2D([0],[0], color=palette[r], lw=line_lw) for r in unique_regs]
        labels  = [f"Regime {r}" if r != 99 else "Unknown" for r in unique_regs]
        if proxies:
            ax.legend(proxies, labels, loc=legend_loc, frameon=False,
                      ncol=max(1, len(unique_regs)//2))

    ax.set_title(f"{ticker} — HMM Regime — {title_note}", pad=10)
    ax.grid(True, alpha=0.25, zorder=2)
    ax.set_xlim(df.index[0], df.index[-1])

    if created:
        plt.tight_layout()
        if plot_to_file:
            plt.savefig(plot_to_file); plt.close()
        else:
            plt.show()

    return ax, {"palette": palette, "unique_regs": unique_regs}

def draw_regime_spans(
    ax,
    idx,
    regime,
    n_states=None,
    *,
    shade_alpha: float = 0.12,
    price_for_legend=None,
    line_lw: float = 1.4,
    legend_loc: str = "upper left"
):
    reg = pd.Series(regime, index=idx).astype(int)
    unique_regs = sorted(set(reg.dropna()) - {99})
    if 99 in set(reg):
        unique_regs.append(99)

    base_colors = ['red','orange','gold','green','blue','indigo','violet',
                   'grey','black','pink','brown','cyan','olive','teal','navy',
                   'firebrick','darkorange','darkgoldenrod','forestgreen','royalblue']
    palette = {r: ('lightgray' if r==99 else base_colors[i % len(base_colors)])
               for i, r in enumerate(unique_regs)}

    block = (reg != reg.shift()).cumsum()
    for _, g in pd.DataFrame({"reg": reg, "_block": block}).groupby("_block"):
        r = int(g["reg"].iloc[0])
        span_idx = g.index
        ax.axvspan(span_idx[0], span_idx[-1],
                   color=palette.get(r, "lightgray"),
                   alpha=shade_alpha, linewidth=0, zorder=0)

    proxies = [Line2D([0],[0], color=palette[r], lw=line_lw) for r in unique_regs]
    labels  = [f"Regime {r}" if r != 99 else "Unknown" for r in unique_regs]
    if proxies:
        ax.legend(proxies, labels, loc=legend_loc, frameon=False,
                  ncol=max(1, len(unique_regs)//2))


# --- 안전한 멀티인덱스 변환 ---
def with_regime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'regime' not in out.columns:
        raise ValueError("df에 'regime' 컬럼이 없습니다. run_hmm/HMMRegimeAnalyzer.run(...) 이후의 df를 넣어주세요.")
    out['regime_block'] = (out['regime'] != out['regime'].shift()).cumsum().astype(int)
    out = out.set_index(['regime', 'regime_block'], append=True)
    out = out.reorder_levels([1, 2, 0]).sort_index()
    names = list(out.index.names)
    names[0] = 'regime'
    names[1] = 'regime_block'
    names[2] = names[2] or 'date'
    out.index = out.index.set_names(names)
    return out


# ---------------- Class (same results as run_hmm) ----------------
class HMMRegimeAnalyzer:
    """
    기존 run_hmm(...)와 동일한 입력/전처리/학습/라벨/플로팅을 수행하여
    df, feats_scaled, model, mapping 을 반환합니다.
    """

    def __init__(
        self,
        *,
        ticker: str = "^KS11",
        start: str = "1990-01-01",
        plot_start: str = "2020-01-01",
        n_states: int = 4,
        n_trials: int = 12,
        ma_window: int = 200,
        rsi_period: int = 14,
        ac_window: int = 60,
        atr_clip_window: int = 252,
        smoothing_window_mr: int = 3,
        use_slope: bool = True,
        use_ac: bool = True,
        use_rsi: bool = True,
        use_volume: bool = True
    ):
        self.ticker = ticker
        self.start = start
        self.plot_start = plot_start
        self.n_states = n_states
        self.n_trials = n_trials
        self.ma_window = ma_window
        self.rsi_period = rsi_period
        self.ac_window = ac_window
        self.atr_clip_window = atr_clip_window
        self.smoothing_window_mr = smoothing_window_mr
        self.use_slope = use_slope
        self.use_ac = use_ac
        self.use_rsi = use_rsi
        self.use_volume = use_volume

        # 내부 산출물
        self.model: Optional[GaussianHMM] = None
        self.mapping: Optional[Dict[int, int]] = None
        self.summary_: Optional[pd.DataFrame] = None
        self.features_used_: Optional[List[str]] = None
        self.best_score_: Optional[float] = None

    # ---- 핵심 실행 (run_hmm과 동등) ----
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, GaussianHMM, Dict[int, int]]:
        # 1) 데이터 다운로드 & 기본 컬럼 처리 (원본과 동일)
        try:
            raw = yf.download(self.ticker, start=self.start, auto_adjust=True, interval="1d")
            if raw.empty:
                raise RuntimeError("Data download failed.")
        except Exception as e:
            print(f"Data download failed: {e}")
            return None, None, None, None

        processed_data = {}
        cols_needed = ['Close', 'High', 'Low']
        use_volume = self.use_volume
        if use_volume:
            cols_needed.append('Volume')

        for col in cols_needed:
            if col not in raw.columns:
                if col == 'Volume':
                    use_volume = False
                    continue
                else:
                    print(f"Error: Required column '{col}' not found.")
                    return None, None, None, None
            try:
                series = _to_series_1d(raw[col], name=col)
                series = pd.to_numeric(series, errors='coerce')
                if series.isnull().all():
                    raise ValueError("All values NaN after numeric conversion.")
                processed_data[col.lower()] = series
            except Exception as e:
                print(f"Error processing column '{col}': {e}")
                if col == 'Volume':
                    use_volume = False
                    continue
                else:
                    return None, None, None, None

        close = processed_data['close']
        high  = processed_data['high']
        low   = processed_data['low']
        volume = processed_data.get('volume', None)

        df = pd.DataFrame({"Close": close}).dropna(subset=['Close'])
        high, low = high.reindex(df.index), low.reindex(df.index)
        if volume is not None:
            volume = volume.reindex(df.index)

        # 2) 지표 계산 (원본과 동일, ATR robust clip 포함)
        df['MAtrend'] = df['Close'].rolling(self.ma_window, min_periods=max(1, self.ma_window//3)).mean()
        tr = true_range(high, low, df['Close'])
        atr100 = atr(tr, window=100)

        atr100_robust = atr100.copy()
        if len(atr100.dropna()) >= 60:
            p10 = atr100.rolling(self.atr_clip_window, min_periods=60).quantile(0.10).fillna(method='bfill').fillna(0.01)
            p90 = atr100.rolling(self.atr_clip_window, min_periods=60).quantile(0.90).fillna(method='bfill').fillna(1.0)
            if not (p10.isnull().all() or p90.isnull().all()):
                atr100_robust = atr100.clip(lower=p10, upper=p90)
        atr100_robust = atr100_robust.replace(0, np.nan).fillna(method='ffill').fillna(method='bfill').fillna(1e-6).reindex(df.index)

        macd, sig = macd_signal(df['Close'])
        df['MACDdiff'] = (macd - sig).astype(float)
        ret1 = np.log(df['Close']).diff()
        ac = rolling_autocorr_mean(ret1, window=self.ac_window, max_lag=5) if self.use_ac else None
        rsi = rsi_wilder(df['Close'], period=self.rsi_period) if self.use_rsi else None

        # 3) 피처 구성 (원본과 동일)
        feats_dict = {'Level_ATR100': (df['Close'] - df['MAtrend']) / atr100_robust}
        if self.use_slope:
            feats_dict['Slope_ATR100'] = df['MACDdiff'] / atr100_robust
        if self.use_ac and ac is not None:
            feats_dict['AC1_5'] = ac
        if self.use_rsi and rsi is not None:
            feats_dict['RSI'] = rsi
        if use_volume and volume is not None:
            vol_atr = (volume / atr100_robust).replace([np.inf, -np.inf], np.nan)
            feats_dict['Volume_ATR'] = vol_atr

        feats = pd.DataFrame(feats_dict)

        # 4) Smoothing (원본과 동일)
        if self.smoothing_window_mr > 1:
            for col in ['RSI', 'AC1_5', 'Volume_ATR']:
                if col in feats:
                    feats[col] = feats[col].rolling(self.smoothing_window_mr, min_periods=1).mean()

        feats = feats.dropna()
        if feats.empty:
            print("Error: No valid features after NaN removal.")
            return None, None, None, None

        # 5) 정렬/스케일/적합 (원본과 동일)
        common_index = df.index.intersection(feats.index)
        df = df.loc[common_index].copy()
        feats = feats.loc[common_index].copy()
        if 'AC1_5' in feats:
            df['AC1_5'] = feats['AC1_5']  # diagnostics
        X = StandardScaler().fit_transform(feats.values)
        if X.shape[0] < self.n_states:
            print(f"Error: Not enough data ({X.shape[0]}) for HMM.")
            return None, None, None, None

        model, best_score = fit_best_hmm(X, n_states=self.n_states, n_trials=self.n_trials)
        if model is None:
            print("HMM fitting failed.")
            return None, None, None, None

        try:
            states = model.predict(X)
        except Exception as e:
            print(f"HMM prediction failed: {e}")
            return None, None, None, None

        # 6) 라벨링/플롯/요약 (원본과 동일)
        mapping, summary = label_states(df, states, n_states=self.n_states)
        df['state']  = states
        df['regime'] = [mapping.get(s, 99) for s in states]

        recent = df.loc[df.index >= pd.to_datetime(self.plot_start)].copy()
        if not recent.empty:
            plot_regimes_banner(recent, ticker=self.ticker,
                                title_note=f"Numeric Labels by Trend, K={self.n_states}")

        print(f"\n--- HMM Results (K={self.n_states}) ---")
        print(f"Features Used: {list(feats.columns)}")
        print("Best log-likelihood:", best_score)
        print_cols = ['state', 'trend_score', 'mom_score', 'macd_score']
        if 'ac1_5_score' in summary.columns:
            print_cols.append('ac1_5_score')
        if not summary.empty:
            print(summary.sort_values('trend_score', ascending=False)[print_cols])
        else:
            print("Summary table could not be generated.")

        feats_scaled = pd.DataFrame(X, index=df.index, columns=feats.columns)

        # 내부 저장 (선택)
        self.model = model
        self.mapping = mapping
        self.summary_ = summary
        self.features_used_ = list(feats.columns)
        self.best_score_ = best_score

        return df, feats_scaled, model, mapping


# --------- Script-style 실행 (옵션) ---------
if __name__ == "__main__":
    # 기존 실행 코드와 완전히 동일한 파라미터
    N_STATES_TO_RUN = 4
    TICKER_TO_RUN = "^GSPC"
    START_DATE = "1995-01-01"
    PLOT_START = "2020-01-01"
    MA_WINDOW = 120
    AC_WINDOW = 60
    USE_SLOPE = False
    USE_AC = False
    USE_RSI = True
    USE_VOLUME = False

    analyzer = HMMRegimeAnalyzer(
        ticker=TICKER_TO_RUN,
        start=START_DATE,
        plot_start=PLOT_START,
        n_states=N_STATES_TO_RUN,
        n_trials=20,
        ma_window=MA_WINDOW,
        rsi_period=14,
        ac_window=AC_WINDOW,
        smoothing_window_mr=1,
        use_slope=USE_SLOPE,
        use_ac=USE_AC,
        use_rsi=USE_RSI,
        use_volume=USE_VOLUME
    )

    df, feats_scaled, model, mapping = analyzer.run()
    if df is None:
        raise SystemExit("HMM 실행 실패: df가 None입니다.")

    # 레짐 인덱싱 버전 생성
    df_ix = with_regime_index(df)
