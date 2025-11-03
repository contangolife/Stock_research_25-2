# -*- coding: utf-8 -*-
# hmm_regime.py
# Flexible HMM Regime Analysis Tool (Class Version, upgraded with run_hmm features)

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


# ---------------- Labeling (영어 레짐 라벨, Level_ATR100 기반) ----------------
def label_states(
    df: pd.DataFrame,
    states: np.ndarray,
    n_states: int,
    model: Optional[GaussianHMM] = None,
    features_df: Optional[pd.DataFrame] = None,
    calc_diagnostics: bool = True
):
    """
    반환: (mapping_int, info_df)
      - mapping_int: {원래상태index -> 1..n, 미할당 99}
      - info_df: trend/mom/MACD/AC1_5 진단표
    정렬 기준:
      1순위) model.means_의 'Level_ATR100' 평균(내림차순)
      2순위) fallback: trend_score(내림차순)
    """
    info_df = pd.DataFrame()

    # --- 진단표 (option) ---
    if calc_diagnostics:
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
                'mom_score': float(safe_lag_autocorr(ret, lag=1, mask=mask)),
                'macd_score': float(df['MACDdiff'][mask].mean()),
                'ac1_5_score': float(df['AC1_5'][mask].mean()) if has_ac1_5 else np.nan
            })
        info_df = pd.DataFrame(info).dropna(subset=['trend_score'])

    # --- model.means_로 정렬 시도 ---
    sorted_state_indices = None
    if (model is not None) and (features_df is not None):
        level_feat_name = 'Level_ATR100' if 'Level_ATR100' in features_df.columns else features_df.columns[0]
        try:
            level_feat_idx = list(features_df.columns).index(level_feat_name)
            level_means_per_state = model.means_[:, level_feat_idx]
            sorted_state_indices = np.argsort(level_means_per_state)[::-1]  # 내림차순
        except Exception as e:
            print(f"Warning: mean-based sorting failed ({e}); fallback to trend_score.")

    if sorted_state_indices is None:
        # fallback: trend_score
        if not info_df.empty:
            sorted_state_indices = info_df.sort_values('trend_score', ascending=False)['state'].to_list()
        else:
            sorted_state_indices = list(range(n_states))

    # --- 정수 레짐 매핑(1..n), 미할당은 99 ---
    mapping_int = {}
    for i, st in enumerate(sorted_state_indices):
        mapping_int[int(st)] = int(i + 1)
    all_pred = np.unique(states)
    for k in all_pred:
        if int(k) not in mapping_int:
            mapping_int[int(k)] = 99

    return mapping_int, info_df

    # --- fallback: 예전 numeric 라벨 방식 (필요시) ---
    if info_df.empty:
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
                'mom_score': float(safe_lag_autocorr(ret, lag=1, mask=mask)),
                'macd_score': float(df['MACDdiff'][mask].mean()),
                'ac1_5_score': float(df['AC1_5'][mask].mean()) if has_ac1_5 else np.nan
            })
        info_df = pd.DataFrame(info).dropna(subset=['trend_score'])

    info_df = info_df.sort_values('trend_score', ascending=False).reset_index(drop=True)
    mapping = {int(row['state']): i + 1 for i, row in info_df.iterrows() if row['trend_score'] > -np.inf}
    all_predicted = np.unique(states)
    for k in all_predicted:
        if k not in mapping:
            mapping[k] = 99
    return mapping, info_df


# ---------------- Plotting (문자 레짐 라벨 대응) ----------------
def plot_regimes_banner(
    df, ticker, title_note="Labels by Trend",
    figsize=(16, 3.6), line_lw=1.6, shade_alpha=0.10, plot_to_file: Optional[str] = None,
    *, ax=None, mode: str = "spans+price", legend_loc="upper left"
):
    if 'regime' not in df.columns:
        raise ValueError("df에 'regime' 컬럼이 없습니다.")

    recent = df.copy()
    # regime을 문자열로 통일 (Bull, Bear, 1, 2, 99 등 모두 처리)
    recent['regime'] = recent['regime'].astype(str)

    regs = set(recent['regime'].dropna())
    unique_regs = sorted(r for r in regs if r != "Unknown")
    if "Unknown" in regs:
        unique_regs.append("Unknown")

    base_colors = ['red','orange','gold','green','blue','indigo','violet',
                   'grey','black','pink','brown','cyan','olive','teal','navy',
                   'firebrick','darkorange','darkgoldenrod','forestgreen','royalblue']
    palette = {
        r: ('lightgray' if r == "Unknown" else base_colors[i % len(base_colors)])
        for i, r in enumerate(unique_regs)
    }

    recent['_block'] = (recent['regime'] != recent['regime'].shift()).cumsum()

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True

    # spans (배경 레짐 영역)
    for _, g in recent.groupby('_block'):
        r = g['regime'].iloc[0]
        ax.axvspan(g.index[0], g.index[-1], color=palette.get(r, 'lightgray'),
                   alpha=shade_alpha, linewidth=0, zorder=0)

    if mode == "spans+price":
        plotted = set()
        for (_, _b), g in recent.groupby(['regime', '_block']):
            r = g['regime'].iloc[0]
            color = palette.get(r, 'lightgray')
            label = r
            if r not in plotted:
                ax.plot(g.index, g['Close'], color=color, lw=line_lw, label=label, zorder=1)
                plotted.add(r)
            else:
                ax.plot(g.index, g['Close'], color=color, lw=line_lw, zorder=1)

        handles, labels = ax.get_legend_handles_labels()

        # Bull -> Weak Bull -> Weak Bear -> Bear -> 기타 -> Unknown 순 정렬
        pref_order = ["Bull", "Weak Bull", "Weak Bear", "Bear"]
        order = sorted(
            range(len(labels)),
            key=lambda i: (
                labels[i] == "Unknown",
                (pref_order.index(labels[i]) if labels[i] in pref_order else 99)
            )
        )
        if handles:
            ax.legend([handles[i] for i in order], [labels[i] for i in order],
                      loc=legend_loc, frameon=False, ncol=max(1, len(unique_regs)//2))
    else:
        proxies = [Line2D([0],[0], color=palette[r], lw=line_lw) for r in unique_regs]
        labels  = [r for r in unique_regs]
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
    reg = pd.Series(regime, index=idx).astype(str)
    regs = set(reg.dropna())
    unique_regs = sorted(r for r in regs if r != "Unknown")
    if "Unknown" in regs:
        unique_regs.append("Unknown")

    base_colors = ['red','orange','gold','green','blue','indigo','violet',
                   'grey','black','pink','brown','cyan','olive','teal','navy',
                   'firebrick','darkorange','darkgoldenrod','forestgreen','royalblue']
    palette = {
        r: ('lightgray' if r == "Unknown" else base_colors[i % len(base_colors)])
        for i, r in enumerate(unique_regs)
    }

    block = (reg != reg.shift()).cumsum()
    for _, g in pd.DataFrame({"reg": reg, "_block": block}).groupby("_block"):
        r = g["reg"].iloc[0]
        span_idx = g.index
        ax.axvspan(span_idx[0], span_idx[-1],
                   color=palette.get(r, "lightgray"),
                   alpha=shade_alpha, linewidth=0, zorder=0)

    proxies = [Line2D([0],[0], color=palette[r], lw=line_lw) for r in unique_regs]
    labels  = [r for r in unique_regs]
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


# ---------------- Class ----------------
class HMMRegimeAnalyzer:
    """
    df, feats_scaled, model, mapping 을 반환.
    - 기존 run()의 인풋/아웃풋 그대로 유지
    - 레짐 라벨: 'Bull', 'Weak Bull', 'Weak Bear', 'Bear', 'Other5', 'Other6', 'Unknown'
    - 마지막에 unscaled feature means 및 다음날 상태 확률까지 출력
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
        self.mapping: Optional[Dict[int, str]] = None
        self.summary_: Optional[pd.DataFrame] = None
        self.features_used_: Optional[List[str]] = None
        self.best_score_: Optional[float] = None
        self.scaler_: Optional[StandardScaler] = None

    # ---- 핵심 실행 ----
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, GaussianHMM, Dict[int, int]]:
        # 1) 데이터 다운로드
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
        if use_volume: cols_needed.append('Volume')
    
        for col in cols_needed:
            if col not in raw.columns:
                if col == 'Volume':
                    use_volume = False; continue
                else:
                    print(f"Error: Required column '{col}' not found."); return None, None, None, None
            try:
                series = _to_series_1d(raw[col], name=col)
                series = pd.to_numeric(series, errors='coerce')
                if series.isnull().all(): raise ValueError("All values NaN after numeric conversion.")
                processed_data[col.lower()] = series
            except Exception as e:
                print(f"Error processing column '{col}': {e}")
                if col == 'Volume':
                    use_volume = False; continue
                else:
                    return None, None, None, None
    
        close = processed_data['close']; high = processed_data['high']; low = processed_data['low']
        volume = processed_data.get('volume', None)
    
        df = pd.DataFrame({"Close": close}).dropna(subset=['Close'])
        high, low = high.reindex(df.index), low.reindex(df.index)
        if volume is not None: volume = volume.reindex(df.index)
    
        # 2) 지표 (ATR clip 포함)
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
    
        # 3) 피처
        feats_dict = {'Level_ATR100': (df['Close'] - df['MAtrend']) / atr100_robust}
        if self.use_slope: feats_dict['Slope_ATR100'] = df['MACDdiff'] / atr100_robust
        if self.use_ac and ac is not None: feats_dict['AC1_5'] = ac
        if self.use_rsi and rsi is not None: feats_dict['RSI'] = rsi
        if use_volume and volume is not None:
            vol_atr = (volume / atr100_robust).replace([np.inf, -np.inf], np.nan)
            feats_dict['Volume_ATR'] = vol_atr
    
        feats = pd.DataFrame(feats_dict)
    
        # 4) smoothing
        if self.smoothing_window_mr > 1:
            for col in ['RSI', 'AC1_5', 'Volume_ATR']:
                if col in feats: feats[col] = feats[col].rolling(self.smoothing_window_mr, min_periods=1).mean()
    
        feats = feats.dropna()
        if feats.empty:
            print("Error: No valid features after NaN removal."); return None, None, None, None
    
        # 5) 정렬/스케일/적합
        common_index = df.index.intersection(feats.index)
        df = df.loc[common_index].copy()
        feats = feats.loc[common_index].copy()
        if 'AC1_5' in feats: df['AC1_5'] = feats['AC1_5']
    
        scaler = StandardScaler()
        X = scaler.fit_transform(feats.values)
        if X.shape[0] < self.n_states:
            print(f"Error: Not enough data ({X.shape[0]}) for HMM."); return None, None, None, None
    
        model, best_score = fit_best_hmm(X, n_states=self.n_states, n_trials=self.n_trials)
        if model is None:
            print("HMM fitting failed."); return None, None, None, None
    
        try:
            states = model.predict(X)
        except Exception as e:
            print(f"HMM prediction failed: {e}"); return None, None, None, None
    
        # 6) 라벨링 (정수 레짐 유지; 내부적으로 영문 라벨도 생성해 출력용으로만 사용)
        mapping, summary = label_states(
            df, states, n_states=self.n_states,
            model=model, features_df=feats, calc_diagnostics=True
        )
        df['state']  = states
        df['regime'] = [mapping.get(int(s), 99) for s in states]  # int regime 유지
    
        # 영문 라벨(출력용): 정렬 순서대로 ['Bull','Weak Bull','Weak Bear','Bear',...] 매핑
        english_labels = ["Bull", "Weak Bull", "Weak Bear", "Bear", "Other5", "Other6"]
        # 정수레짐 -> 영문
        # (정수레짐 1이 'Bull'…)
        uniq_regs_sorted = sorted(set(v for v in mapping.values() if v != 99))
        int_to_eng = {reg_i: (english_labels[reg_i-1] if reg_i-1 < len(english_labels) else f"Other{reg_i}")
                      for reg_i in uniq_regs_sorted}
    
        recent = df.loc[df.index >= pd.to_datetime(self.plot_start)].copy()
        if not recent.empty:
            plot_regimes_banner(recent, ticker=self.ticker,
                                title_note=f"Numeric Regimes (means-sorted), K={self.n_states}")
    
        # 결과 출력
        print(f"\n--- HMM Results (K={self.n_states}) ---")
        print(f"Features Used: {list(feats.columns)}")
        print("Best log-likelihood:", best_score)
        if not summary.empty:
            print_cols = ['state', 'trend_score', 'mom_score', 'macd_score']
            if 'ac1_5_score' in summary.columns: print_cols.append('ac1_5_score')
            print(summary.sort_values('trend_score', ascending=False)[print_cols])
        else:
            print("Summary table could not be generated.")
    
        feats_scaled = pd.DataFrame(X, index=df.index, columns=feats.columns)
    
        # ---- 최종 진단 (unscaled means + 다음날 상태확률) ----
        try:
            print("\n--- Final Model Regime Characteristics (Unscaled Feature Means) ---")
            unscaled_means = (model.means_ * scaler.scale_) + scaler.mean_
            mean_df = pd.DataFrame(unscaled_means, columns=feats.columns)
            # 상태 i의 정수레짐 라벨(1..n)과 영문 라벨을 같이 표시
            mean_df['regime_int'] = [mapping.get(i, 99) for i in range(model.n_components)]
            mean_df['regime_eng'] = [int_to_eng.get(mapping.get(i, 99), "Unknown") for i in range(model.n_components)]
            cols = ['regime_int', 'regime_eng'] + [c for c in feats.columns]
            sort_col = 'Level_ATR100' if 'Level_ATR100' in feats.columns else feats.columns[0]
            print(mean_df[cols].sort_values(sort_col, ascending=False).to_string(float_format="%.2f"))
    
            # 다음날 상태확률 (현재 마지막 state 기준)
            last_state = df['state'].iloc[-1]
            probs = model.transmat_[last_state, :]
            print("\n--- Next Day State Probabilities ---")
            print(f"Based on last state: state={last_state}, regime={mapping.get(int(last_state), 99)} ({int_to_eng.get(mapping.get(int(last_state), 99), 'Unknown')})")
            print("state | P(next) | regime_int | regime_eng")
            print("---------------------------------------")
            for i, p in enumerate(probs):
                r_int = mapping.get(i, 99)
                r_eng = int_to_eng.get(r_int, "Unknown")
                print(f"{i:>5} | {p:7.2%} | {r_int:10} | {r_eng}")
        except Exception as e:
            print(f"Could not generate final diagnostics: {e}")
    
        # 내부 저장
        self.model = model
        self.mapping = mapping            # {orig_state -> int regime}
        self.summary_ = summary
        self.features_used_ = list(feats.columns)
        self.best_score_ = best_score
        self.scaler_ = scaler
    
        return df, feats_scaled, model, mapping


# # --------- Script-style 실행 (옵션) ---------
# if __name__ == "__main__":
#     N_STATES_TO_RUN = 4
#     TICKER_TO_RUN = "^GSPC"
#     START_DATE = "1995-01-01"
#     PLOT_START = "2020-01-01"
#     MA_WINDOW = 120
#     AC_WINDOW = 60
#     USE_SLOPE = False
#     USE_AC = False
#     USE_RSI = True
#     USE_VOLUME = False

#     analyzer = HMMRegimeAnalyzer(
#         ticker=TICKER_TO_RUN,
#         start=START_DATE,
#         plot_start=PLOT_START,
#         n_states=N_STATES_TO_RUN,
#         n_trials=20,
#         ma_window=MA_WINDOW,
#         rsi_period=14,
#         ac_window=AC_WINDOW,
#         smoothing_window_mr=1,
#         use_slope=USE_SLOPE,
#         use_ac=USE_AC,
#         use_rsi=USE_RSI,
#         use_volume=USE_VOLUME
#     )

#     df, feats_scaled, model, mapping = analyzer.run()
#     if df is None:
#         raise SystemExit("HMM 실행 실패: df가 None입니다.")

#     df_ix = with_regime_index(df)
