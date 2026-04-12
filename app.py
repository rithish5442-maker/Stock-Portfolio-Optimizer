import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.8rem 0 0.2rem 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="metric-container"] {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Stock Portfolio Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Markowitz MPT · Sharpe Optimization · Monte Carlo Simulation</div>', unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Portfolio Settings")

    tickers_input = st.text_input(
        "Stock Tickers (comma-separated)",
        value="AAPL,MSFT,GOOGL,AMZN,META",
        help="Enter valid NSE/NYSE ticker symbols e.g. RELIANCE.NS, TCS.NS or AAPL, MSFT"
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365 * 3))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    num_simulations = st.slider("Monte Carlo Simulations", 500, 10000, 3000, step=500)
    risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 15.0, 4.5, step=0.1) / 100

    st.markdown("---")
    run_btn = st.button("🚀 Run Optimizer", type="primary", use_container_width=True)
    st.caption("💡 Tip: Use `.NS` suffix for Indian stocks e.g. `TCS.NS`")

# ─── Core Functions ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(tickers, start, end):
    try:
        raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        if len(tickers) == 1:
            df = raw[["Close"]].copy()
            df.columns = tickers
        else:
            df = raw["Close"].copy()
        df.dropna(axis=1, how="all", inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception:
        return None


def calc_returns(prices):
    return prices.pct_change().dropna()


def portfolio_performance(weights, mean_returns, cov_matrix, rf):
    ann_return = np.sum(mean_returns * weights) * 252
    ann_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else 0
    return ann_return, ann_vol, sharpe


def optimize_portfolio(mean_returns, cov_matrix, rf, objective="sharpe"):
    n = len(mean_returns)
    init_w = np.array([1 / n] * n)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

    if objective == "sharpe":
        fun = lambda w: -portfolio_performance(w, mean_returns, cov_matrix, rf)[2]
    else:  # min vol
        fun = lambda w: portfolio_performance(w, mean_returns, cov_matrix, rf)[1]

    result = minimize(fun, init_w, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000})
    return result.x


def efficient_frontier(mean_returns, cov_matrix, rf, num_points=80):
    n = len(mean_returns)
    target_rets = np.linspace(mean_returns.min() * 252, mean_returns.max() * 252, num_points)
    eff_vols = []

    for target in target_rets:
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {"type": "eq", "fun": lambda x, t=target: portfolio_performance(x, mean_returns, cov_matrix, rf)[0] - t}
        ]
        res = minimize(
            lambda x: portfolio_performance(x, mean_returns, cov_matrix, rf)[1],
            [1 / n] * n,
            method="SLSQP",
            bounds=tuple((0, 1) for _ in range(n)),
            constraints=constraints,
            options={"maxiter": 500}
        )
        eff_vols.append(res.fun * 100 if res.success else np.nan)

    return target_rets * 100, eff_vols


def monte_carlo(mean_returns, cov_matrix, rf, num_sims):
    n = len(mean_returns)
    returns_out, vols_out, sharpes_out, weights_out = [], [], [], []

    for _ in range(num_sims):
        w = np.random.dirichlet(np.ones(n))
        r, s, sh = portfolio_performance(w, mean_returns, cov_matrix, rf)
        returns_out.append(r * 100)
        vols_out.append(s * 100)
        sharpes_out.append(sh)
        weights_out.append(w)

    return (np.array(returns_out), np.array(vols_out),
            np.array(sharpes_out), weights_out)


# ─── Main App ───────────────────────────────────────────────────────────────────
if run_btn or "prices" not in st.session_state:
    tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers_list) < 2:
        st.warning("Please enter at least 2 stock tickers.")
        st.stop()

    with st.spinner("📥 Fetching market data..."):
        prices = load_data(tickers_list, start_date, end_date)

    if prices is None or prices.empty:
        st.error("❌ Could not fetch data. Check tickers / date range.")
        st.stop()

    st.session_state["prices"] = prices
    st.session_state["tickers_list"] = tickers_list
    st.session_state["rf"] = risk_free_rate
    st.session_state["num_sims"] = num_simulations
    # Clear cached computed results when settings change
    for key in ["ef_result", "ms_result", "mv_result", "mc_result"]:
        st.session_state.pop(key, None)

prices = st.session_state.get("prices")
if prices is None:
    st.info("👈 Configure settings and click **Run Optimizer** to begin.")
    st.stop()

rf = st.session_state.get("rf", risk_free_rate)
num_sims = st.session_state.get("num_sims", num_simulations)
available = list(prices.columns)
removed = set(st.session_state["tickers_list"]) - set(available)
if removed:
    st.warning(f"⚠️ Skipped unavailable tickers: {', '.join(removed)}")

daily_ret = calc_returns(prices)
mean_ret = daily_ret.mean()
cov_mat = daily_ret.cov()

ann_ret = mean_ret * 252 * 100
ann_vol = daily_ret.std() * np.sqrt(252) * 100
sharpe_ind = (ann_ret / 100 - rf) / (ann_vol / 100)

# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Stock Comparison",
    "⚖️ Risk vs Return",
    "🎯 Efficient Frontier",
    "🏆 Sharpe Optimization",
    "🎲 Monte Carlo"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — STOCK COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📈 Stock Price History & Comparison")

    norm = prices / prices.iloc[0] * 100
    fig1 = go.Figure()
    for col in norm.columns:
        fig1.add_trace(go.Scatter(x=norm.index, y=norm[col], name=col,
                                  mode="lines", line=dict(width=2)))
    fig1.update_layout(title="Normalized Price (Base = 100)", template="plotly_dark",
                       height=420, hovermode="x unified",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        colors = ["#00CC96" if v > 0 else "#EF553B" for v in ann_ret.values]
        fig2 = go.Figure(go.Bar(x=ann_ret.index, y=ann_ret.values, marker_color=colors,
                                text=[f"{v:.2f}%" for v in ann_ret.values], textposition="outside"))
        fig2.update_layout(title="Annualized Return (%)", template="plotly_dark", height=360,
                           yaxis_title="Return (%)")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        fig3 = go.Figure(go.Bar(x=ann_vol.index, y=ann_vol.values, marker_color="#AB63FA",
                                text=[f"{v:.2f}%" for v in ann_vol.values], textposition="outside"))
        fig3.update_layout(title="Annualized Volatility (%)", template="plotly_dark", height=360,
                           yaxis_title="Volatility (%)")
        st.plotly_chart(fig3, use_container_width=True)

    corr = daily_ret.corr().round(2)
    fig4 = px.imshow(corr, text_auto=True, color_continuous_scale="RdYlGn",
                     zmin=-1, zmax=1, title="Correlation Heatmap")
    fig4.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK VS RETURN
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("⚖️ Individual Stock Risk vs Return")

    fig5 = go.Figure()
    for ticker in available:
        fig5.add_trace(go.Scatter(
            x=[ann_vol[ticker]], y=[ann_ret[ticker]],
            mode="markers+text", name=ticker, text=[ticker],
            textposition="top center",
            marker=dict(size=16, line=dict(color="white", width=1)),
            hovertemplate=f"<b>{ticker}</b><br>Return: {ann_ret[ticker]:.2f}%<br>"
                          f"Risk: {ann_vol[ticker]:.2f}%<br>Sharpe: {sharpe_ind[ticker]:.3f}<extra></extra>"
        ))

    # Capital Market Line approximation
    x_cml = np.linspace(0, ann_vol.max() * 1.2, 100)
    max_sharpe_slope = sharpe_ind.max()
    y_cml = rf * 100 + max_sharpe_slope * x_cml
    fig5.add_trace(go.Scatter(x=x_cml, y=y_cml, mode="lines", name="CML (approx)",
                              line=dict(color="gold", dash="dash", width=1.5)))

    fig5.update_layout(title="Risk vs Return — Individual Stocks + Capital Market Line",
                       xaxis_title="Annualized Volatility (%)", yaxis_title="Annualized Return (%)",
                       template="plotly_dark", height=520, showlegend=True)
    st.plotly_chart(fig5, use_container_width=True)

    max_dd = (prices / prices.cummax() - 1).min().mul(100).round(2)
    summary = pd.DataFrame({
        "Annual Return (%)": ann_ret.round(2),
        "Annual Volatility (%)": ann_vol.round(2),
        "Sharpe Ratio": sharpe_ind.round(3),
        "Max Drawdown (%)": max_dd
    })
    st.dataframe(summary.style.background_gradient(cmap="RdYlGn", subset=["Sharpe Ratio"]),
                 use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — EFFICIENT FRONTIER
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🎯 Efficient Frontier — Markowitz Modern Portfolio Theory")

    if "ef_result" not in st.session_state:
        with st.spinner("Computing efficient frontier..."):
            ef_rets, ef_vols = efficient_frontier(mean_ret, cov_mat, rf)
            ms_w = optimize_portfolio(mean_ret, cov_mat, rf, "sharpe")
            ms_r, ms_v, ms_sh = portfolio_performance(ms_w, mean_ret, cov_mat, rf)
            mv_w = optimize_portfolio(mean_ret, cov_mat, rf, "min_vol")
            mv_r, mv_v, mv_sh = portfolio_performance(mv_w, mean_ret, cov_mat, rf)
            st.session_state["ef_result"] = (ef_rets, ef_vols)
            st.session_state["ms_result"] = (ms_w, ms_r, ms_v, ms_sh)
            st.session_state["mv_result"] = (mv_w, mv_r, mv_v, mv_sh)
    else:
        ef_rets, ef_vols = st.session_state["ef_result"]
        ms_w, ms_r, ms_v, ms_sh = st.session_state["ms_result"]
        mv_w, mv_r, mv_v, mv_sh = st.session_state["mv_result"]

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=ef_vols, y=ef_rets, mode="lines", name="Efficient Frontier",
                              line=dict(color="#667eea", width=3)))
    fig6.add_trace(go.Scatter(x=[ms_v * 100], y=[ms_r * 100], mode="markers",
                              name=f"Max Sharpe ({ms_sh:.2f})",
                              marker=dict(color="gold", size=16, symbol="star",
                                          line=dict(color="white", width=1))))
    fig6.add_trace(go.Scatter(x=[mv_v * 100], y=[mv_r * 100], mode="markers",
                              name="Min Volatility",
                              marker=dict(color="#00CC96", size=14, symbol="diamond",
                                          line=dict(color="white", width=1))))
    for ticker in available:
        fig6.add_trace(go.Scatter(
            x=[ann_vol[ticker]], y=[ann_ret[ticker]],
            mode="markers+text", text=[ticker], textposition="top center",
            marker=dict(size=10, symbol="circle-open",
                        line=dict(width=2, color="white")),
            showlegend=False,
            hovertemplate=f"<b>{ticker}</b><br>Return: {ann_ret[ticker]:.2f}%<br>Risk: {ann_vol[ticker]:.2f}%<extra></extra>"
        ))

    fig6.update_layout(title="Efficient Frontier + Key Portfolios",
                       xaxis_title="Annualized Volatility (%)", yaxis_title="Annualized Return (%)",
                       template="plotly_dark", height=560, hovermode="closest")
    st.plotly_chart(fig6, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**⭐ Max Sharpe Portfolio**")
        for t, w in zip(available, ms_w):
            st.write(f"- {t}: **{w*100:.1f}%**")
    with c2:
        st.markdown("**💚 Min Volatility Portfolio**")
        for t, w in zip(available, mv_w):
            st.write(f"- {t}: **{w*100:.1f}%**")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — SHARPE OPTIMIZATION
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🏆 Maximum Sharpe Ratio Portfolio")

    if "ms_result" not in st.session_state:
        ms_w = optimize_portfolio(mean_ret, cov_mat, rf, "sharpe")
        ms_r, ms_v, ms_sh = portfolio_performance(ms_w, mean_ret, cov_mat, rf)
        st.session_state["ms_result"] = (ms_w, ms_r, ms_v, ms_sh)
    else:
        ms_w, ms_r, ms_v, ms_sh = st.session_state["ms_result"]

    c1, c2, c3 = st.columns(3)
    c1.metric("📈 Expected Annual Return", f"{ms_r*100:.2f}%")
    c2.metric("📉 Annual Volatility (Risk)", f"{ms_v*100:.2f}%")
    c3.metric("⭐ Sharpe Ratio", f"{ms_sh:.3f}")

    st.markdown("---")

    fig7 = go.Figure(go.Pie(
        labels=available,
        values=(ms_w * 100).round(2),
        hole=0.42,
        textinfo="label+percent",
        marker=dict(line=dict(color="white", width=2))
    ))
    fig7.update_layout(title="Optimal Allocation — Max Sharpe Portfolio",
                       template="plotly_dark", height=450)
    st.plotly_chart(fig7, use_container_width=True)

    weights_df = pd.DataFrame({
        "Ticker": available,
        "Weight (%)": (ms_w * 100).round(2),
        "Return Contribution (%)": (ms_w * mean_ret.values * 252 * 100).round(3)
    }).sort_values("Weight (%)", ascending=False)
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("💰 Investment Allocation Calculator")
    invest_amt = st.number_input("Investment Amount (₹)", min_value=1000, value=100000, step=5000)
    alloc_df = pd.DataFrame({
        "Stock": available,
        "Allocation (%)": (ms_w * 100).round(2),
        "Amount (₹)": (ms_w * invest_amt).round(0).astype(int)
    }).sort_values("Allocation (%)", ascending=False)
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — MONTE CARLO
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🎲 Monte Carlo Portfolio Simulation")

    if "mc_result" not in st.session_state:
        with st.spinner(f"Running {num_sims:,} random portfolio simulations..."):
            mc_rets, mc_vols, mc_sharpes, mc_weights = monte_carlo(mean_ret, cov_mat, rf, num_sims)
            st.session_state["mc_result"] = (mc_rets, mc_vols, mc_sharpes, mc_weights)
    else:
        mc_rets, mc_vols, mc_sharpes, mc_weights = st.session_state["mc_result"]

    best_idx = np.argmax(mc_sharpes)

    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(
        x=mc_vols, y=mc_rets, mode="markers",
        name="Simulated Portfolios",
        marker=dict(size=3, color=mc_sharpes, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Sharpe Ratio"), opacity=0.7),
        hovertemplate="Return: %{y:.2f}%<br>Risk: %{x:.2f}%<extra></extra>"
    ))
    fig8.add_trace(go.Scatter(
        x=[mc_vols[best_idx]], y=[mc_rets[best_idx]], mode="markers",
        name=f"Best Portfolio (Sharpe={mc_sharpes[best_idx]:.3f})",
        marker=dict(color="gold", size=18, symbol="star", line=dict(color="white", width=2))
    ))
    fig8.update_layout(
        title=f"Monte Carlo Simulation — {num_sims:,} Random Portfolios",
        xaxis_title="Annualized Volatility (%)", yaxis_title="Annualized Return (%)",
        template="plotly_dark", height=560, hovermode="closest"
    )
    st.plotly_chart(fig8, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Best Sharpe Ratio", f"{mc_sharpes[best_idx]:.3f}")
    c2.metric("📈 Return at Best Sharpe", f"{mc_rets[best_idx]:.2f}%")
    c3.metric("📉 Risk at Best Sharpe", f"{mc_vols[best_idx]:.2f}%")

    st.markdown("---")
    best_mc_w = mc_weights[best_idx]
    st.markdown("**🎯 Best Monte Carlo Portfolio Weights:**")
    mc_w_df = pd.DataFrame({
        "Ticker": available,
        "Weight (%)": (best_mc_w * 100).round(2)
    }).sort_values("Weight (%)", ascending=False)
    st.dataframe(mc_w_df, use_container_width=True, hide_index=True)

    fig9 = go.Figure(go.Histogram(
        x=mc_sharpes, nbinsx=60, marker_color="#667eea", name="Sharpe Distribution"
    ))
    fig9.add_vline(x=mc_sharpes[best_idx], line_dash="dash", line_color="gold",
                   annotation_text=f"Best: {mc_sharpes[best_idx]:.3f}", annotation_position="top right")
    fig9.update_layout(
        title="Sharpe Ratio Distribution Across All Simulated Portfolios",
        xaxis_title="Sharpe Ratio", yaxis_title="Count",
        template="plotly_dark", height=350
    )
    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("---")
    st.caption("📌 Built with Streamlit · yfinance · SciPy · Plotly | Portfolio optimization using Markowitz MPT")
