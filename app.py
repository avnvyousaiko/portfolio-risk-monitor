import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 0. 页面全局设置
# ==========================================
st.set_page_config(page_title="量化组合风险监控 Dashboard", layout="wide")
st.title("📊 基金投资组合风险监控看板")

# ==========================================
# 1. 数据加载与预处理
# ==========================================
@st.cache_data
def load_data():
    df_returns = pd.read_csv('portfolio_returns_processed.csv')
    df_returns['date'] = pd.to_datetime(df_returns['date'])
    df_returns.set_index('date', inplace=True)
    
    try:
        df_holdings = pd.read_csv('holdings.csv')
        df_holdings.columns = df_holdings.columns.str.strip().str.lower()
        df_holdings['ticker'] = df_holdings['ticker'].astype(str).str.zfill(6)
        
        if 'weight' not in df_holdings.columns:
            daily_mv = df_holdings.groupby('date')['market_value'].transform('sum')
            df_holdings['weight'] = df_holdings['market_value'] / daily_mv
            
        df_holdings = df_holdings.reset_index(drop=True)
    except:
        df_holdings = pd.DataFrame() 
        
    return df_returns, df_holdings

df_ret, df_holdings = load_data()

# 核心常量参数
TRADING_DAYS = 252
RISK_FREE_RATE = 0.02

# 预先计算全局核心指标
cumulative_return = df_ret['nav'].iloc[-1] - 1
n_days = len(df_ret)
annual_return = (1 + cumulative_return) ** (TRADING_DAYS / n_days) - 1
annual_volatility = df_ret['portfolio_return'].std() * np.sqrt(TRADING_DAYS)
sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_volatility
max_drawdown = df_ret['drawdown'].min()

bench_cum_return = df_ret['bench_nav'].iloc[-1] - 1
bench_annual_return = (1 + bench_cum_return) ** (TRADING_DAYS / n_days) - 1
# 计算年化超额收益
excess_return_ann = annual_return - bench_annual_return

current_market_value = df_holdings[df_holdings['date'] == df_holdings['date'].max()]['market_value'].sum() if not df_holdings.empty else 0

# ==========================================
# 2. 页面布局搭建
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["组合总览", "净值与回撤", "持仓与行业暴露", "风险指标"])

# --- Page 1: 组合总览 ---
with tab1:
    st.subheader("💡 核心绩效指标 (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前组合总市值 (元)", f"¥ {current_market_value:,.2f}")
    col2.metric("累计收益率", f"{cumulative_return*100:.2f}%")
    col3.metric("组合年化收益率", f"{annual_return*100:.2f}%")
    col4.metric("最大回撤", f"{max_drawdown*100:.2f}%")
    
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("年化波动率", f"{annual_volatility*100:.2f}%")
    col6.metric("Sharpe Ratio (夏普比率)", f"{sharpe_ratio:.2f}")
    # 🌟 按照要求：最后一格替换为相对基准超额收益
    col7.metric("相对基准超额收益 (年化)", f"{excess_return_ann*100:.2f}%")

# --- Page 2: 净值与回撤 ---
with tab2:
    st.subheader("📈 历史走势对比")
    fig_nav = go.Figure()
    fig_nav.add_trace(go.Scatter(x=df_ret.index, y=df_ret['nav'], name='组合净值', line=dict(color='firebrick', width=2)))
    fig_nav.add_trace(go.Scatter(x=df_ret.index, y=df_ret['bench_nav'], name='沪深300基准', line=dict(color='royalblue', width=1.5, dash='dash')))
    fig_nav.update_layout(title="组合净值 vs 基准净值", hovermode="x unified")
    st.plotly_chart(fig_nav, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        colors = ['red' if val > 0 else 'green' for val in df_ret['portfolio_return']]
        fig_ret = go.Figure(data=[go.Bar(x=df_ret.index, y=df_ret['portfolio_return'], marker_color=colors)])
        fig_ret.update_layout(title="每日绝对收益率分布")
        st.plotly_chart(fig_ret, use_container_width=True)
        
    with col2:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=df_ret.index, y=df_ret['drawdown'], fill='tozeroy', fillcolor='rgba(255,0,0,0.2)', line=dict(color='red', width=1), name='回撤'))
        fig_dd.update_layout(title="历史回撤曲线", yaxis=dict(tickformat=".2%"))
        st.plotly_chart(fig_dd, use_container_width=True)

# --- Page 3: 持仓与行业暴露 ---
with tab3:
    if not df_holdings.empty and 'industry' in df_holdings.columns:
        latest_date = df_holdings['date'].max()
        st.subheader(f"📊 截面持仓 analysis (截至 {latest_date})")
        
        current_port = df_holdings[df_holdings['date'] == latest_date].copy()
        top10 = current_port.sort_values('weight', ascending=False).head(10).reset_index(drop=True)
        top10.columns = top10.columns.astype(str)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_top10 = px.bar(top10, x='ticker', y='weight', title="Top 10 重仓股占比", text_auto='.2%')
            st.plotly_chart(fig_top10, use_container_width=True)
            hhi_stock = (current_port['weight'] ** 2).sum()
            st.info(f"**个股集中度**: Top 10 权重合计 **{top10['weight'].sum()*100:.1f}%** | HHI: **{hhi_stock:.4f}**")

        with col2:
            ind_weight = current_port.groupby('industry')['weight'].sum().reset_index()
            fig_ind = px.pie(ind_weight, values='weight', names='industry', title="行业权重暴露", hole=0.3)
            st.plotly_chart(fig_ind, use_container_width=True)
            hhi_ind = (ind_weight['weight'] ** 2).sum()
            st.info(f"**行业集中度**: 第一大行业 **{ind_weight.sort_values('weight', ascending=False).iloc[0]['industry']}** | 行业 HHI: **{hhi_ind:.4f}**")
    else:
        st.warning("未能成功读取行业信息。")

# --- Page 4: 风险指标 ---
with tab4:
    st.subheader("🛡️ 动态与下行风险监控")
    window_vol = 20
    window_beta = 60
    
    df_ret['rolling_vol'] = df_ret['portfolio_return'].rolling(window_vol).std() * np.sqrt(TRADING_DAYS)
    cov_roll = df_ret['portfolio_return'].rolling(window_beta).cov(df_ret['benchmark_return'])
    var_roll = df_ret['benchmark_return'].rolling(window_beta).var()
    df_ret['rolling_beta'] = cov_roll / var_roll
    
    historical_VaR_95 = np.percentile(df_ret['portfolio_return'], 5)
    tracking_error = (df_ret['portfolio_return'] - df_ret['benchmark_return']).std() * np.sqrt(TRADING_DAYS)
    information_ratio = excess_return_ann / tracking_error if tracking_error != 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("95% 历史 VaR (日频)", f"{historical_VaR_95*100:.2f}%")
    c2.metric("Tracking Error (跟踪误差)", f"{tracking_error*100:.2f}%")
    c3.metric("Information Ratio (信息比率)", f"{information_ratio:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        fig_rvol = px.line(df_ret, y='rolling_vol', title=f"Rolling {window_vol}-Day 年化波动率")
        fig_rvol.add_hline(y=annual_volatility, line_dash="dot", annotation_text="全局平均")
        st.plotly_chart(fig_rvol, use_container_width=True)
    with col2:
        fig_rbeta = px.line(df_ret, y='rolling_beta', title=f"Rolling {window_beta}-Day 动态 Beta")
        fig_rbeta.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="Beta=1")
        st.plotly_chart(fig_rbeta, use_container_width=True)
