import sys
import os
import math
import time
import datetime
import logging
import traceback  
from logging.handlers import RotatingFileHandler
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from datetime import timezone
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import yfinance as yf  
import pytz            
import pandas_market_calendars as mcal
from ib_insync import *
from supabase import create_client, Client

from qqq_strategy_email import (
    QQQSignalContext,
    evaluate_qqq_signals,
    format_signal_email_section,
)
from qqq_contract_scanner import (
    LiveOptionQuote,
    build_downtrend_combo_candidate,
    build_uptrend_funded_call_candidate,
    build_wall_debit_spread_candidate,
    format_contract_scan_section,
)

CONFIG_DIR = os.path.expanduser('~/market_dashboard')
if CONFIG_DIR not in sys.path:
    sys.path.append(CONFIG_DIR)

try:
    import market_config as cfg
except ImportError:
    print("❌ 致命错误：找不到 market_config.py 密钥文件！")
    sys.exit(1)

logging.getLogger('ib_insync').setLevel(logging.CRITICAL)

# ================= 资金与时区配置 =================
NY_TZ = pytz.timezone('America/New_York') 
ACCOUNT_BASE = 100000.0  
MARGIN_LIMIT_PCT = 0.40  
COOLDOWN_DAYS = 3        

@dataclass
class OptionLeg:
    symbol: str
    right: str
    strike: float
    expiry: str
    mid: float
    bid: float
    ask: float
    delta: float
    theta: float
    gamma: float
    vega: float
    iv: float
    volume: int
    open_interest: int

    @property
    def spread_pct(self) -> float:
        if self.mid <= 0: return 999.0
        return (self.ask - self.bid) / self.mid * 100

    @property
    def is_liquid(self) -> bool:
        return self.spread_pct < 15.0 and self.open_interest > 10

class UnifiedLedgerBuilder:
    def __init__(self, host='127.0.0.1', port=4001, client_id=889):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.market_data = {}
        self.action_logs = []
        self.qqq_signal_context = None
        self.qqq_strategy_signals = ()
        
        self.setup_logging()
        self.supabase = self._init_supabase()

    def _init_supabase(self, retries=3):
        for i in range(retries):
            try:
                return create_client(cfg.SUPABASE_URL, cfg.SUPABASE_KEY)
            except Exception as e:
                if i == retries - 1: sys.exit(1)
                time.sleep(2)

    def setup_logging(self):
        log_dir = os.path.expanduser('~/TradingRadar/logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger('BuilderDaemon')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            log_file = os.path.join(log_dir, 'builder_daemon.log')
            file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_action(self, msg):
        print(msg)
        self.action_logs.append(msg)
        self.logger.info(msg)

    def send_report(self, subject, body):
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['From'] = cfg.SENDER_EMAIL
        msg['To'] = cfg.QQ_RECEIVER_EMAIL
        msg['Subject'] = Header(subject, 'utf-8')
        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=10)
            server.login(cfg.SENDER_EMAIL, cfg.APP_PASSWORD)
            server.sendmail(cfg.SENDER_EMAIL, [cfg.QQ_RECEIVER_EMAIL], msg.as_string())
            server.quit()
        except Exception as e:
            self.logger.error(f"❌ 邮件发送失败: {e}")

    def execute_db_with_retry(self, query, action_name, max_retries=3):
        for attempt in range(max_retries):
            try:
                res = query.execute()
                return res
            except Exception as e:
                self.logger.warning(f"⚠️ {action_name} 数据库操作异常 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        self.log_action(f"❌ {action_name} 彻底失败，请检查网络或 Supabase 状态。")
        return None

    def connect(self):
        self.log_action(f"[*] 执行中枢连接 IBKR...")
        self.ib.connect(self.host, self.port, clientId=self.client_id, readonly=True)
        self.ib.reqMarketDataType(1)

    def disconnect(self):
        if self.ib.isConnected(): self.ib.disconnect()

    def get_stock_price(self, symbol):
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        ticker = self.ib.reqMktData(contract, '', False, False)
        self.ib.sleep(1)
        
        price = None
        if ticker.last and not math.isnan(ticker.last): 
            price = ticker.last
        elif ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0: 
            price = (ticker.bid + ticker.ask) / 2
        elif ticker.close and not math.isnan(ticker.close): 
            price = ticker.close
            
        self.market_data[symbol] = price
        self.ib.cancelMktData(contract)
        return price

    def get_vix(self):
        try:
            vix_ticker = yf.Ticker('^VIX')
            vix_price = float(vix_ticker.history(period='1d')['Close'].iloc[-1])
            if vix_price > 0 and not math.isnan(vix_price):
                return vix_price
        except Exception as e:
            self.logger.warning(f"yfinance 抓取 VIX 失败，尝试备用方案: {e}")

        try:
            contract = Index('VIX', 'CBOE')
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(2)
            
            price = None
            if ticker.last and not math.isnan(ticker.last): price = ticker.last
            elif ticker.bid and ticker.ask and ticker.bid > 0: price = (ticker.bid + ticker.ask) / 2
            elif ticker.close and not math.isnan(ticker.close): price = ticker.close
            
            self.ib.cancelMktData(contract)
            return price if price else 20.0
        except Exception as e:
            self.logger.warning(f"获取 VIX 全部失败: {e}")
        return 20.0  

    def is_third_friday(self, date_obj):
        return date_obj.weekday() == 4 and 15 <= date_obj.day <= 21

    def get_liquid_expiry(self, expirations, target_days, tolerance_pct=0.3):
        # 【修复】：强制美东时区锁
        today = datetime.datetime.now(NY_TZ).date()
        target_date = today + datetime.timedelta(days=target_days)
        min_days = int(target_days * (1 - tolerance_pct))
        max_days = int(target_days * (1 + tolerance_pct))
        
        valid_dates = []
        monthly_dates = []
        
        for exp in expirations:
            exp_date = datetime.datetime.strptime(exp, '%Y%m%d').date()
            if exp_date <= today: continue  
            days_to_exp = (exp_date - today).days
            if min_days <= days_to_exp <= max_days:
                valid_dates.append(exp_date)
                if self.is_third_friday(exp_date): monthly_dates.append(exp_date)
        
        if monthly_dates:
            return min(monthly_dates, key=lambda d: abs(d - target_date)).strftime('%Y%m%d')
        if valid_dates:
            return min(valid_dates, key=lambda d: abs(d - target_date)).strftime('%Y%m%d')
            
        all_future_dates = [datetime.datetime.strptime(exp, '%Y%m%d').date() for exp in expirations if datetime.datetime.strptime(exp, '%Y%m%d').date() > today]
        if all_future_dates:
            return min(all_future_dates, key=lambda d: abs(d - target_date)).strftime('%Y%m%d')
            
        return expirations[-1] if expirations else None

    def get_greeks_with_retry(self, opt_contract, max_retries=4):
        for attempt in range(max_retries):
            ticker = self.ib.reqMktData(opt_contract, "106", False, False)
            self.ib.sleep(3 + attempt * 0.5)  
            try:
                if ticker.modelGreeks and ticker.modelGreeks.delta is not None:
                    bid = ticker.bid if ticker.bid and not math.isnan(ticker.bid) else 0
                    ask = ticker.ask if ticker.ask and not math.isnan(ticker.ask) else 0
                    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else ticker.last
                    
                    if bid > 0 and ask > 0 and mid > 0:
                        if (ask - bid) / mid > 0.30:
                            return None 

                    greeks = {
                        'delta': ticker.modelGreeks.delta, 
                        'theta': ticker.modelGreeks.theta if ticker.modelGreeks.theta else 0,
                        'vega': ticker.modelGreeks.vega if ticker.modelGreeks.vega else 0, 
                        'gamma': ticker.modelGreeks.gamma if ticker.modelGreeks.gamma else 0
                    }
                    return {'mid': mid if mid and not math.isnan(mid) else 0, **greeks}
            finally:
                self.ib.cancelMktData(opt_contract)
        return None

    def find_optimal_leg(self, symbol, right, target_days, target_otm_pct=None, explicit_strike=None, target_delta=None, min_premium=0.15):
        stock = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(stock)
        current_price = self.market_data.get(symbol) or self.get_stock_price(symbol)
        
        if current_price is None or math.isnan(current_price):
            self.log_action(f"⚠️ {symbol} 无法获取现价，跳过该腿寻找。")
            return None
        
        chains = self.ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
        
        if not chains:
            self.log_action(f"⚠️ {symbol} 无法获取期权链数据，跳过该腿寻找。")
            return None
            
        chain = next((c for c in chains if c.exchange == 'SMART'), chains[0])
        best_expiry = self.get_liquid_expiry(chain.expirations, target_days)
        if not best_expiry: return None
        
        target_strike_price = float(explicit_strike) if explicit_strike is not None else current_price * (1 + target_otm_pct)
        strikes = sorted(chain.strikes)
        closest_strikes = sorted(strikes, key=lambda x: abs(x - target_strike_price))[:3]
        
        best_contract_data = None
        min_delta_diff = 999
        
        for strike in closest_strikes:
            opt = Option(symbol, best_expiry, strike, right, 'SMART')
            self.ib.qualifyContracts(opt)
            data = self.get_greeks_with_retry(opt, max_retries=4)
            
            # 【修复核心】：终极兜底，全面提取 close, last, bid/ask
            if data is None:
                ticker = self.ib.reqMktData(opt, "106", False, False)
                self.ib.sleep(3)
                
                bid = ticker.bid if ticker.bid and not math.isnan(ticker.bid) else 0
                ask = ticker.ask if ticker.ask and not math.isnan(ticker.ask) else 0
                last_px = ticker.last if ticker.last and not math.isnan(ticker.last) else 0
                close_px = ticker.close if ticker.close and not math.isnan(ticker.close) else 0
                self.ib.cancelMktData(opt)

                mid = 0.0
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                elif last_px > 0:
                    mid = last_px
                elif close_px > 0:  # 盘后流产的终结者
                    mid = close_px
                
                if mid >= min_premium:
                    self.log_action(f"⚠️ {symbol} ${strike}{right} Greeks计算超时，使用现价 ${mid:.2f} 强行锁定！")
                    dummy_delta = -0.25 if right == 'P' else 0.25
                    data = {'mid': mid, 'delta': dummy_delta, 'theta': 0, 'vega': 0, 'gamma': 0}
                else:
                    continue

            actual_delta = data['delta']
            if abs(actual_delta) > 1.0: continue
            
            expected_sign = -1 if right == 'P' else 1
            if (actual_delta * expected_sign) < 0: continue

            if data['mid'] < min_premium:
                self.logger.info(f"过滤低价值合约: {symbol} ${strike} {right} | Mid=${data['mid']} < ${min_premium}")
                continue
            
            contract_dict = {
                'symbol': symbol, 'right': right, 'strike': strike, 'expiry': best_expiry, 
                'mid': data['mid'], 'delta': actual_delta, 'theta': data['theta'], 
                'vega': data['vega'], 'gamma': data['gamma']
            }
            
            if target_delta:
                diff = abs(abs(actual_delta) - abs(target_delta))
                if diff < min_delta_diff:
                    min_delta_diff = diff
                    best_contract_data = contract_dict
            else:
                best_contract_data = contract_dict
                break
                
        return best_contract_data

    # ================= 进阶模块：MILP网格搜索与多因子组合构建 =================
    
    def _get_clean_option_chain(self, symbol: str, right: str, target_days: int, price_range_pct: float = 0.15) -> Tuple[List[OptionLeg], float]:
        stock = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(stock)
        current_price = self.get_stock_price(symbol)
        
        if not current_price or math.isnan(current_price):
            return [], 0.0

        chains = self.ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
        
        if not chains:
            return [], current_price

        chain = next((c for c in chains if c.exchange == 'SMART'), chains[0])
        best_expiry = self.get_liquid_expiry(chain.expirations, target_days)
        if not best_expiry: return [], current_price
        
        strikes = sorted([s for s in chain.strikes if current_price * (1 - price_range_pct) <= s <= current_price * (1 + price_range_pct)])
        
        clean_chain = []
        batch_size = 20
        
        for i in range(0, len(strikes), batch_size):
            batch_strikes = strikes[i:i+batch_size]
            contracts = [Option(symbol, best_expiry, s, right, 'SMART') for s in batch_strikes]
            try:
                self.ib.qualifyContracts(*contracts)
            except Exception:
                continue

            opt_tickers = [self.ib.reqMktData(c, "106", False, False) for c in contracts]
            
            for _ in range(45):
                self.ib.sleep(0.1)
                valid_greeks = sum(1 for t in opt_tickers if t.modelGreeks and t.modelGreeks.impliedVol)
                if valid_greeks >= len(contracts) * 0.8:
                    break
            self.ib.sleep(0.5)

            for t in opt_tickers:
                if not t.modelGreeks or not t.modelGreeks.impliedVol or math.isnan(t.modelGreeks.impliedVol):
                    continue
                if t.modelGreeks.delta is None or math.isnan(t.modelGreeks.delta):
                    continue

                bid = t.bid if t.bid and not math.isnan(t.bid) else 0
                ask = t.ask if t.ask and not math.isnan(t.ask) else 0
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

                if mid <= 0: continue

                leg = OptionLeg(
                    symbol=symbol, right=right, strike=t.contract.strike, expiry=best_expiry,
                    mid=mid, bid=bid, ask=ask, delta=t.modelGreeks.delta,
                    theta=t.modelGreeks.theta or 0, gamma=t.modelGreeks.gamma or 0,
                    vega=t.modelGreeks.vega or 0, iv=t.modelGreeks.impliedVol,
                    volume=t.volume or 0, open_interest=t.openInterest or 0
                )
                
                if leg.is_liquid:
                    clean_chain.append(leg)

            for c in contracts:
                self.ib.cancelMktData(c)
            self.ib.sleep(0.5)

        return clean_chain, current_price

    def _calculate_bwb_pop(self, leg_a: OptionLeg, leg_b: OptionLeg, leg_c: OptionLeg, net_credit: float, current_price: float) -> Tuple[float, float, float]:
        breakeven_lower = leg_b.strike - net_credit
        breakeven_upper = current_price * 1.5
        try:
            prob_above_b = 1 + leg_b.delta
            bep_adjustment = net_credit / leg_b.strike
            pop = max(0.5, min(0.95, prob_above_b * (1 + bep_adjustment)))
        except:
            pop = 1 + leg_b.delta if leg_b.delta > -1 else 0.5
        return pop, breakeven_lower, breakeven_upper

    def _calculate_utility_score(self, net_credit: float, max_loss: float, net_theta: float, net_vega: float, pop: float, risk_reward_ratio: float) -> float:
        if max_loss <= 0: return 0
        holding_days = 10
        expected_reward = (net_credit * 100) + (net_theta * 100 * holding_days)
        vega_risk_penalty = 1.0
        
        if net_vega > 0:
            vega_risk = abs(net_vega) * 100 * 0.20
            vega_risk_penalty = max(0.5, 1 - (vega_risk / expected_reward)) if expected_reward > 0 else 0.5
            
        risk_adjusted_return = (expected_reward / max_loss) * vega_risk_penalty
        utility_score = (risk_adjusted_return * 0.40) + (pop * 10 * 0.35) + (risk_reward_ratio * 0.25)
        return utility_score

    def _build_bwb_portfolio(self, leg_a: OptionLeg, leg_b: OptionLeg, leg_c: OptionLeg, current_price: float) -> Optional[Dict]:
        net_credit = (2 * leg_b.mid) - leg_a.mid - leg_c.mid
        width_ab = leg_a.strike - leg_b.strike
        width_bc = leg_b.strike - leg_c.strike

        max_loss = (width_bc - width_ab) * 100 - (net_credit * 100)
        if max_loss <= 0 or net_credit < 0.15: return None
        
        max_profit = net_credit * 100

        net_delta = -leg_a.delta + 2 * (-leg_b.delta) - leg_c.delta
        net_theta = leg_a.theta - 2 * leg_b.theta + leg_c.theta
        net_vega = leg_a.vega - 2 * leg_b.vega + leg_c.vega
        net_gamma = leg_a.gamma - 2 * leg_b.gamma + leg_c.gamma

        risk_reward_ratio = max_profit / max_loss
        pop, bep_lower, bep_upper = self._calculate_bwb_pop(leg_a, leg_b, leg_c, net_credit, current_price)
        utility_score = self._calculate_utility_score(net_credit, max_loss, net_theta, net_vega, pop, risk_reward_ratio)

        return {
            'legs': [
                {'leg_role': 'BWB_Upper_Wing', 'action': 'long', 'data': leg_a, 'qty': 1},
                {'leg_role': 'BWB_Body_Short', 'action': 'short', 'data': leg_b, 'qty': 2},
                {'leg_role': 'BWB_Lower_Wing', 'action': 'long', 'data': leg_c, 'qty': 1}
            ],
            'net_credit': net_credit,
            'max_loss': max_loss,
            'max_profit': max_profit,
            'net_theta': net_theta,
            'net_vega': net_vega,
            'pop': pop,
            'risk_reward_ratio': risk_reward_ratio,
            'utility_score': utility_score
        }

    def scan_best_bwb_structure(self, symbol: str, target_days: int, max_loss_limit: float = 2500) -> Optional[Dict]:
        self.log_action(f"   ⚙️ 启动高级组合寻优 -> 目标: [PUT_BWB] | 标的: {symbol}")
        
        clean_chain, current_price = self._get_clean_option_chain(symbol, 'P', target_days, price_range_pct=0.15)
        
        if len(clean_chain) < 5:
            self.log_action("   ❌ 清洗后合格期权腿不足，无法构建 BWB。")
            return None
            
        clean_chain.sort(key=lambda leg: leg.strike, reverse=True)
        
        best_portfolio = None
        highest_score = -9999.0
        valid_portfolios_count = 0
        
        n = len(clean_chain)
        for i_a in range(n - 2):
            leg_a = clean_chain[i_a]
            if leg_a.delta < -0.45 or leg_a.delta > -0.20: continue
                
            for i_b in range(i_a + 1, n - 1):
                leg_b = clean_chain[i_b]
                if leg_b.delta < -0.25 or leg_b.delta > -0.05: continue
                    
                width_ab = leg_a.strike - leg_b.strike
                
                for i_c in range(i_b + 1, n):
                    leg_c = clean_chain[i_c]
                    width_bc = leg_b.strike - leg_c.strike
                    
                    if width_bc < width_ab: continue
                        
                    portfolio = self._build_bwb_portfolio(leg_a, leg_b, leg_c, current_price)
                    if not portfolio: continue
                    if portfolio['max_loss'] > max_loss_limit: continue
                    
                    valid_portfolios_count += 1
                    
                    if portfolio['utility_score'] > highest_score:
                        highest_score = portfolio['utility_score']
                        best_portfolio = portfolio

        if best_portfolio:
            self.log_action(f"   🏆 寻优完成! 共扫描 {valid_portfolios_count} 种有效组合。最高效用得分: {highest_score:.2f}")
            leg_A = best_portfolio['legs'][0]['data']
            leg_B = best_portfolio['legs'][1]['data']
            leg_C = best_portfolio['legs'][2]['data']
            
            self.log_action(f"      [买] {leg_A.expiry} ${leg_A.strike}P (Δ:{leg_A.delta:.2f})")
            self.log_action(f"      [卖 x 2] {leg_B.expiry} ${leg_B.strike}P (Δ:{leg_B.delta:.2f})")
            self.log_action(f"      [买] {leg_C.expiry} ${leg_C.strike}P (Δ:{leg_C.delta:.2f})")
            self.log_action(f"      💰 净收权利金: ${best_portfolio['net_credit'] * 100:.2f}")
            self.log_action(f"      🛡️ 极限最大回撤: ${best_portfolio['max_loss']:.2f} | 胜率估算: {best_portfolio['pop'] * 100:.1f}%")
            self.log_action(f"      ⚖️ R/R 比率: {best_portfolio['risk_reward_ratio']:.2f} | 净 Vega: {best_portfolio['net_vega']:.3f}")
            return best_portfolio
        else:
            self.log_action("   ⚠️ 遍历结束。当前期权链无满足风控与盈利双重标准的结构组合。")
            return None

    # ================= 数据库操作与其他组件 =================
    
    def load_pre_market_data(self, ticker):
        query = self.supabase.table('stock_options_daily').select('*').eq('ticker', ticker).order('date', desc=True).limit(1)
        res = self.execute_db_with_retry(query, "加载盘前雷达数据")
        if res and res.data:
            return res.data[0]
        return None

    def load_pre_market_data_for_date(self, ticker, signal_date):
        query = (
            self.supabase.table('stock_options_daily')
            .select('*')
            .eq('ticker', ticker)
            .eq('date', signal_date.isoformat())
            .limit(1)
        )
        res = self.execute_db_with_retry(query, "加载QQQ信号日数据")
        if res and res.data:
            return res.data[0]
        return None

    def load_qqq_post_close_context(self, signal_date):
        query = (
            self.supabase.table('stock_spot_post_close')
            .select('*')
            .eq('ticker', 'QQQ')
            .eq('date', signal_date.isoformat())
            .limit(1)
        )
        res = self.execute_db_with_retry(query, "加载QQQ盘后信号数据")
        return res.data[0] if res and res.data else None

    def load_qqq_monthly_wall_context(self, entry_date):
        query = (
            self.supabase.table('stock_options_pre_market')
            .select('*')
            .eq('ticker', 'QQQ')
            .eq('date', entry_date.isoformat())
            .limit(1)
        )
        res = self.execute_db_with_retry(query, "加载QQQ月权墙数据")
        return res.data[0] if res and res.data else None

    @staticmethod
    def _finite_float(value):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    @staticmethod
    def _extract_five_factor_regime(row):
        if not row:
            return None
        # Only these columns use the frozen six-state vocabulary. Generic regime
        # columns may contain unrelated labels such as HIGH/elevated.
        for key in ('five_factor_regime', 'joint_regime'):
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    @staticmethod
    def _parse_expiry(value):
        if value is None or not str(value).strip():
            return None
        raw = str(value).strip()
        for date_format in ('%Y-%m-%d', '%Y%m%d'):
            try:
                return datetime.datetime.strptime(raw[:10], date_format).date()
            except ValueError:
                continue
        return None

    @classmethod
    def _extract_wall_expiry(cls, row):
        if not row:
            return None
        for key in ('monthly_wall_expiry', 'wall_expiry', 'option_wall_expiry'):
            parsed = cls._parse_expiry(row.get(key))
            if parsed:
                return parsed
        put_expiry = cls._parse_expiry(row.get('put_wall_expiry'))
        call_expiry = cls._parse_expiry(row.get('call_wall_expiry'))
        if put_expiry and put_expiry == call_expiry:
            return put_expiry
        return None

    @classmethod
    def _extract_wall_source_date(cls, row):
        if not row:
            return None
        for key in ('monthly_wall_oi_source_date', 'wall_oi_source_date'):
            parsed = cls._parse_expiry(row.get(key))
            if parsed:
                return parsed
        return None

    @staticmethod
    def _session_market_close(session_date, calendar=None):
        nyse = calendar or mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=session_date, end_date=session_date)
        if schedule.empty:
            return None
        market_close = schedule.iloc[0]['market_close']
        if hasattr(market_close, 'to_pydatetime'):
            market_close = market_close.to_pydatetime()
        if market_close.tzinfo is None:
            market_close = market_close.replace(tzinfo=timezone.utc)
        return market_close.astimezone(NY_TZ)

    def _load_completed_qqq_bars(self, history=None, now=None, calendar=None):
        if history is None:
            history = yf.Ticker('QQQ').history(
                period='3mo',
                interval='1d',
                auto_adjust=False,
            )
        required_columns = {'Open', 'High', 'Low', 'Close'}
        if history.empty or not required_columns.issubset(history.columns):
            raise RuntimeError("yfinance没有返回QQQ完整OHLC日线")

        bars = history[['Open', 'High', 'Low', 'Close']].copy()
        bars.index = [
            index.date() if hasattr(index, 'date') else index for index in bars.index
        ]
        bars = bars[~bars.index.duplicated(keep='last')].sort_index()

        now = now or datetime.datetime.now(NY_TZ)
        market_close = self._session_market_close(now.date(), calendar=calendar)
        publication_cutoff = (
            market_close + datetime.timedelta(minutes=10) if market_close else None
        )
        # Use the NYSE's actual close, including 13:00 ET half days. Keep a
        # ten-minute publication buffer before accepting Yahoo's current bar.
        if now.date() in bars.index and (
            publication_cutoff is None or now < publication_cutoff
        ):
            bars = bars.drop(index=now.date())
        bars = bars.dropna(subset=['Open', 'High', 'Low', 'Close'])
        if len(bars) < 2:
            raise RuntimeError("QQQ已完成日线不足两天，无法计算信号")
        return bars.tail(2)

    @staticmethod
    def _next_market_day(signal_date, calendar=None):
        nyse = calendar or mcal.get_calendar('NYSE')
        schedule = nyse.schedule(
            start_date=signal_date + datetime.timedelta(days=1),
            end_date=signal_date + datetime.timedelta(days=14),
        )
        if schedule.empty:
            raise RuntimeError("无法确定QQQ信号的T+1交易日")
        return schedule.index[0].date()

    @staticmethod
    def _third_friday(year, month):
        first = datetime.date(year, month, 1)
        first_friday_offset = (4 - first.weekday()) % 7
        return first + datetime.timedelta(days=first_friday_offset + 14)

    def _next_standard_monthly_expiry_window(self, signal_date, calendar=None):
        nyse = calendar or mcal.get_calendar('NYSE')
        year, month = signal_date.year, signal_date.month
        for month_offset in range(4):
            candidate_month = month + month_offset
            candidate_year = year + (candidate_month - 1) // 12
            candidate_month = (candidate_month - 1) % 12 + 1
            third_friday = self._third_friday(candidate_year, candidate_month)
            expiry_schedule = nyse.schedule(
                start_date=third_friday - datetime.timedelta(days=7),
                end_date=third_friday,
            )
            if expiry_schedule.empty:
                continue
            settlement_date = expiry_schedule.index[-1].date()
            if settlement_date <= signal_date:
                continue
            remaining = nyse.schedule(
                start_date=signal_date + datetime.timedelta(days=1),
                end_date=settlement_date,
            )
            # Frozen wall research defines the signal at T when there are six
            # sessions strictly after T through settlement; T+1 then leaves five.
            return settlement_date, len(remaining)
        raise RuntimeError("无法确定下一标准月权到期窗口")

    def _entry_stale_reason(self, entry_date, now=None, calendar=None):
        now = now or datetime.datetime.now(NY_TZ)
        if entry_date < now.date():
            return f"T+1交易日{entry_date.isoformat()}早于当前日期{now.date().isoformat()}"
        if entry_date > now.date():
            return None
        market_close = self._session_market_close(entry_date, calendar=calendar)
        if market_close is not None and now >= market_close:
            return f"T+1交易日{entry_date.isoformat()}已于{market_close.strftime('%H:%M %Z')}收盘"
        return None

    @staticmethod
    def _nearest_option_strikes(all_strikes, centers, count_per_center=21):
        valid = sorted(
            {
                float(strike)
                for strike in all_strikes
                if UnifiedLedgerBuilder._finite_float(strike) is not None
                and float(strike) > 0
            }
        )
        selected = set()
        for center in centers:
            selected.update(
                sorted(valid, key=lambda strike: abs(strike - center))[
                    :count_per_center
                ]
            )
        return sorted(selected)

    @staticmethod
    def _first_nonnegative_int(*values):
        for value in values:
            parsed = UnifiedLedgerBuilder._finite_float(value)
            if parsed is not None and parsed >= 0:
                return int(parsed)
        return None

    @staticmethod
    def _closest_option_expiry(expirations, target_days, today=None):
        today = today or datetime.datetime.now(NY_TZ).date()
        future = []
        for raw in expirations:
            try:
                expiry = datetime.datetime.strptime(str(raw), '%Y%m%d').date()
            except ValueError:
                continue
            dte = (expiry - today).days
            if dte > 0:
                future.append((abs(dte - target_days), dte, expiry))
        if not future:
            return None
        return min(future)[2].strftime('%Y%m%d')

    def _resolve_qqq_option_chain(self, target_days=45, exact_expiry=None):
        stock = Stock('QQQ', 'SMART', 'USD')
        self.ib.qualifyContracts(stock)
        chains = self.ib.reqSecDefOptParams(
            stock.symbol, '', stock.secType, stock.conId
        )
        if not chains:
            raise RuntimeError("IBKR未返回QQQ期权链定义")
        chain = next((item for item in chains if item.exchange == 'SMART'), chains[0])
        if exact_expiry is not None:
            requested = exact_expiry.strftime('%Y%m%d')
            expiry = requested if requested in chain.expirations else None
        else:
            # Frozen QQQ candidates are parameterized by DTE, not by monthly
            # expiry preference. A 44D weekly is therefore a better 45D match
            # than a 58D standard monthly. Wall trades pass exact_expiry above.
            expiry = self._closest_option_expiry(chain.expirations, target_days)
        if not expiry:
            requested_text = exact_expiry.isoformat() if exact_expiry else f"约{target_days}DTE"
            raise RuntimeError(f"IBKR链中没有{requested_text}的可用到期日")
        return chain, expiry

    def _request_qqq_option_quotes(self, expiry, strikes, rights):
        quotes = []
        contracts = [
            Option('QQQ', expiry, strike, right, 'SMART')
            for right in rights
            for strike in strikes
        ]
        batch_size = 16
        expiry_iso = datetime.datetime.strptime(expiry, '%Y%m%d').date().isoformat()
        for start in range(0, len(contracts), batch_size):
            batch = contracts[start:start + batch_size]
            try:
                qualified = self.ib.qualifyContracts(*batch)
            except Exception as exc:
                self.logger.warning(f"QQQ候选合约资格校验失败: {exc}")
                continue
            qualified = [contract for contract in qualified if contract.conId]
            if not qualified:
                continue
            tickers = [
                self.ib.reqMktData(contract, '100,101,106', False, False)
                for contract in qualified
            ]
            try:
                for _ in range(35):
                    self.ib.sleep(0.1)
                    ready = sum(
                        1
                        for ticker in tickers
                        if ticker.modelGreeks
                        and self._finite_float(ticker.modelGreeks.delta) is not None
                        and self._finite_float(ticker.bid) is not None
                        and self._finite_float(ticker.ask) is not None
                    )
                    if ready >= max(1, int(len(tickers) * 0.75)):
                        break
                self.ib.sleep(0.25)
                for ticker in tickers:
                    contract = ticker.contract
                    greeks = ticker.modelGreeks
                    delta = self._finite_float(
                        greeks.delta if greeks is not None else None
                    )
                    if delta is None:
                        continue
                    bid = self._finite_float(ticker.bid)
                    ask = self._finite_float(ticker.ask)
                    right = str(contract.right).upper()
                    side_oi = getattr(
                        ticker,
                        'putOpenInterest' if right == 'P' else 'callOpenInterest',
                        None,
                    )
                    side_volume = getattr(
                        ticker,
                        'putVolume' if right == 'P' else 'callVolume',
                        None,
                    )
                    quotes.append(
                        LiveOptionQuote(
                            underlying='QQQ',
                            expiry=expiry_iso,
                            right=right,
                            strike=float(contract.strike),
                            bid=bid if bid is not None else math.nan,
                            ask=ask if ask is not None else math.nan,
                            delta=delta,
                            iv=self._finite_float(
                                greeks.impliedVol if greeks is not None else None
                            ),
                            volume=self._first_nonnegative_int(
                                getattr(ticker, 'volume', None), side_volume
                            ),
                            open_interest=self._first_nonnegative_int(
                                getattr(ticker, 'openInterest', None), side_oi
                            ),
                            market_data_type=self._first_nonnegative_int(
                                getattr(ticker, 'marketDataType', None)
                            ),
                            con_id=self._first_nonnegative_int(contract.conId),
                            local_symbol=(
                                str(contract.localSymbol).strip()
                                if getattr(contract, 'localSymbol', None)
                                else None
                            ),
                        )
                    )
            finally:
                for contract in qualified:
                    self.ib.cancelMktData(contract)
        return quotes

    def _load_qqq_candidate_quotes(
        self,
        spot,
        target_days=45,
        exact_expiry=None,
        centers=None,
        rights=('P', 'C'),
        count_per_center=21,
    ):
        chain, expiry = self._resolve_qqq_option_chain(
            target_days=target_days,
            exact_expiry=exact_expiry,
        )
        centers = list(centers or [spot])
        strikes = self._nearest_option_strikes(
            chain.strikes, centers, count_per_center=count_per_center
        )
        if not strikes:
            raise RuntimeError("IBKR期权链没有候选执行价")
        return self._request_qqq_option_quotes(expiry, strikes, rights), expiry

    def scan_qqq_research_signals(self):
        self.log_action("\n[QQQ 冻结策略信号扫描]")
        try:
            now = datetime.datetime.now(NY_TZ)
            bars = self._load_completed_qqq_bars(now=now)
            previous_bar = bars.iloc[-2]
            signal_bar = bars.iloc[-1]
            signal_date = bars.index[-1]
            entry_date = self._next_market_day(signal_date)
            wall_expiry, wall_sessions = self._next_standard_monthly_expiry_window(signal_date)

            signal_row = self.load_qqq_post_close_context(signal_date)
            if signal_row is None:
                self.log_action(
                    f"  [WARN] {signal_date.isoformat()}缺少stock_spot_post_close记录；"
                    "五因子gate按缺失处理。"
                )
                signal_row = {}
            wall_row = self.load_qqq_monthly_wall_context(entry_date)
            if wall_row is None:
                self.log_action(
                    f"  [WARN] {entry_date.isoformat()}缺少stock_options_pre_market记录；"
                    "标准月权墙按缺失处理。"
                )
                wall_row = {}

            context = QQQSignalContext(
                signal_date=signal_date,
                entry_date=entry_date,
                prior_close=float(previous_bar['Close']),
                open=float(signal_bar['Open']),
                high=float(signal_bar['High']),
                low=float(signal_bar['Low']),
                close=float(signal_bar['Close']),
                put_wall=self._finite_float(wall_row.get('monthly_put_wall')),
                call_wall=self._finite_float(wall_row.get('monthly_call_wall')),
                wall_expiry=wall_expiry,
                wall_data_expiry=self._extract_wall_expiry(wall_row),
                wall_data_source_date=self._extract_wall_source_date(wall_row),
                wall_sessions_to_expiry=wall_sessions,
                five_factor_regime=self._extract_five_factor_regime(signal_row),
                iv_rank=self._finite_float(
                    signal_row.get('iv_rank_pct', signal_row.get('ivr_pct'))),
                skew_25d=None,
                entry_stale_reason=self._entry_stale_reason(entry_date, now=now),
            )
            signals = evaluate_qqq_signals(context)
            self.qqq_signal_context = context
            self.qqq_strategy_signals = signals
            section = format_signal_email_section(context, signals)
            for line in section.splitlines():
                self.log_action(line)
        except Exception as exc:
            # 研究信号绝不能影响原有对账、维护和建仓路径。
            self.log_action(f"  [WARN] QQQ研究信号扫描失败，原交易流程继续: {exc}")

    def scan_qqq_option_contract_candidates(self):
        """Enrich current QQQ signals with read-only IBKR option candidates."""
        self.log_action("\n[QQQ IBKR 合约候选扫描]")
        try:
            context = self.qqq_signal_context
            signals = tuple(self.qqq_strategy_signals or ())
            if context is None:
                self.log_action("  [WARN] 缺少QQQ信号上下文，无法扫描具体合约。")
                return
            active = [
                signal
                for signal in signals
                if signal.status in {'TRIGGERED', 'WATCH'}
            ]
            if not active:
                self.log_action(format_contract_scan_section([]))
                return
            now = datetime.datetime.now(NY_TZ)
            if context.entry_date != now.date():
                self.log_action(
                    f"  [WARN] 当前{now.date().isoformat()}不是信号T+1入场日"
                    f"{context.entry_date.isoformat()}，不展示可能误导的实时合约。"
                )
                return
            if not self.ib.isConnected():
                self.log_action("  [WARN] IBKR未连接，保留策略文字但无法取得逐腿报价。")
                return

            spot = self.get_stock_price('QQQ')
            if self._finite_float(spot) is None or spot <= 0:
                raise RuntimeError("IBKR未返回有效QQQ现价")
            entry_cost = float(
                getattr(cfg, 'QQQ_OPTION_ENTRY_COST_PER_CONTRACT', 0.67)
            )
            results = []
            common = [
                signal
                for signal in active
                if signal.code in {
                    'uptrend_funded_call',
                    'downtrend_defined_risk_combo',
                }
            ]
            if common:
                centers = [spot]
                if any(
                    signal.code == 'downtrend_defined_risk_combo'
                    for signal in common
                ):
                    centers.append(spot * 0.90)
                quotes, _ = self._load_qqq_candidate_quotes(
                    spot,
                    target_days=45,
                    centers=centers,
                    rights=('P', 'C'),
                )
                puts = [quote for quote in quotes if quote.right == 'P']
                calls = [quote for quote in quotes if quote.right == 'C']
                for signal in common:
                    if signal.code == 'uptrend_funded_call':
                        results.append(
                            build_uptrend_funded_call_candidate(
                                puts,
                                calls,
                                spot,
                                now,
                                signal_status=signal.status,
                                entry_cost_per_contract=entry_cost,
                            )
                        )
                    else:
                        results.append(
                            build_downtrend_combo_candidate(
                                puts,
                                calls,
                                spot,
                                now,
                                signal_status=signal.status,
                                entry_cost_per_contract=entry_cost,
                            )
                        )

            for signal in active:
                if signal.code != 'monthly_wall_break_debit_spread':
                    continue
                call_break = (
                    context.call_wall is not None
                    and context.close > context.call_wall
                )
                right = 'C' if call_break else 'P'
                wall = context.call_wall if call_break else context.put_wall
                if wall is None or context.wall_expiry is None:
                    continue
                outer = (
                    wall + spot * 0.05 if right == 'C' else wall - spot * 0.05
                )
                quotes, expiry = self._load_qqq_candidate_quotes(
                    spot,
                    exact_expiry=context.wall_expiry,
                    centers=[wall, outer],
                    rights=(right,),
                    count_per_center=7,
                )
                results.append(
                    build_wall_debit_spread_candidate(
                        quotes,
                        spot=spot,
                        quote_time=now,
                        expiry=datetime.datetime.strptime(
                            expiry, '%Y%m%d'
                        ).date().isoformat(),
                        right=right,
                        wall_strike=wall,
                        signal_status=signal.status,
                        entry_cost_per_contract=entry_cost,
                    )
                )

            section = format_contract_scan_section(results)
            for line in section.splitlines():
                self.log_action(line)
        except Exception as exc:
            # Quote enrichment is informational and must never block the ledger.
            self.log_action(
                f"  [WARN] QQQ具体合约扫描失败，策略文字仍保留，原交易流程继续: {exc}"
            )

    def get_latest_macro_score(self):
        query = self.supabase.table('market_history').select('micro_score').order('record_date', desc=True).limit(1)
        res = self.execute_db_with_retry(query, "获取宏观V8.6评分")
        if res and res.data:
            return float(res.data[0]['micro_score'])
        return None

    # ================= 🚀 新增：实盘自动对账引擎 (确保开仓前容量核算准确) =================
    def reconcile_executions(self):
        self.log_action("\n[🔄 启动实盘成交对账引擎 (Reconciliation)...]")
        try:
            fills = self.ib.reqExecutions()
            if not fills:
                self.log_action("  -> 今日无实盘成交记录，跳过对账。")
                return

            fill_dict = {}
            for fill in fills:
                c = fill.contract
                exec_ = fill.execution
                comm_report = fill.commissionReport
                
                exp = c.lastTradeDateOrContractMonth
                if not exp: continue
                expiry = exp[:8] if len(exp) >= 8 else exp
                    
                key = (c.symbol, expiry, float(c.strike), c.right, exec_.side)
                
                if key not in fill_dict:
                    fill_dict[key] = {'shares': 0.0, 'total_cost': 0.0, 'commission': 0.0, 'conid': c.conId}
                    
                fill_dict[key]['shares'] += float(exec_.shares)
                fill_dict[key]['total_cost'] += float(exec_.shares) * float(exec_.price)
                if comm_report and comm_report.commission:
                    fill_dict[key]['commission'] += float(comm_report.commission)

            for k, v in fill_dict.items():
                if v['shares'] > 0:
                    v['vwap'] = v['total_cost'] / v['shares']

            query = self.supabase.table('unified_active_legs').select('*').in_('status', ['ACTIVE', 'ROLL_REQUESTED'])
            res = self.execute_db_with_retry(query, "读取待对账记录")
            if not res or not res.data: return
            
            for leg in res.data:
                sym, exp, stk, right = leg['symbol'], leg['expiry'], float(leg['strike']), leg['right_type']
                pos_type = leg['position_type']
                
                entry_action = 'BOT' if pos_type == 'long' else 'SLD'
                close_action = 'SLD' if pos_type == 'long' else 'BOT'
                
                entry_key = (sym, exp, stk, right, entry_action)
                close_key = (sym, exp, stk, right, close_action)
                
                updates = {}
                
                if entry_key in fill_dict and leg.get('real_entry_price') is None:
                    f_data = fill_dict[entry_key]
                    updates['real_entry_price'] = round(f_data['vwap'], 4)
                    updates['commission'] = round(f_data['commission'], 4)
                    updates['ib_conid'] = f_data['conid']
                    self.log_action(f"  -> [对账成功] 更新开仓真实成本: {sym} ${stk}{right} | 均价 ${updates['real_entry_price']}")
                    
                if leg['status'] == 'ROLL_REQUESTED' and close_key in fill_dict:
                    f_data = fill_dict[close_key]
                    updates['real_close_price'] = round(f_data['vwap'], 4)
                    old_comm = float(leg.get('commission') or 0.0)
                    updates['commission'] = round(old_comm + f_data['commission'], 4)
                    updates['status'] = 'ROLLED'
                    self.log_action(f"  -> [平仓对账] 确认展期/平仓完成: {sym} ${stk}{right} | 平仓价 ${updates['real_close_price']}")
                    
                if updates:
                    update_query = self.supabase.table('unified_active_legs').update(updates).eq('id', leg['id'])
                    self.execute_db_with_retry(update_query, "对账写入库")

        except Exception as e:
            self.log_action(f"⚠️ 自动对账异常: {e}")

    def execute_maintenance(self):
        self.log_action("\n[阶段一] 局部展期手术 (Maintenance)")
        query = self.supabase.table('unified_active_legs').select('*').eq('status', 'ROLL_REQUESTED')
        res = self.execute_db_with_retry(query, "扫描待展期病号")
        
        if not res or not res.data:
            self.log_action("🟢 账本内所有腿运行健康，无需展期干预。")
            return

        for leg in res.data:
            leg_id = leg['id']
            leg_role = leg['leg_role']
            self.log_action(f"💉 正在对 [{leg_role}] 执行展期手术...")
            
            new_leg_data = None
            qqq_ctx = self.load_pre_market_data('QQQ')
            
            if leg_role == 'Income_Put':
                put_wall = qqq_ctx.get('put_wall') if qqq_ctx else None
                new_leg_data = self.find_optimal_leg('QQQ', 'P', leg['target_days'], explicit_strike=put_wall) if put_wall else self.find_optimal_leg('QQQ', 'P', leg['target_days'], target_otm_pct=-0.03)
            elif leg_role == 'Tail_Hedge':
                new_leg_data = self.find_optimal_leg('QQQ', 'P', leg['target_days'], target_otm_pct=-0.15)
            else:
                self.log_action(f"⚠️ [{leg_role}] 不属于标准收租/保险序列，跳过自动展期。")
                continue

            if new_leg_data:
                payload = {
                    "leg_role": leg_role,
                    "status": "ACTIVE",
                    "symbol": new_leg_data['symbol'],
                    "right_type": new_leg_data['right'],
                    "strike": new_leg_data['strike'],
                    "expiry": new_leg_data['expiry'],
                    "position_type": leg['position_type'],
                    "entry_mid": new_leg_data['mid'],
                    "target_days": leg['target_days']
                }
                
                insert_query = self.supabase.table('unified_active_legs').insert(payload)
                insert_res = self.execute_db_with_retry(insert_query, "写入新展期阵地")
                
                if insert_res and insert_res.data:
                    update_query = self.supabase.table('unified_active_legs').update({'status': 'ROLLED', 'close_mid': leg['entry_mid']}).eq('id', leg_id)
                    self.execute_db_with_retry(update_query, "安全关闭旧阵地")
                    self.log_action(f"✅ 手术成功。置换新兵种: {new_leg_data['symbol']} {new_leg_data['expiry']} ${new_leg_data['strike']}{new_leg_data['right']}")
                else:
                    self.log_action(f"❌ 新腿数据库插入失败，展期中止")
            else:
                self.log_action(f"❌ 寻优获取新腿 [{leg_role}] 失败，请人工检修。")

    def execute_laddering(self):
        self.log_action("\n[阶段二] 宏观容量管理与智能添腿 (Laddering)")
        
        macro_score = self.get_latest_macro_score()
        
        if macro_score is None:
            self.log_action("⚠️ 无法从 market_history 获取 V8.6 宏观分数！系统进入【静默保护模式】，今日中止建仓。")
            return
            
        self.log_action(f"[*] 慢脑雷达 V8.6 昨收盘得分: {macro_score:.1f} 分")

        if macro_score >= 80:
            self.log_action("🚨 宏观系统性危机确认 (得分>=80)！停止所有卖出建仓动作，执行中枢强制下线。")
            return
            
        elif 50 <= macro_score < 80:
            self.log_action("🚸 宏观高度警戒 (得分 50-79)。资金上限强制减半，禁止部署 TQQQ 增强腿。")
            current_margin_limit_pct = MARGIN_LIMIT_PCT / 2
            allow_enhance = False
        else:
            self.log_action("🟢 宏观微观结构安全 (得分 <50)。授权全量额度与 TQQQ 右侧增强打击。")
            current_margin_limit_pct = MARGIN_LIMIT_PCT
            allow_enhance = True

        actual_net_liquidation = ACCOUNT_BASE 
        margin_limit_usd = actual_net_liquidation * current_margin_limit_pct
        
        query = self.supabase.table('unified_active_legs').select('*').eq('status', 'ACTIVE').order('created_at', desc=True)
        res = self.execute_db_with_retry(query, "审查现役资金容量")
        
        if res is None:
            return
            
        active_legs = res.data
        
        total_margin_est = 0.0
        for leg in active_legs:
            # 🚀 资金容量算法升级：优先提取真实物理成本
            real_entry = leg.get('real_entry_price')
            cost_basis = float(real_entry) if real_entry is not None else float(leg['entry_mid'])
            
            if leg['position_type'] == 'short':
                total_margin_est += float(leg['strike']) * 100 * 0.15
            else:
                total_margin_est += cost_basis * 100

        self.log_action(f"[*] 账本容量审查: 已用 ${total_margin_est:,.2f} / 当前动态上限 ${margin_limit_usd:,.2f}")

        if total_margin_est >= margin_limit_usd:
            self.log_action("🛑 保证金池安全水位已达上限。添腿程序中止。")
            return

        if active_legs:
            latest_leg_date = max([datetime.datetime.fromisoformat(leg['created_at']) for leg in active_legs])
            days_since_last = (datetime.datetime.now(timezone.utc) - latest_leg_date).days
            if days_since_last < COOLDOWN_DAYS:
                self.log_action(f"🛑 账本处于加仓冷静期 (仅过 {days_since_last} 天)。添腿程序中止。")
                return

        self.log_action("\n🟢 资金与时间许可，进入战术分配与路由...")
        
        current_vix = self.get_vix()
        qqq_price = self.get_stock_price('QQQ')
        tqqq_price = self.get_stock_price('TQQQ')
        
        qqq_ctx = self.load_pre_market_data('QQQ')
        put_wall = qqq_ctx.get('put_wall') if qqq_ctx else None
        
        raw_skew = qqq_ctx.get('skew_25d') if qqq_ctx else 0
        skew = float(raw_skew) if raw_skew is not None else 0.0
        
        self.log_action(f"  -> QQQ现价: ${qqq_price:.2f} | TQQQ现价: ${tqqq_price:.2f}")
        self.log_action(f"  -> 实时VIX: {current_vix:.2f} | 25D偏度: {skew}% | Put墙: ${put_wall}")

        legs_to_insert = []
        net_income = 0

        # ================= 智能战术路由 =================
        if current_vix >= 25 or skew > 10.0:
            self.log_action(f"🔥 侦测到高波/恐慌环境：尝试启动 [BWB 破翼蝶式] 收割散户溢价。")
            bwb_portfolio = self.scan_best_bwb_structure('QQQ', target_days=25, max_loss_limit=1500)
            
            if bwb_portfolio:
                for item in bwb_portfolio['legs']:
                    leg_data = item['data']
                    for _ in range(item['qty']):
                        legs_to_insert.append({
                            "leg_role": item['leg_role'],
                            "status": "ACTIVE",
                            "symbol": leg_data.symbol,
                            "right_type": leg_data.right,
                            "strike": leg_data.strike,
                            "expiry": leg_data.expiry,
                            "position_type": item['action'],
                            "entry_mid": leg_data.mid,
                            "target_days": 25
                        })
                net_income = bwb_portfolio['net_credit']
                allow_enhance = False
            else:
                self.log_action("❌ BWB 寻优无结果，退回常规高波防御收租模式 (Put Credit Spread)。")
                income_short = self.find_optimal_leg('QQQ', 'P', 25, target_otm_pct=-0.08)
                income_long = self.find_optimal_leg('QQQ', 'P', 25, target_otm_pct=-0.11)
                
                if not income_short or not income_long:
                    self.log_action("❌ 无法获取常规防御组合合约，添腿流产。")
                    return
                    
                legs_to_insert.append({"leg_role": "Income_Put", "status": "ACTIVE", "symbol": income_short['symbol'], "right_type": income_short['right'], "strike": income_short['strike'], "expiry": income_short['expiry'], "position_type": "short", "entry_mid": income_short['mid'], "target_days": 25})
                legs_to_insert.append({"leg_role": "Income_Put_Protection", "status": "ACTIVE", "symbol": income_long['symbol'], "right_type": income_long['right'], "strike": income_long['strike'], "expiry": income_long['expiry'], "position_type": "long", "entry_mid": income_long['mid'], "target_days": 25})
                
                net_income = income_short['mid'] - income_long['mid']
                allow_enhance = False  
                
        else:
            self.log_action("⚖️ 市场处于常态震荡环境：执行标准梯队 (卖裸Put + 买远期保险)。")
            income_leg = self.find_optimal_leg('QQQ', 'P', 25, explicit_strike=put_wall) if put_wall else self.find_optimal_leg('QQQ', 'P', 25, target_otm_pct=-0.03)
            hedge_leg = self.find_optimal_leg('QQQ', 'P', 120, target_otm_pct=-0.12)
            
            if not income_leg or not hedge_leg:
                self.log_action("❌ 无法获取基础收租/保险合约，添腿流产。")
                return
                
            legs_to_insert.append({"leg_role": "Income_Put", "status": "ACTIVE", "symbol": income_leg['symbol'], "right_type": income_leg['right'], "strike": income_leg['strike'], "expiry": income_leg['expiry'], "position_type": "short", "entry_mid": income_leg['mid'], "target_days": 25})
            legs_to_insert.append({"leg_role": "Tail_Hedge", "status": "ACTIVE", "symbol": hedge_leg['symbol'], "right_type": hedge_leg['right'], "strike": hedge_leg['strike'], "expiry": hedge_leg['expiry'], "position_type": "long", "entry_mid": hedge_leg['mid'], "target_days": 120})
            
            net_income = income_leg['mid'] - hedge_leg['mid']

        # ================= TQQQ 右侧增强扫描 =================
        if allow_enhance and net_income > 0:
            enhance_budget = net_income * 0.30  
            self.log_action(f"\n🎯 启动右侧增强扫描 (可用免费预算: ${enhance_budget * 100:.2f})...")
            
            if current_vix < 15:
                self.log_action("   🌊 极低波区间：尝试部署 TQQQ 裸买看涨 (Naked Call) 抓取极致凸性。")
                enhance_long = self.find_optimal_leg('TQQQ', 'C', 75, target_otm_pct=0.05)
                
                if enhance_long and enhance_long['mid'] <= enhance_budget:
                    legs_to_insert.append({"leg_role": "Enhance_Call", "status": "ACTIVE", "symbol": enhance_long['symbol'], "right_type": enhance_long['right'], "strike": enhance_long['strike'], "expiry": enhance_long['expiry'], "position_type": "long", "entry_mid": enhance_long['mid'], "target_days": 75})
                else:
                    self.log_action(f"   ⚠️ TQQQ 单腿 Call (预算不够或无数据)，本次放弃右侧增强。")
                    
            else: 
                self.log_action("   ⚖️ 常波区间：尝试部署 TQQQ 看涨价差 (Bull Call Spread) 对冲 Vega。")
                enhance_long = self.find_optimal_leg('TQQQ', 'C', 75, target_otm_pct=0.05)
                enhance_short = self.find_optimal_leg('TQQQ', 'C', 75, target_otm_pct=0.10)
                
                if enhance_long and enhance_short:
                    spread_cost = enhance_long['mid'] - enhance_short['mid']
                    if 0 < spread_cost <= enhance_budget:
                        legs_to_insert.append({"leg_role": "Enhance_Call", "status": "ACTIVE", "symbol": enhance_long['symbol'], "right_type": enhance_long['right'], "strike": enhance_long['strike'], "expiry": enhance_long['expiry'], "position_type": "long", "entry_mid": enhance_long['mid'], "target_days": 75})
                        legs_to_insert.append({"leg_role": "Enhance_Call_Short", "status": "ACTIVE", "symbol": enhance_short['symbol'], "right_type": enhance_short['right'], "strike": enhance_short['strike'], "expiry": enhance_short['expiry'], "position_type": "short", "entry_mid": enhance_short['mid'], "target_days": 75})
                    else:
                        self.log_action(f"   ⚠️ TQQQ 看涨价差 (结构倒挂或预算不足)，本次放弃右侧增强。")
                else:
                    self.log_action("   ⚠️ TQQQ 合约数据缺失，本次放弃右侧增强。")

        # ================= 写入账本 =================
        if legs_to_insert:
            insert_query = self.supabase.table('unified_active_legs').insert(legs_to_insert)
            res = self.execute_db_with_retry(insert_query, "批量部署新阵地")
            if res and res.data:
                self.log_action("\n✅ 成功向统一大账本注水：")
                for leg in legs_to_insert:
                    action = "卖出" if leg['position_type'] == 'short' else "买入"
                    price_action = "收取" if leg['position_type'] == 'short' else "支付"
                    self.log_action(f"   * [{leg['leg_role']}] {action} {leg['symbol']} {leg['expiry']} ${leg['strike']}{leg['right_type']} | {price_action}: ${leg['entry_mid']:.2f}")

    def print_current_inventory(self):
        self.log_action("\n[📊 当前总账本现役兵种一览]")
        query = self.supabase.table('unified_active_legs').select('*').eq('status', 'ACTIVE').order('created_at', desc=True)
        res = self.execute_db_with_retry(query, "获取持仓透视")
        
        if res and res.data:
            for leg in res.data:
                action = "卖出" if leg['position_type'] == 'short' else "买入"
                real_entry = leg.get('real_entry_price')
                cost_basis = float(real_entry) if real_entry is not None else float(leg['entry_mid'])
                tag = "🎯实盘" if real_entry is not None else "📝理论"
                self.log_action(f"  * {action} [{leg['leg_role']}] | {leg['symbol']} {leg['expiry']} ${leg['strike']}{leg['right_type']} | {tag}建仓价: ${cost_basis:.2f}")
        else:
            self.log_action("  ⚪ 统一账本当前为空或获取失败。")

    def is_trading_day(self):
        try:
            today_str = datetime.datetime.now(NY_TZ).strftime('%Y-%m-%d')
            nyse = mcal.get_calendar('NYSE')
            schedule = nyse.schedule(start_date=today_str, end_date=today_str)
            return not schedule.empty
        except Exception as e:
            self.logger.warning(f"日历检测异常: {e}")
            return False

    def run(self):
        if not self.is_trading_day():
            self.log_action("🛑 今天是美股休市日，总司令进入法定休眠。")
            return
        try:
            # This scanner does not depend on IBKR and should still report if
            # the broker connection later fails.
            self.scan_qqq_research_signals()
            self.connect()
            self.scan_qqq_option_contract_candidates()
            # 🚀 在建仓和推演前先对账，确保账面干净！
            self.reconcile_executions()
            
            self.execute_maintenance()
            self.execute_laddering()
            self.print_current_inventory()
        except Exception as e:
            err_msg = traceback.format_exc()
            self.log_action(f"\n[X] 运行发生致命错误: {e}\n{err_msg}")
        finally:
            try:
                date_str = datetime.datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M')
                subject = f"【双脑驱动】统一账本中枢执行战报 ({date_str})"
                
                if any("[X] 运行发生致命错误" in log for log in self.action_logs):
                    subject = "🔴 崩溃警报 | " + subject
                    
                self.send_report(subject, "\n".join(self.action_logs))
            except Exception as e2:
                self.logger.error(f"最终发送邮件也失败了: {e2}")
            
            self.disconnect()

if __name__ == "__main__":
    builder = UnifiedLedgerBuilder(port=4001, client_id=889) 
    builder.run()
