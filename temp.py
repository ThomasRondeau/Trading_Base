from Connections.Trading_App import TradingApp
import time
import threading


"""

def run_live():
    #Exemple de trading live avec risk management.
    from ibkr_wrapper import TradingApp
    
    # 1. Connexion IBKR
    print("Connecting to IBKR...")
    app = TradingApp()
    app.connect("127.0.0.1", 7497, clientId=1)
    
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    
    timeout = 10
    start = time.time()
    while not app.connected and (time.time() - start) < timeout:
        time.sleep(0.5)
    
    if not app.connected:
        print("Failed to connect to IBKR")
        return
    
    print("Connected!\n")
    
    # 2. Créer le risk manager
    risk_manager = RiskManager(
        initial_capital=100000,
        max_position_size_pct=0.1,  # Max 10% par position
        max_risk_per_trade_pct=0.02,  # Max 2% de risque par trade
        stop_loss_pct=0.05  # Stop loss à 5%
    )
    
    # 3. Créer la stratégie
    strategy = SimpleMovingAverageStrategy(short_window=20, long_window=50)
    
    # 4. Créer le runner avec le risk manager
    runner = StrategyRunner(app, strategy)
    runner.configure(symbols=['AAPL', 'MSFT'], bar_size=5)
    
    # Injecter le risk manager dans le runner (vous devrez adapter le code)
    # runner.risk_manager = risk_manager
    
    # 5. Démarrer
    runner.run()
    
    # 6. Laisser tourner
    try:
        while True:
            time.sleep(30)
            stats = risk_manager.get_statistics()
            print(f"\n--- Risk Manager Stats ---")
            print(f"Capital: ${stats['current_capital']:,.2f}")
            print(f"P&L: ${stats['total_pnl']:+,.2f} ({stats['return_pct']:+.2f}%)")
            print(f"Trades: {stats['total_trades']} (Win rate: {stats['win_rate']:.1f}%)")
            print("------------------------\n")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        runner.stop()
        app.disconnect()
        print("Done!")


# ============= EXEMPLE D'UTILISATION BACKTEST =============

def run_backtest():
    #Exemple de backtest avec risk management.
    from ibkr_wrapper import TradingApp
    
    # 1. Connexion IBKR pour récupérer les données historiques
    print("Connecting to IBKR to fetch data...")
    app = TradingApp()
    app.connect("127.0.0.1", 7497, clientId=1)
    
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    
    timeout = 10
    start = time.time()
    while not app.connected and (time.time() - start) < timeout:
        time.sleep(0.5)
    
    if not app.connected:
        print("Failed to connect to IBKR")
        return
    
    print("Connected!\n")
    
    # 2. Récupérer les données historiques
    symbols = ['AAPL', 'MSFT']
    historical_data = {}
    
    for idx, symbol in enumerate(symbols):
        contract = app.get_contract(symbol)
        df = app.get_historical_data(5000 + idx, contract)
        if not df.empty:
            historical_data[symbol] = df
    
    app.disconnect()
    
    if not historical_data:
        print("No historical data retrieved")
        return
    
    # 3. Créer le risk manager
    risk_manager = RiskManager(
        initial_capital=100000,
        max_position_size_pct=0.1,
        max_risk_per_trade_pct=0.02,
        stop_loss_pct=0.05
    )
    
    # 4. Créer la stratégie
    strategy = SimpleMovingAverageStrategy(short_window=20, long_window=50)
    
    # 5. Créer le backtest runner
    backtest = BacktestRunner(strategy, risk_manager)
    
    # 6. Charger les données
    for symbol, df in historical_data.items():
        backtest.load_data(symbol, df)
    
    # 7. Exécuter le backtest
    results = backtest.run()
    
    # 8. Afficher les trades
    print("\n📈 TRADE HISTORY")
    print("="*70)
    for i, trade in enumerate(results['trades'], 1):
        print(f"Trade #{i}: {trade['symbol']}")
        print(f"  Entry: ${trade['entry_price']:.2f} at {trade['entry_time']}")
        print(f"  Exit:  ${trade['exit_price']:.2f} at {trade['exit_time']}")
        print(f"  P&L:   ${trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%) - {trade['reason']}")
        print()


if __name__ == "__main__":
    # Choisir le mode
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        run_backtest()
    else:
        print("Usage: python script.py [backtest|live]")
        print("Running backtest by default...\n")
        run_backtest()


"""

"""

app = TradingApp()
app.connect("127.0.0.1", 7497, clientId=1)
thread = threading.Thread(target=app.run)
thread.start()
time.sleep(2)

# Callback pour afficher les données
def print_live_data(reqId, data):
    print(f"[{data['last_update']}] {data['symbol']}: "
          f"Last=${data['last_price']}, Bid=${data['bid']}, Ask=${data['ask']}")

def print_bar(reqId, bar):
    print(f"[{bar['time']}] OHLC: {bar['open']}/{bar['high']}/{bar['low']}/{bar['close']}, Vol: {bar['volume']}")

# Streaming tick-by-tick
contract = app.get_contract("TSLA", exchange="BYX")
app.start_live_data(1, contract, callback=print_live_data)

# Streaming barres 5 secondes
#app.start_live_bars(2, contract, bar_size=5, callback=print_bar)

# Laisser tourner
time.sleep(60)

# Arrêter
app.stop_live_data(1)
#app.stop_live_bars(2)
app.disconnect()


"""
