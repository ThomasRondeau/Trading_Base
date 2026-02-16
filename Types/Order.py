class Order:
    def __init__(self, strategy_ID : int, symbol: str, side: str, quantity: int):
        self.strategy_ID = strategy_ID
        self.symbol = symbol
        self.side = side  # 'BUY' ou 'SELL'
        self.quantity = quantity
