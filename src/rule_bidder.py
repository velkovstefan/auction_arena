class RuleBidder:
    def __init__(self, name: str, budget: int, min_markup_pct: float = 0.1,
                 overestimate_pct: float = 10.0):
        self.name = name
        self.model_name = 'rule'
        self.initial_budget = budget
        self.budget = budget
        self.min_markup_pct = min_markup_pct
        self.overestimate_pct = overestimate_pct

        self.profit = 0
        self.items_won = []
        self.withdraw = False

    def _estimated_value(self, item) -> float:
        return item.true_value * (1 + self.overestimate_pct / 100)
    
    def act(self, item, highest_bid: int, explore: bool = True):
        ceiling = self._estimated_value(item)

        if highest_bid <= 0:
            next_price = item.price
        else:
            next_price = highest_bid + self.min_markup_pct * item.price

        if next_price <= ceiling and next_price <= self.budget:
            return int(next_price)

        return -1
    def finish_item(self, won: bool, item=None, winning_bid: int = None):
        if won and item is not None and winning_bid is not None:
            profit = item.true_value - winning_bid
            self.profit += profit
            self.items_won.append((item, winning_bid))
            self.budget -= winning_bid

    def reset_episode(self):
        self.budget = self.initial_budget
        self.profit = 0
        self.items_won = []
        self.withdraw = False