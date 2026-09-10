import json
import random
from collections import defaultdict

CATEGORIES = [
    'widget', 'household', 'tool', 'decor', 'gadget',
    'appliance', 'collectible', 'instrument', 'electronics', 'equipment',
]
CATEGORY_IDX = {c: i for i, c in enumerate(CATEGORIES)}

ACTION_WITHDRAW = 0
ACTION_RAISE_MIN = 1
ACTION_RAISE_MID = 2
ACTION_RAISE_BIG = 3
ACTIONS = [ACTION_WITHDRAW, ACTION_RAISE_MIN, ACTION_RAISE_MID, ACTION_RAISE_BIG]
ACTION_MARKUP = {
    ACTION_RAISE_MIN: None,   # filled in per-agent from min_markup_pct
    ACTION_RAISE_MID: 0.25,
    ACTION_RAISE_BIG: 0.5,
}

def _bucket(x: float) -> int:
    if x < 0.4:
        return 0
    if x < 0.7:
        return 1
    return 2

def _bucket_bid_ratio(highest_bid: float, price: float) -> int:
    if highest_bid <= 0:
        return 0
    ratio = highest_bid / price
    if ratio <= 1.2:
        return 1
    if ratio <= 1.5:
        return 2
    if ratio <= 2.0:
        return 3
    return 4


def _bucket_budget_ratio(budget: float, initial_budget: float) -> int:
    ratio = budget / initial_budget if initial_budget > 0 else 0
    if ratio < 0.2:
        return 0
    if ratio < 0.4:
        return 1
    if ratio < 0.6:
        return 2
    if ratio < 0.8:
        return 3
    return 4


class QLearningBidder():
    def __init__(self, name: str, budget: int, min_markup_pct: float = 0.1, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 1.0):
        self.name = name
        self.model_name = 'qlearning'
        self.initial_budget = budget
        self.budget = budget
        self.min_markup_pct = min_markup_pct
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = defaultdict(lambda: [0.0] * len(ACTIONS))

        # per-episode bookkeeping
        self.profit = 0
        self.items_won = []
        self.withdraw = False
        # trajectory of (state, action) for the item currently being bid on
        self._cur_trajectory = []

    # ---------------- state ----------------
    
    def get_state(self, item, highest_bid: int):
        return (
            CATEGORY_IDX.get(item.category, len(CATEGORIES)),
            _bucket(item.condition),
            _bucket(item.rarity),
            item.brand_tier,
            _bucket_bid_ratio(highest_bid, item.price),
            _bucket_budget_ratio(self.budget, self.initial_budget),
        )

    # ---------------- acting ----------------

    def choose_action(self, state, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q_values = self.q_table[state] # ja zema redicata so vrednostite za soodvetniot state
        max_q = max(q_values)
        best = [a for a, q in zip(ACTIONS, q_values) if q == max_q]

        # ACTIONS = [0, 1, 2, 3]
        # zip pairs them up: (0, 2.1), (1, 5.4), (2, 5.4), (3, -1.0)
        # keep only the actions whose q matches max_q → best = [1, 2]
        
        return random.choice(best)

    def act(self, item, highest_bid: int, explore: bool = True):
        state = self.get_state(item, highest_bid)
        action = self.choose_action(state, explore=explore)

        if action == ACTION_WITHDRAW:
            bid_price = -1
        else:
            markup_pct = self.min_markup_pct if action == ACTION_RAISE_MIN else ACTION_MARKUP[action]
            base = highest_bid if highest_bid > 0 else item.price
            increase = item.price * markup_pct
            bid_price = int(base + increase) if highest_bid > 0 else int(item.price)
            if bid_price > self.budget:
                action = ACTION_WITHDRAW
                bid_price = -1
        self._cur_trajectory.append((state, action))
        return bid_price

    # ---------------- learning ----------------
    
    def update(self, state, action, reward: float, next_state, done: bool):
        q_values = self.q_table[state]
        target = reward if done else reward + self.gamma * max(self.q_table[next_state])
        q_values[action] += self.alpha * (target - q_values[action]) 

    def finish_item(self, won: bool, item=None, winning_bid: int = None):
        traj = self._cur_trajectory
        if not traj:
            return

        terminal_reward = 0.0
        if won and item is not None and winning_bid is not None:
            terminal_reward = float(item.true_value - winning_bid)
            self.profit += terminal_reward
            self.items_won.append((item, winning_bid))
            self.budget -= winning_bid

        for i in range(len(traj) - 1):
            state, action = traj[i]
            next_state, _ = traj[i + 1]
            self.update(state, action, reward=0.0, next_state=next_state, done=False)

        last_state, last_action = traj[-1]
        # dummy next_state fine since done=True means it's never read
        self.update(last_state, last_action, reward=terminal_reward, next_state=last_state, done=True)

        self._cur_trajectory = []

    def reset_episode(self):
            self.budget = self.initial_budget
            self.profit = 0
            self.items_won = []
            self.withdraw = False
            self._cur_trajectory = []
    
    # ---------------- persistence ----------------

    def save_q_table(self, path: str):
        serializable = {json.dumps(list(state)): values for state, values in self.q_table.items()}
        with open(path, 'w') as f:
            json.dump({
                'name': self.name,
                'min_markup_pct': self.min_markup_pct,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'q_table': serializable,
            }, f)

    def load_q_table(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.q_table = defaultdict(lambda: [0.0] * len(ACTIONS))
        for state_str, values in data['q_table'].items():
            self.q_table[tuple(json.loads(state_str))] = values