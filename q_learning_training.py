import argparse
import random
import copy

from src.item_base import create_items
from src.q_learning_bidder import QLearningBidder
from src.rule_bidder import RuleBidder

def run_item_auction(item, agents, min_markup_pct: float, max_rounds: int = 30, explore: bool = True):
        highest_bid = -1
        highest_bidder = None
        active = {a.name: True for a in agents}  # False once withdrawn

        for _ in range(max_rounds):
            any_new_bid = False
            for agent in agents:
                if agent is highest_bidder or not active[agent.name]:
                    continue
                bid_price = agent.act(item, highest_bid, explore=explore)
                if bid_price is None or bid_price < 0:
                    active[agent.name] = False
                    continue
                if bid_price > highest_bid:
                    highest_bid = bid_price
                    highest_bidder = agent
                    any_new_bid = True
                else:
                    # invalid/insufficient raise -- treat as withdraw
                    active[agent.name] = False

            if not any_new_bid:
                break

        # resolve outcome for every agent
        for agent in agents:
            won = (agent is highest_bidder)
            agent.finish_item(won=won, item=item, winning_bid=highest_bid if won else None)

        return highest_bidder, highest_bid

def run_episode(items, agents, min_markup_pct: float, explore: bool = True):
    for agent in agents:
        agent.reset_episode()
    shuffled = items.copy()
    random.shuffle(shuffled)
    for item in shuffled:
        item.reset_price()
        run_item_auction(item, agents, min_markup_pct=min_markup_pct, explore=explore)
    return {agent.name: agent.profit for agent in agents}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--items', type=str, default='data/example/items_dataset.jsonl')
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--num_qlearning', type=int, default=2)
    parser.add_argument('--num_rule', type=int, default=2)
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--min_markup_pct', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay_episodes', type=int, default=None,
                         help='Episodes over which epsilon decays linearly. Defaults to 80%% of --episodes.')
    parser.add_argument('--eval_every', type=int, default=100,
                         help='Run --eval_episodes greedy (no-exploration) episodes every N training episodes, averaged for the plotted curve.')
    parser.add_argument('--eval_episodes', type=int, default=20,
                         help='Number of greedy eval episodes averaged at each checkpoint (reduces noise from small item sets).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='.')
    args = parser.parse_args()

    random.seed(args.seed)

    items = create_items(args.items)

    if not all(it.category is not None for it in items):
            print("WARNING: some items are missing category/condition/rarity/brand_tier -- "
                  "the Q-learning state won't be meaningful for those items.")

    agents = []
    for a in range(args.num_qlearning):
        agents.append(
             QLearningBidder(
                  name=f"Q{a+1}", budget= args.budget, min_markup_pct=args.min_markup_pct, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon_start,
             )
        )

    for a in range(args.num_rule):
        agents.append(
             RuleBidder(
                name=f"Rule{a+1}", budget=args.budget, min_markup_pct=args.min_markup_pct,
                overestimate_pct=random.choice([0, 10, 20])
             )
        )

    
    q_agents = [a for a in agents if isinstance(a, QLearningBidder)]
    decay_episodes = args.epsilon_decay_episodes or int(0.8 * args.episodes)
    history = {agent.name: [] for agent in agents}

    for ep in range(1, args.episodes + 1):
        frac = min(1.0, ep / max(1, decay_episodes))
        eps = args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)

        for qa in q_agents:
            qa.epsilon = eps
        run_episode(items=items, agents=agents, min_markup_pct=args.min_markup_pct, explore=True)

        if ep % args.eval_every == 0 or ep == args.episodes:
            sums = {agent.name: 0.0 for agent in agents}
            for _ in range(args.eval_episodes):
                eval_profits = run_episode(items, agents, min_markup_pct=args.min_markup_pct, explore=False)
                for name, profit in eval_profits.items():
                    sums[name] += profit
            avg_profits = {name: total / args.eval_episodes for name, total in sums.items()}
            for name, profit in avg_profits.items():
                history[name].append((ep, profit))
            profits_str = ', '.join(f"{n}={p:.0f}" for n, p in avg_profits.items())
            print(f"Episode {ep:6d} | epsilon={eps:.3f} | avg eval profit over {args.eval_episodes} episodes: {profits_str}")


    # save Q-tables
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    for qa in q_agents:
        path = os.path.join(args.out_dir, f"q_table_{qa.name}.json")
        qa.save_q_table(path)
        print(f"Saved {path} ({len(qa.q_table)} states learned)")

    # plot training curve
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
    
            plt.figure(figsize=(9, 5))
            for name, points in history.items():
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                plt.plot(xs, ys, label=name)
            plt.xlabel('Episode')
            plt.ylabel(f'Profit (avg over {args.eval_episodes} greedy eval episodes)')
            plt.title('Training curve: profit per agent over time')
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(args.out_dir, 'training_curve.png')
            plt.savefig(plot_path, dpi=150)
            print(f"Saved {plot_path}")
        except Exception as e:
            print(f"Could not save plot: {e!r}")


if __name__ == '__main__':
    main()