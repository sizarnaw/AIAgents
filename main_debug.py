import random
import time

from TaxiEnv import TaxiEnv
import argparse
import submission
import Agent
import  sys


def run_agents():
    parser = argparse.ArgumentParser(description='Test your submission by pitting agents against each other.')
    parser.add_argument('-improvedgreedy', type=str,
                        help='First agent')
    parser.add_argument('-random', type=str,
                        help='Second agent')
    parser.add_argument('-time', '--0.5', type=float, nargs='?', help='Time limit for each turn in seconds', default=1)
    parser.add_argument('-seed', '--1234', nargs='?', type=int, help='Seed to be used for generating the game',
                        default=random.randint(0, 176400))
    parser.add_argument('-c', '--1000', nargs='?', type=int, help='Number of steps each taxi gets before game is over',
                        default=4761)
    parser.add_argument('--print_game', action='store_true')

    args = parser.parse_args()

    agents = {
        "random": Agent.AgentRandom(),
        "greedy": Agent.AgentGreedy(),
        "minimax": submission.AgentMinimax(),
        "improvedgreedy": submission.AgentGreedyImproved(),
        "alphabeta": submission.AgentAlphaBeta(),
        "expectimax": submission.AgentExpectimax()
    }

    #agent_names = sys.argv
    agent_names = ["improvedgreedy", "greedy"]
    env = TaxiEnv()
    seed = 900
    env.generate(seed)
    print_game = True
    if print_game:
        print('initial board:')
        env.print()

    for _ in range(10000):
        for i, agent_name in enumerate(agent_names):
            agent = agents[agent_name]
            start = time.time()
            op = agent.run_step(env, i, 0.5)
            end = time.time()
            if end - start > 1000:
                raise RuntimeError("Agent used too much time!")
            env.apply_operator(i, op)
            if print_game:
                print('taxi ' + str(i) + ' chose ' + op)
                env.print()
        if env.done():
            break
    balances = env.get_balances()
    print(balances)
    if balances[0] == balances[1]:
        print('draw')
    else:
        print('taxi', balances.index(max(balances)), 'wins!')


if __name__ == "__main__":
    run_agents()
