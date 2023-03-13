import time

from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random
import numpy as np




def min_distance_from_passenger(env: TaxiEnv, position):
    passengers = env.passengers
    distance = [manhattan_distance(position, p.position) for p in passengers]
    return min(distance)




class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        str_drop = "drop off passenger"
        str_pick = "pick up passenger"
        index_selected = -1
        taxi = env.get_taxi(agent_id)

        if str_drop in operators:
            index_selected = operators.index(str_drop)
        elif str_pick in operators:
            index_selected = operators.index(str_pick)
        else:
            children_heuristics = [self.heuristic(child, agent_id) for child in children]
            if taxi.passenger is not None:
                max_heuristic = min(children_heuristics)
            else:
                max_heuristic = max(children_heuristics)
            index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]

    def cash_differece(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id + 1) % 2)
        return taxi.cash - other_taxi.cash
    def close_gas(self, env: TaxiEnv, taxi_id:int):
        taxi = env.get_taxi(taxi_id)
        dis_list=[]
        for g in env.gas_stations:
            dis_list.append(manhattan_distance(g.position, taxi.position))
        return min(dis_list)
    def heuristic(self, env: TaxiEnv, taxi_id: int):
        if env.done() == True:
            if self.cash_differece(env, taxi_id) > 0:
                return np.inf
            elif self.cash_differece(env, taxi_id) < 0:
                return -np.inf
            else:
                return 0
        taxi = env.get_taxi(taxi_id)
        if env.taxi_is_occupied(taxi_id) == True:
            m_dis = manhattan_distance(taxi.position, taxi.passenger.destination)
            if m_dis > taxi.fuel:
                # not sure that this return is good
                return self.close_gas(env, taxi_id) + manhattan_distance(taxi.position, taxi.passenger.destination)
            else:
                return manhattan_distance(taxi.position, taxi.passenger.destination)
        p_list = []
        for p in env.passengers:
            p_list.append(p)
        travel_dist_list = []
        reward_list = []
        diff_list = []
        cash_diff = self.cash_differece(env, taxi_id)
        for p in p_list:
            travel_dist_list.append(
                manhattan_distance(p.position, taxi.position) + manhattan_distance(p.destination, p.position))
            reward_list.append(2 * manhattan_distance(p.position, p.destination))
            diff_list.append(reward_list[-1] - travel_dist_list[-1])
        max_diff = max(diff_list)
        max_dist = travel_dist_list[diff_list.index(max_diff)]
        if max_dist > taxi.fuel:
            return self.close_gas(env, taxi_id) + max_diff + cash_diff
        else:
            return max_diff + (taxi.fuel - max_dist) + cash_diff
class AgentMinimax(AgentGreedyImproved):
    def heuristic(self, env: TaxiEnv, taxi_id: int):
        if env.done() == True:
            if self.cash_differece(env, taxi_id)>0:
                return np.inf
            elif self.cash_differece(env, taxi_id) < 0:
                return -np.inf
            else:
                return 0
        taxi = env.get_taxi(taxi_id)
        if env.taxi_is_occupied(taxi_id) == True:
            m_dis = manhattan_distance(taxi.position, taxi.passenger.destination)
            if m_dis > taxi.fuel:
                # not sure that this return is good
                return self.close_gas(env,taxi_id)+8-manhattan_distance(taxi.position, taxi.passenger.destination)+taxi.cash*16
            else:
                return 8-manhattan_distance(taxi.position, taxi.passenger.destination)+taxi.fuel+taxi.cash*16
        p_list = []
        for p in env.passengers:
            p_list.append(p)
        travel_dist_list = []
        reward_list=[]
        diff_list = []
        cash_diff = self.cash_differece(env, taxi_id)
        for p in p_list:
            travel_dist_list.append(manhattan_distance(p.position, taxi.position) + manhattan_distance(p.destination,p.position))
            reward_list.append(2*manhattan_distance(p.position,p.destination))
            diff_list.append(reward_list[-1]-travel_dist_list[-1])
        max_diff = max(diff_list)
        max_dist = travel_dist_list[diff_list.index(max_diff)]
        if max_dist > taxi.fuel:
            return self.close_gas(env, taxi_id) + max_diff + taxi.cash*16
        else:
            return max_diff + (taxi.fuel - max_dist) + taxi.cash*16
    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        turn_limit = time.time() + 0.8 * time_limit
        time_range = True
        depth = 2
        operators = env.get_legal_operators(agent_id)
        step_to_do = None
        step_saved = operators[random.randrange(0, len(operators))]
        while time_range:
            step_to_do = self.RBMinMax(env, agent_id, turn_limit, depth, depth,0)
            if step_to_do != None:
                step_saved = step_to_do
            if time.time() >= turn_limit or step_to_do == np.inf:
                time_range = False
            else:
                depth += 2
        return step_saved
    def RBMinMax(self, env: TaxiEnv, agent_id, time_limit, depth, in_depth,turn)->int:
        if time.time() > time_limit:
            return None
        taxi1 = env.get_taxi(agent_id)
        taxi2 = env.get_taxi((agent_id + 1) % 2)
        if depth == 0 or env.done() == True:
            return self.heuristic(env, agent_id)
        if turn % 2 == 0:
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
            children_max = [self.RBMinMax(child, 1-agent_id,time_limit,depth-1,in_depth,turn+1) for child in children]
            children_max_fix = [child for child in children_max if child == None]
            if len(children_max_fix) > 0:
                return None
            max_child = max(children_max)
            if depth == in_depth:
                index_selected = children_max.index(max_child)
                return operators[index_selected]
            else:
                return max_child
        else:
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
            children_min = [self.RBMinMax(child, 1-agent_id, time_limit, depth - 1, in_depth,turn+1) for child in
                            children]
            children_min_fix = [child for child in children_min if child == None]
            if len(children_min_fix) > 0:
                return None
            min_child = min(children_min)
            return min_child

class AgentAlphaBeta(AgentMinimax):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        turn_limit = time.time() + 0.8 * time_limit
        time_range = True
        depth = 2
        operators = env.get_legal_operators(agent_id)
        step_to_do = None
        step_saved = operators[random.randrange(0, len(operators))]
        while time_range:
            step_to_do = self.RBMinMax(env, agent_id, turn_limit, depth, depth,0,-np.inf,np.inf)
            if step_to_do != None and step_to_do != np.inf and step_to_do != -np.inf:
                step_saved = step_to_do
            if time.time() >= turn_limit or step_to_do == None:
                time_range = False
            else:
                depth += 2
        return step_saved
    def RBMinMax(self, env: TaxiEnv, agent_id, time_limit, depth, in_depth,turn,alpha,beta)->int:
        if time.time() > time_limit:
            return None
        taxi1 = env.get_taxi(agent_id)
        taxi2 = env.get_taxi((agent_id + 1) % 2)
        if depth == 0 or env.done() == True:
            return self.heuristic(env, agent_id)
        if turn % 2 == 0:
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
            children_max = [self.RBMinMax(child, 1-agent_id,time_limit,depth-1,in_depth,turn+1,alpha,beta) for child in children]
            children_max_fix = [child for child in children_max if child == None]
            if len(children_max_fix) > 0:
                return None
            max_child = max(children_max)
            alpha = max(max_child,alpha)
            if max_child > beta:
                return np.inf
            if depth == in_depth:
                if max_child == np.inf:
                    return np.inf
                index_selected = children_max.index(max_child)
                return operators[index_selected]
            else:
                return max_child
        else:
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
            children_min = [self.RBMinMax(child, 1-agent_id, time_limit, depth - 1, in_depth,turn+1,alpha,beta) for child in
                            children]
            children_none = [child for child in children_min if child == None]
            if len(children_none) > 0:
                return None
            #children_min_fix = [child for child in children_min if child > np.inf]
            min_child = min(children_min)
            beta = min(min_child,beta)
            if min_child < alpha:
                return -np.inf
            return min_child

class AgentExpectimax(AgentMinimax):
    # TODO: section d : 1
    def calc_expectency(self,operators, children_value):
        no_movement_op = ['pick up passenger', 'refuel', 'drop off passenger']
        weighted_sum = 0
        expectency = 0
        for op in operators:
            if op in no_movement_op:
                weighted_sum+=2
            else:
                weighted_sum+=1
        for child, op in zip(children_value, operators):
            if op in no_movement_op:
                expectency += 2*child/weighted_sum
            else:
                expectency += child/weighted_sum
        return expectency

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        turn_limit = time.time() + 0.8 * time_limit
        time_range = True
        depth = 2
        operators = env.get_legal_operators(agent_id)
        step_to_do = None
        step_saved = operators[random.randrange(0, len(operators))]
        while time_range:
            step_to_do = self.RBMinMax(env, agent_id, turn_limit, depth, depth, 0)
            if step_to_do != None:
                step_saved = step_to_do
            if time.time() >= turn_limit or step_to_do == np.inf:
                time_range = False
            else:
                depth += 2
        return step_saved

    def RBMinMax(self, env: TaxiEnv, agent_id, time_limit, depth, in_depth, turn) -> int:
        if time.time() > time_limit:
            return None
        taxi1 = env.get_taxi(agent_id)
        taxi2 = env.get_taxi((agent_id + 1) % 2)
        if depth == 0 or env.done() == True:
            return self.heuristic(env, agent_id)
        if turn % 2 == 0:
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
            children_max = [self.RBMinMax(child, 1 - agent_id, time_limit, depth - 1, in_depth, turn + 1) for child in
                            children]
            children_max_fix = [child for child in children_max if child == None]
            if len(children_max_fix) > 0:
                return None
            max_child = max(children_max)
            if depth == in_depth:
                index_selected = children_max.index(max_child)
                return operators[index_selected]
            else:
                return max_child
        else:
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
            children_exp = [self.RBMinMax(child, 1 - agent_id, time_limit, depth - 1, in_depth, turn + 1) for child in
                            children]
            children_exp_fix = [child for child in children_exp if child == None]
            if len(children_exp_fix) > 0:
                return None
            return self.calc_expectency(operators,children_exp)
