import random
import argparse
import sys
from copy import deepcopy
import time
import math
sys.setrecursionlimit(50000)

###################
### Problem 1: BnB###
###################


def bnb(items, time_limit, capacity):
    ans = [0] * len(items)
    items_sorted = sorted(
        items, key=lambda x: x['value']/x['weight'], reverse=True)
    best_total_val = 0
    LB, UB, temp_capacity_left = 0, 0, capacity
    UB_composition, best_composition, curr_composition = [
        0] * len(items), [0] * len(items), [0] * len(items)
    starting_time = time.time()
    trace = []

    # Calculate intial UB
    for i in range(len(items_sorted)):
        if temp_capacity_left >= items_sorted[i]['weight']:
            UB += items_sorted[i]['value']
            temp_capacity_left -= items_sorted[i]['weight']
            UB_composition[i] = 1
        else:
            UB += items_sorted[i]['value'] * \
                (temp_capacity_left / items_sorted[i]['weight'])
            UB_composition[i] = temp_capacity_left / items_sorted[i]['weight']
            break
        if temp_capacity_left == 0:
            break

    def retrieve_UB_last_idx(updated_composition, from_idx_backwards=len(UB_composition)-1):
        for i in range(from_idx_backwards, -1, -1):
            if updated_composition[i] > 0:
                return i

    def get_or_default(val, default_val):
        if val is None:
            return default_val
        return val

    def get_updated_UB_and_composition(i, UB, UB_composition, action: str) -> tuple[float, list[float]]:
        updated_composition = deepcopy(UB_composition)
        updated_UB = UB
        if action == 'exclude':
            if UB_composition[i] > 0:
                updated_UB -= items_sorted[i]['value']*UB_composition[i]
                weight_tofill = items_sorted[i]['weight'] * UB_composition[i]
                updated_composition[i] = 0
                UB_last_idx = get_or_default(
                    retrieve_UB_last_idx(updated_composition), i+1)
                if UB_last_idx <= i:
                    if updated_composition[UB_last_idx] != 1:
                        print(f'weight_tofill: {weight_tofill}')
                        print(
                            f"trying to exclude {i} but UB_last_idx is {UB_last_idx}")
                        print(updated_composition)
                        raise Exception(
                            "Error: updated_composition[UB_last_idx] != 1")
                    UB_last_idx = i + 1
                else:
                    if UB_last_idx == len(items_sorted):  # no more items to fill
                        return updated_UB, updated_composition
                    updated_UB -= items_sorted[UB_last_idx]['value'] * \
                        updated_composition[UB_last_idx]
                    weight_tofill += items_sorted[UB_last_idx]['weight'] * \
                        updated_composition[UB_last_idx]
                    updated_composition[UB_last_idx] = 0
                while weight_tofill > 0:
                    if UB_last_idx == len(items_sorted):
                        break
                    if items_sorted[UB_last_idx]['weight'] > weight_tofill:
                        new_item_percentage = weight_tofill / \
                            items_sorted[UB_last_idx]['weight']
                        updated_UB += items_sorted[UB_last_idx]['value'] * \
                            new_item_percentage
                        updated_composition[UB_last_idx] = new_item_percentage
                        break
                    else:
                        updated_UB += items_sorted[UB_last_idx]['value']
                        weight_tofill -= items_sorted[UB_last_idx]['weight']
                        updated_composition[UB_last_idx] = 1
                        UB_last_idx += 1
        elif action == 'include':
            if UB_composition[i] > 0:
                weight_to_discard = items_sorted[i]['weight'] * \
                    (1 - UB_composition[i])
                updated_UB += items_sorted[i]['value'] * \
                    (1 - UB_composition[i])
            else:
                weight_to_discard = items_sorted[i]['weight']
                updated_UB += items_sorted[i]['value']

            UB_last_idx = retrieve_UB_last_idx(updated_composition)
            while weight_to_discard > 0:
                if updated_composition[UB_last_idx] < 1:
                    weight_to_discard += items_sorted[UB_last_idx]['weight'] * (
                        1 - updated_composition[UB_last_idx])
                    updated_UB += items_sorted[UB_last_idx]['value'] * \
                        (1 - updated_composition[UB_last_idx])
                    updated_composition[UB_last_idx] = 1
                    continue
                if weight_to_discard <= items_sorted[UB_last_idx]['weight']:
                    new_item_percentage = 1 - weight_to_discard / \
                        items_sorted[UB_last_idx]['weight']
                    updated_UB -= items_sorted[UB_last_idx]['value'] * \
                        (1 - new_item_percentage)
                    updated_composition[UB_last_idx] = new_item_percentage
                    break
                else:
                    updated_UB -= items_sorted[UB_last_idx]['value']
                    weight_to_discard -= items_sorted[UB_last_idx]['weight']
                    updated_composition[UB_last_idx] = 0
                    UB_last_idx = retrieve_UB_last_idx(
                        updated_composition, UB_last_idx-1)
            updated_composition[i] = 1
        return updated_UB, updated_composition

    def bnb_recur(i, LB, UB, UB_composition, capacity_left):
        if time.time() - starting_time > time_limit:
            raise Exception("Time limit exceeded")
        nonlocal best_total_val, best_composition, curr_composition, trace
        if i == len(items_sorted) or UB <= best_total_val:
            return

        # case 1: not taking i
        new_UB, new_composition = get_updated_UB_and_composition(
            i, UB, UB_composition, 'exclude')
        bnb_recur(i+1, LB, new_UB, new_composition, capacity_left)

        # case 2: taking i
        if items_sorted[i]['weight'] <= capacity_left:
            curr_composition[i] = 1
            if LB + items_sorted[i]['value'] > best_total_val:
                best_total_val = LB + items_sorted[i]['value']
                trace.append((time.time() - starting_time, best_total_val))
                best_composition = deepcopy(curr_composition)
            new_UB, new_composition = get_updated_UB_and_composition(
                i, UB, UB_composition, 'include')
            bnb_recur(i+1, LB + items_sorted[i]['value'], new_UB,
                      new_composition, capacity_left - items_sorted[i]['weight'])
            curr_composition[i] = 0

    bnb_recur(0, LB, UB, UB_composition, capacity)

    for i, taken in enumerate(best_composition):
        if taken:
            ans[items_sorted[i]['orig_idx']] = 1
    return best_total_val, ans, trace, time.time() - starting_time

######################
### Problem 2: Approx###
#######################


def _knapsack_dp(items, capacity, trace, starting_time, big_total_val, time_limit):
    n = len(items)
    capacity = int(capacity)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Build table dp[][] in bottom-up manner
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if time.time() - starting_time > time_limit:
                raise Exception("Time limit exceeded")
            if items[i-1]['weight'] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-items[i-1]
                               ['weight']] + items[i-1]['value'])
            else:
                dp[i][w] = dp[i-1][w]

    # find the selected items
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if time.time() - starting_time > time_limit:
            raise Exception("Time limit exceeded")
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i-1)
            trace.append((time.time() - starting_time,
                         big_total_val + items[i-1]['value']))
            w -= items[i-1]['weight']

    return dp[n][capacity], selected_items


def approx(items, time_limit, capacity, ratio):
    threadhold = ratio * capacity
    ans = [0] * len(items)
    items_sorted = sorted(
        items, key=lambda x: x['value']/x['weight'], reverse=True)
    best_total_val = 0
    large, small = [], []
    trace, starting_time = [], time.time()
    for item in items_sorted:
        if item['weight'] > threadhold:
            large.append(item)
        else:
            small.append(item)

    # Greedy on large items
    capacity_left = capacity
    for item in large:
        if item['weight'] <= capacity_left:
            ans[item['orig_idx']] = 1
            best_total_val += item['value']
            trace.append((time.time() - starting_time, best_total_val))
            capacity_left -= item['weight']

    # dp on small items
    val, selected_items_small = _knapsack_dp(
        small, capacity_left, trace, starting_time, best_total_val, time_limit)
    best_total_val += val
    for i in selected_items_small:
        ans[small[i]['orig_idx']] = 1
    return best_total_val, ans, trace, time.time() - starting_time

##########################
### Problem 3.1: Genetic###
##########################


class LS1:
    def __init__(self, items, capacity, population_size=50, generations=100, mutation_rate=0.05, time_limit=900, seed=42):
        self.items = items
        self.capacity = capacity
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.time_limit = time_limit
        self.random = random.Random(self.seed)
        self.best_value = 0
        self.best_solution = None
        self.trace = []
        self.start_time = time.time()

    def generate_initial_population(self):
        population = []
        item_count = len(self.items)  # Get the number of items available
        indices = list(range(item_count))  # Create a list of indices

        for _ in range(self.population_size):
            # Shuffle indices for each new individual
            self.random.shuffle(indices)
            individual = [0] * item_count  # Initialize individual with zeros
            current_weight = 0  # Track the current weight of the knapsack

            for idx in indices:
                item = self.items[idx]
                if self.random.random() > 0.5 and current_weight + item['weight'] <= self.capacity:
                    individual[idx] = 1
                    current_weight += item['weight']

            # Add the generated individual to the population
            population.append(individual)

        return population

    def calculate_fitness(self, individual):
        total_value = sum(item['value']
                          for item, inc in zip(self.items, individual) if inc)
        total_weight = sum(item['weight']
                           for item, inc in zip(self.items, individual) if inc)
        return total_value if total_weight <= self.capacity else 0

    def select_parents(self, population, selection_size):
        sorted_population = sorted(
            population, key=lambda ind: self.calculate_fitness(ind), reverse=True)
        return sorted_population[:selection_size]

    def crossover(self, parent1, parent2):
        child1 = []
        child2 = []
        current_weight1 = 0
        current_weight2 = 0

        for i in range(len(parent1)):
            if self.random.random() > 0.5:  # Randomly choose genes from parents
                # Decide if adding the gene from parent1 to child1 is feasible
                if current_weight1 + (self.items[i]['weight'] if parent1[i] == 1 else 0) <= self.capacity:
                    child1.append(parent1[i])
                    current_weight1 += self.items[i]['weight'] * parent1[i]
                else:
                    # Do not add the item if it exceeds capacity
                    child1.append(0)

                # Similarly for parent2 to child2
                if current_weight2 + (self.items[i]['weight'] if parent2[i] == 1 else 0) <= self.capacity:
                    child2.append(parent2[i])
                    current_weight2 += self.items[i]['weight'] * parent2[i]
                else:
                    child2.append(0)
            else:
                # Reverse the roles for the second child
                if current_weight1 + (self.items[i]['weight'] if parent2[i] == 1 else 0) <= self.capacity:
                    child1.append(parent2[i])
                    current_weight1 += self.items[i]['weight'] * parent2[i]
                else:
                    child1.append(0)

                if current_weight2 + (self.items[i]['weight'] if parent1[i] == 1 else 0) <= self.capacity:
                    child2.append(parent1[i])
                    current_weight2 += self.items[i]['weight'] * parent1[i]
                else:
                    child2.append(0)

        return child1, child2

    def mutate(self, individual):
        current_weight = sum(self.items[i]['weight']
                             for i, included in enumerate(individual) if included)
        mutated = individual[:]

        for i in range(len(mutated)):
            if self.random.random() < self.mutation_rate:
                if mutated[i] == 1:
                    # Remove the item (flip from 1 to 0)
                    mutated[i] = 0
                    current_weight -= self.items[i]['weight']
                else:
                    # Add the item (flip from 0 to 1) only if it does not exceed the capacity
                    if current_weight + self.items[i]['weight'] <= self.capacity:
                        mutated[i] = 1
                        current_weight += self.items[i]['weight']

        return mutated

    def genetic_algorithm(self):
        start_time = time.time()
        population = self.generate_initial_population()
        best_value = max(self.calculate_fitness(ind) for ind in population)
        print(f"Initial best_value: {best_value}")
        trace = [(time.time() - start_time, best_value)]

        # for example, 10% of the population
        elitism_count = max(1, int(self.population_size * 0.6))

        for _ in range(self.generations):
            if time.time() - start_time > self.time_limit:
                raise Exception("Time limit exceeded")
            new_population = []
            parents = self.select_parents(
                population, self.population_size // 2)
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self.crossover(parents[i], parents[i+1])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            # Sort population by fitness and keep the best individuals
            population_sorted = sorted(
                population, key=lambda ind: self.calculate_fitness(ind), reverse=True)
            new_population.extend(population_sorted[:elitism_count])

            # Ensure the population size remains constant
            new_population = sorted(new_population, key=lambda ind: self.calculate_fitness(
                ind), reverse=True)[:self.population_size]

            # Since it's sorted, the first one is the best
            current_best = new_population[0]
            best_value = self.calculate_fitness(current_best)
            elapsed_time = time.time() - start_time
            trace.append((elapsed_time, best_value))

            population = new_population

        best_solution = population[0]
        return best_solution, trace

    def execute(self):
        best_solution, trace = self.genetic_algorithm()
        total_value = self.calculate_fitness(best_solution)
        chosen_items = [idx for idx, included in enumerate(
            best_solution) if included]
        chosen_items_onehot = [0] * len(self.items)
        for idx in chosen_items:
            chosen_items_onehot[self.items[idx]['orig_idx']] = 1
        return total_value, chosen_items_onehot, trace, time.time() - self.start_time

######################################
### Problem 3.2: Simulated Annealing###
######################################


class LS2:
    def __init__(self, items, capacity, temp, cooling_rate, seed=None, time_limit=900):
        self.items = items
        self.capacity = capacity
        self.temp = temp  # Initial temperature
        self.cooling_rate = cooling_rate  # Cooling rate
        self.current_solution = [0] * len(items)
        self.current_value = 0
        self.best_solution = list(self.current_solution)
        self.best_value = 0
        self.start_time = time.time()
        self.time_limit = time_limit
        random.seed(time.time())

    def calculate_fitness(self, solution):
        total_weight = sum(item['weight'] * include for item,
                           include in zip(self.items, solution))
        total_value = sum(item['value'] * include for item,
                          include in zip(self.items, solution))
        if total_weight <= self.capacity:
            return total_value
        else:
            return 0  # Penalize solutions that exceed the capacity

    def generate_neighbor(self):
        neighbor = list(self.current_solution)
        index = random.randint(0, len(neighbor) - 1)
        neighbor[index] = 1 - neighbor[index]  # Flip the item inclusion
        if sum(self.items[i]['weight'] * include for i, include in enumerate(neighbor)) > self.capacity:
            # Revert if not feasible
            neighbor[index] = self.current_solution[index]
        return neighbor

    def accept_probability(self, candidate_value):
        if candidate_value > self.current_value:
            return 1.0
        else:
            return math.exp((candidate_value - self.current_value) / self.temp)

    def run(self):
        start_time = time.time()
        trace = []
        while self.temp > 1:
            if time.time() - start_time > self.time_limit:
                raise Exception("Time limit exceeded")
            neighbor = self.generate_neighbor()
            neighbor_value = self.calculate_fitness(neighbor)
            if self.accept_probability(neighbor_value) > random.random():
                self.current_solution = neighbor
                self.current_value = neighbor_value
            if self.current_value > self.best_value:
                self.best_value = self.current_value
                self.best_solution = list(self.current_solution)
                elapsed_time = time.time() - start_time
                # Append new max and time
                trace.append((elapsed_time, self.best_value))
            self.temp *= (1 - self.cooling_rate)
        return self.best_solution, self.best_value, trace, time.time() - start_time


def main():
    parser = argparse.ArgumentParser(
        description="Algorithm selector for datasets")
    parser.add_argument("-inst", type=str, required=True,
                        help="Filename of the dataset")
    parser.add_argument("-alg", type=str, choices=['BnB', 'Approx', 'LS1', 'LS2'],
                        required=True, help="Algorithm to use [BnB, Approx, LS1, LS2]")
    parser.add_argument("-time", type=int, required=True,
                        help="Cut-off time in seconds")
    parser.add_argument("-seed", type=int, required=False, help="Random seed")

    args = parser.parse_args()

    filename = args.inst
    algorithm = args.alg
    time_limit = args.time
    seed = args.seed
    if seed is None:
        seed = 42

    with open(filename, 'r') as file:
        lines = file.readlines()
        capacity = float(lines[0].split()[1].strip())
        items = []
        i = 0
        for line in lines[1:]:
            items.append({
                "value": float(line.split()[0].strip()),
                "weight": float(line.split()[1].strip()),
                "orig_idx": i,
            })
            i += 1

    if algorithm == 'BnB':
        val, comp, trace, time = bnb(items, time_limit, capacity)
        with open(f'{filename.split("/")[-1]}_{algorithm}_{time_limit}.sol', 'w') as file:
            file.write(f'{val}\n')
            file.write(','.join([str(c) for c in comp]))
        with open(f'{filename.split("/")[-1]}_{algorithm}_{time_limit}.trace', 'w') as file:
            for t in trace:
                file.write(f'{t[0]}, {t[1]}\n')

    elif algorithm == 'Approx':
        val, comp, trace, time = approx(items, time_limit, capacity, 0.1)
        with open(f'{filename.split("/")[-1]}_{algorithm}_{time_limit}.sol', 'w') as file:
            file.write(f'{val}\n')
            file.write(','.join([str(c) for c in comp]))
        with open(f'{filename.split("/")[-1]}_{algorithm}_{time_limit}.trace', 'w') as file:
            for t in trace:
                file.write(f'{t[0]}, {t[1]}\n')

    elif algorithm == 'LS1':
        ls1 = LS1(items, capacity, population_size=300, generations=100,
                  mutation_rate=0.01, time_limit=time_limit, seed=seed)
        val, comp, trace, time = ls1.execute()
        with open(f'{filename.split("/")[-1]}_{algorithm}_{time_limit}.sol', 'w') as file:
            file.write(f'{val}\n')
            file.write(','.join([str(c) for c in comp]))
        with open(f'{filename.split("/")[-1]}_{algorithm}_{time_limit}.trace', 'w') as file:
            for t in trace:
                file.write(f'{t[0]}, {t[1]}\n')

    elif algorithm == 'LS2':
        sa = LS2(items, capacity, temp=15000, cooling_rate=0.0001, seed=seed)
        best_solution, best_value, trace, time = sa.run()  # Assuming `run()` returns these
        solution_filename = f'{filename.split("/")[-1]}_{algorithm}_{time_limit}.sol'
        trace_filename = f'{filename.split("/")[-1]}_{algorithm}_{time_limit}.trace'

        with open(solution_filename, 'w') as file:
            file.write(f'{best_value}\n')
            file.write(','.join([str(int(item)) for item in best_solution]))
            print(f"Best value: {best_value}\n")

        with open(trace_filename, 'w') as file:
            for t in trace:
                file.write(f'{t[0]}, {t[1]}\n')
    else:
        print("Invalid algorithm specified.")
        sys.exit(1)


if __name__ == "__main__":
    main()
