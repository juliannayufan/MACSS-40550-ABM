import math
import random
import networkx as nx

import mesa
from mesa import Model
from agents import State, VirusAgent

# Define helper functions that get global values for state variables
def number_state(model, state):
    return sum(1 for a in model.grid.get_all_cell_contents() if a.state is state)


def number_infected(model):
    return number_state(model, State.INFECTED)


def number_susceptible(model):
    return number_state(model, State.SUSCEPTIBLE)


def number_resistant(model):
    return number_state(model, State.RESISTANT)


class VirusOnNetwork(Model):
    # Define initiation
    def __init__(
        self,
        num_nodes=10,
        avg_node_degree=3,
        network_type="single",
        initial_outbreak_size=1,
        virus_spread_chance=0.4,
        virus_check_frequency=0.4,
        recovery_chance=0.3,
        gain_resistance_chance=0.5,
        seed=None,
    ):
        if seed is not None:
            seed = int(seed)
        super().__init__(rng=seed)
        random.seed(seed)
        # Set up network: number of nodes, base probability of connection, type of network (binary or weighted)
        self.num_nodes = num_nodes
        self.network_type = network_type
        prob = avg_node_degree / self.num_nodes
        # Set up binary network, creating edges randomly and setting all edge weights to one
        if self.network_type == "single":
            self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=prob)
            for u, v in self.G.edges():
                self.G.edges[u, v]['weight'] = 1              
        # Weighted network. Twice as many edges created, with weights on [0,1] (to get same average weighted degree as binary network)
        elif self.network_type == "weighted":
            self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=2*prob)
            for u, v in self.G.edges():
                self.G.edges[u, v]['weight'] = self.random.random()
        # Raise error if another kind of network is requested
        else:
            raise ValueError("Unsupported network type")

        # Get list of edge weights and node positions; used for visualization
        self.weights = [2*self.G[u][v]['weight'] for u, v in self.G.edges]
        self.position = nx.circular_layout(self.G)

        # Create grid from network object
        self.grid = mesa.space.NetworkGrid(self.G)

        # Initialize other global variables
        self.initial_outbreak_size = (
            initial_outbreak_size if initial_outbreak_size <= num_nodes else num_nodes
        )
        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance

        # Define data collection
        self.datacollector = mesa.DataCollector(
            {
                "Infected": number_infected,
                "Susceptible": number_susceptible,
                "Resistant": number_resistant,
                "R over S": self.resistant_susceptible_ratio,
            }
        )

        # Create agents
        for node in self.G.nodes():
            a = VirusAgent(
                self,
                State.SUSCEPTIBLE,
                self.virus_spread_chance,
                self.virus_check_frequency,
                self.recovery_chance,
                self.gain_resistance_chance,
            )

            # Add the agent to the node
            self.grid.place_agent(a, node)

        # Assign initially infected nodes
        infected_nodes = self.random.sample(list(self.G), self.initial_outbreak_size)
        for a in self.grid.get_cell_list_contents(infected_nodes):
            a.state = State.INFECTED

        self.running = True
        self.datacollector.collect(self)

    # Helper function for DataCollector
    def resistant_susceptible_ratio(self):
        try:
            return number_state(self, State.RESISTANT) / number_state(
                self, State.SUSCEPTIBLE
            )
        except ZeroDivisionError:
            return math.inf

    # Agents take a step, then data is collected
    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
