import numpy as np

from mesa import Model
from agents import Boid
from mesa.experimental.continuous_space import ContinuousSpace

class BoidFlockers(Model):
    ## Initialize model, including all relevant parameters. Inherit seed propety from parent class
    def __init__(
        self,
        population_size=100,
        width=100,
        height=100,
        speed=1,
        vision=10,
        separation=2,
        cohere=0.03,
        separate=0.015,
        match=0.05,
        seed=None,
    ):
        if seed is not None:
            seed = int(seed)
        super().__init__(rng=seed)
        # Set up the continuous space
        self.space = ContinuousSpace(
            [[0, width], [0, height]],
            torus=False,
            random=self.random,
            n_agents=population_size,
        )
        # Create arrays of coordinate placements and initial headings of agents
        positions = self.rng.random(size=(population_size, 2)) * self.space.size
        directions = self.rng.uniform(-1, 1, size=(population_size, 2))
        ## Create agents and place them on the map
        Boid.create_agents(
            self,
            population_size,
            self.space,
            position=positions,
            direction=directions,
            cohere=cohere,
            separate=separate,
            match=match,
            speed=speed,
            vision=vision,
            separation=separation,
        )
    ## Define model step
    def step(self):
        self.agents.shuffle_do("flock")
