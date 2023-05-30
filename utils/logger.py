class logger:
    def __init__(self, env) -> None:
        self.__env__ = env

    def print_observation_space(self):
        print("_____OBSERVATION SPACE_____ \n")
        # (Lower bound, Upper bound, Shape, Data type)
        print("Observation Space", self.__env__.observation_space)
        print("Observation Space High",
              self.__env__.observation_space.high)  # Upper bound
        print("Observation Space Low",
              self.__env__.observation_space.low)  # Lower bound
        # Get a random observation
        print("Sample observation", self.__env__.observation_space.sample())

    def print_action_space(self):
        print("_____ACTION SPACE_____ \n")
        print("Action Space", self.__env__.action_space)
        # Number of actions
        print("Action Space Shape", self.__env__.action_space.n)
        # Take a random action
        print("Action Space Sample", self.__env__.action_space.sample())

    def print_obs_action_space(self):
        self.print_observation_space()
        self.print_action_space()
