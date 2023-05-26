class logger:
    def __init__(self, env) -> None:
        self.__env__ = env
    
    def print_observation_space(self):
        print("_____OBSERVATION SPACE_____ \n")
        print("Observation Space", self.__env__.observation_space) # (Lower bound, Upper bound, Shape, Data type)
        print("Sample observation", self.__env__.observation_space.sample()) # Get a random observation
        print("Observation Space High", self.__env__.observation_space.high) # Upper bound
        print("Observation Space Low", self.__env__.observation_space.low) # Lower bound
    
    def print_action_space(self):
        print("\n _____ACTION SPACE_____ \n")
        print("Action Space Shape", self.__env__.action_space.n) # Number of actions
        print("Action Space Sample", self.__env__.action_space.sample()) # Take a random action