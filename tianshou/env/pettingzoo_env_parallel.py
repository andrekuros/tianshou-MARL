# import warnings
# from typing import Any, Dict, List, Tuple
# from abc import ABC
# from gymnasium import spaces
# from pettingzoo.utils.env import ParallelEnv
# #from pettingzoo.utils.conversions import parallel_wrapper_fn
# from pettingzoo.utils.wrappers import BaseParallelWrapper
# from pettingzoo.utils.conversions import parallel_wrapper_fn


# class PettingZooParallelEnv(ParallelEnv, ABC):
    
#     def __init__(self, env: BaseParallelWrapper):
#         super().__init__()
#         self.env = env
#         self.agents = self.env.possible_agents
        
#         # Assuming all agents have equal observation and action spaces
#         self.observation_space: Any = self.env.observation_space(self.agents[0])
#         self.action_space: Any = self.env.action_space(self.agents[0])    
    
#     def reset(self, *args: Any, **kwargs: Any) -> Tuple[dict, dict]:
#         return self.env.reset(*args, **kwargs)

#     def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict[str, Any]]]:
#         observation, rew, term, trunc, info = self.env.step(actions)        
#         return observation, rew, term, trunc, info

#     def close(self) -> None:
#         self.env.close()

#     def seed(self, seed: Any = None) -> None:
#         self.env.seed(seed)

#     def render(self, **kwargs) -> Any:
#         return self.env.render(**kwargs)
    
import warnings
from typing import Any, Dict, List, Tuple
from abc import ABC
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
#from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.utils.wrappers import BaseParallelWrapper


class PettingZooParallelEnv(ParallelEnv, ABC):
    
    def __init__(self, env: BaseParallelWrapper):
        super().__init__()
        self.env = env
        self.agents = self.env.possible_agents        

        # Assuming all agents have equal observation and action spaces
        self.observation_space: Any = self.env.observation_spaces[self.agents[0]]
        self.action_space: Any = self.env.action_spaces[self.agents[0]]


    def reset(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        
        observations, info = self.env.reset(*args, **kwargs)   
        
        return observations, info

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict[str, Any]]]:
        observations, rewards, dones, infos = self.env.step(actions)
        return observations, rewards, dones, infos

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Any = None) -> None:
        self.env.seed(seed)

    def render(self, **kwargs) -> Any:
        return self.env.render(**kwargs)
    
        


