from pprint import pprint
from typing import Generic, TypeVar

import numpy as np
import torch

from tianshou.config import BasicExperimentConfig, LoggerConfig, RLAgentConfig, RLSamplingConfig
from tianshou.data import Collector
from tianshou.highlevel.agent import AgentFactory
from tianshou.highlevel.env import EnvFactory
from tianshou.highlevel.logger import LoggerFactory
from tianshou.policy import BasePolicy
from tianshou.trainer import BaseTrainer

TPolicy = TypeVar("TPolicy", bound=BasePolicy)
TTrainer = TypeVar("TTrainer", bound=BaseTrainer)


class RLExperiment(Generic[TPolicy, TTrainer]):
    def __init__(self,
            config: BasicExperimentConfig,
            logger_config: LoggerConfig,
            general_config: RLAgentConfig,
            sampling_config: RLSamplingConfig,
            env_factory: EnvFactory,
            logger_factory: LoggerFactory,
            agent_factory: AgentFactory):
        self.config = config
        self.logger_config = logger_config
        self.general_config = general_config
        self.sampling_config = sampling_config
        self.env_factory = env_factory
        self.logger_factory = logger_factory
        self.agent_factory = agent_factory

    def _set_seed(self):
        seed = self.config.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _build_config_dict(self) -> dict:
        return {
            # TODO
        }

    def run(self, log_name: str):
        self._set_seed()

        envs = self.env_factory.create_envs()

        full_config = self._build_config_dict()
        full_config.update(envs.info())

        run_id = self.config.resume_id
        logger = self.logger_factory.create_logger(log_name=log_name, run_id=run_id, config_dict=full_config)

        policy = self.agent_factory.create_policy(envs, self.config.device)
        if self.config.resume_path:
            self.agent_factory.load_checkpoint(policy, self.config.resume_path, envs, self.config.device)

        train_collector, test_collector = self.agent_factory.create_train_test_collector(policy, envs)

        if not self.config.watch:
            trainer = self.agent_factory.create_trainer(policy, train_collector, test_collector, envs, logger)
            result = trainer.run()
            pprint(result)  # TODO logging

        self._watch_agent(self.config.watch_num_episodes, policy, test_collector, self.config.render)

    @staticmethod
    def _watch_agent(num_episodes, policy: BasePolicy, test_collector: Collector, render):
        policy.eval()
        test_collector.reset()
        result = test_collector.collect(n_episode=num_episodes, render=render)
        print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')

