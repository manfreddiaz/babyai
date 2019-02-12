import numpy as np
import torch
import gym

import babyai.utils as utils
import babyai.model as models
from babyai.bot import Bot
from babyai.utils import ModelAgent


class InteractiveIIL:

    def __init__(self, args):
        self.args = args

        # seeding
        utils.seed(args.seed)

        self.env = gym.make(id=args.env)

        self.episodes = 300  # args.episodes
        self.horizon = self.env.max_steps
        self.initial_decay = 0.99  # args.decay

        self.observation_preprocessor = utils.ObssPreprocessor(model_name=args.model,
                                                               obs_space=self.env.observation_space,
                                                               load_vocab_from=getattr(self.args, 'pretrained_model', None))
        # TODO: for now I am only running the small model
        self.model = models.ACModel(obs_space=self.env.observation_space,
                                    action_space=self.env.action_space)
        self.learner = ModelAgent(model_or_name=self.model,
                                  obss_preprocessor=self.observation_preprocessor,
                                  argmax=True)
        self.teacher = Bot(self.env)

        self.data = []

        self.observation_preprocessor.vocab.save()
        utils.save_model(self.model, args.model)

        self.model.train()
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr, eps=self.args.optim_eps)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cpu':
            print('running on cpu...')

    def train(self):
        for episode in range(self.episodes):
            alpha = self.initial_decay ** episode

            observation = self.env.reset()
            last_action = None

            done = False
            while not done:
                active_agent = np.random.choice(
                    a=[self.teacher, self.learner],
                    p=[alpha, 1. - alpha]
                )
                optimal_action = self.teacher.replan(action_taken=last_action)
                if active_agent == self.teacher:
                    action = optimal_action
                else:
                    action = self.learner.act(observation)

                next_observation, reward, done, info = self.env.step(action)

                self.data.append([observation, optimal_action, done])
                last_action = action
                observation = next_observation

            self._train_epoch()

    def _train_epoch(self):
        batch_size = self.args.batch_size
        data_set_size = len(self.data)

        # NOTE: this is a really smart idea
        randomized_indexes = np.arange(0, len(self.data))
        np.random.shuffle(randomized_indexes)

        for index in range(0, data_set_size, batch_size):
            batch = [self.data[i] for i in randomized_indexes[index:index + batch_size]]
            _log = self._train_batch(batch)

    def _train_batch(self, batch):
        pass

