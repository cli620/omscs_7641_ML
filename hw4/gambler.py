
import numpy as np

from gym.envs.toy_text import discrete


class GamblerEnv(discrete.DiscreteEnv):
    """

    """

    def __init__(self, size=1000, ph=0.4):

        nS = size + 1
        nA = size // 2 + 1

        self.nrow = nS
        self.ncol = 1

        isd = np.zeros(nS)
        isd[1] = 1.

        P = {s: {a: [] for a in range(nA)} for s in range(nS)} # start off uniformed.

        for state in range(nS - 1):
            for action in range(nA):

                if action == 0:
                    P[state][action].append((1.0, 0, 0.0, True))
                elif action <= min(state, nS - 1 - state):
                    new_state = state + action
                    done = new_state == nS - 1
                    reward = float(done)
                    # reward = new_state #action/nS
                    reward = action/min(state, nS - 1 - state)
                    P[state][action].append((ph, new_state, reward, done))

                    new_state = state - action
                    done = new_state <= 0
                    reward = 0.0
                    # if done:
                    #     reward = 0.0
                    # else:
                    #     reward = new_state
                    # reward = new_state #-action/nS
                    reward = -action/min(state, nS - 1 - state)
                    P[state][action].append((1.-ph, new_state, reward, done))
                # else:
                    # keep the current state because this other state isn't possible
                    #
                    # P[state][action].append((1.0, state, 0.0, False))


        super(GamblerEnv, self).__init__(nS, nA, P, isd)

    def render(self):

        print('Last Stake = {}'.format(self.last_action))
        print('Capital = {}'.format(self.s))
