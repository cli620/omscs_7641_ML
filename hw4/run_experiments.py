# Script that runs the experiment
import gym
import sys
import pybox2d.Box2D
from mdp import MDP
from gym.envs.toy_text.frozen_lake import generate_random_map


sys.path.append(' .')
gym.envs.register('Gambler-v0', entry_point='gambler:GamblerEnv', max_episode_steps=1000)
gym.envs.register('Cliff-v0', entry_point='cliff_walking:WindyCliffWalkingEnv', max_episode_steps=1000)
gym.envs.register('CustomRewardedFrozenLake-v0', entry_point='custom_frozen_lake:RewardingFrozenLakeEnv', max_episode_steps=1000)
if __name__ == '__main__':

    frozen_flag = True
    gamble_flag = False
    if frozen_flag:
        vpq_flag = [False, False, True]
        n = 25
        # random_map = generate_random_map(size=n, p=0.8)
        my_map =['SFHHFHHHHHHHHHFFFFFFHFFFH',
                 'FFFFFFFFFFHFFFFFFFHFFFFHF',
                 'FFHFFFFHHFHFFFHHFFFFFFFFF',
                 'FFFFFFFFFFHFFFFFFFFFFHFFF',
                 'HFFFFFFFFFFFFFFFFHFFFFFF',
                 'FFFFFFFFFFFFFFFFFFFFFFFHF',
                 'FHFFHFFFFFHFFFFFFFFHFHHHH',
                 'FFFFFHFFFHFFHFFFFFFFFHFHF',
                 'FFFHFFHFFFFFHHFFHFFFFFFFF',
                 'FHHHHHHFHHHFFFFFFHHHFHFFF',
                 'FHHHFFFFFFFFFFFFFFFFHFHFH',
                 'FHFHHFHFFHFFHFFFFFFFHFFFF',
                 'HFFFFFFFFFFHFFFFFFFHFFFFH',
                 'FFFFFFFFFFFHFHFFFFFFHHFHF',
                 'FFHFHFFFFHFHFHFFFFFFFFFFF',
                 'HFFFFFFFFHFFFFFFFFFFFHFHF',
                 'FFFHFFFFFFHFHFFFFFHFHHFFF',
                 'HFFFFFHFHHHFFFHFFFFHFFFFH',
                 'FFFFFHFFFHFFFFHFFFHFFFFHF',
                 'HFFFFFFFFFFFFFFFFFFHFHFFF',
                 'FFFHFFFFFFFHFFFHFFFFHHFFF',
                 'FFFFFFFFFFHFFHHFFFFFFFFFF',
                 'FFFFFFFFFFFFFFFFFHFFFFFHF',
                 'FHHHHFFFHFFFFFFHFFHFFFFFF',
                 'HHHFHHFHFHHFFFFFFFFFFFFFG']
        # frozen_lake_env = gym.make('FrozenLake-v0',desc=random_map).unwrapped
        # Q_design of experiments
        holerewards = [-2]# [-0.5, -1, -2, -5, -50, -1] #[-2, -1, -2, -2, -2, -2, -2, -2, -2, -2]
        Edecay = [.999]#[.99, .99, .99, .99, .99, .999] #[.99, .99, .999, .95, .9, .8, .99, .99, .99, .99]
        Adecay = [.99]#[.99, .99, .99, .99, .99, .99] #[.99, .99, .99, .99, .99, .99, .999, .95, .9, .8]
        num_ep = [100000]#[50000, 50000, 50000, 50000, 50000, 100000] #120000
        for i in range(len(Adecay)):
            mapname = 'FrozenLake_experiment'+str(i) #'FrozenLake'+str(n)+'x'+str(n)
            print('-'*5, mapname, ' holeRew: ', holerewards[i], ' | eps decay: ', Edecay[i], ' | learn rate decay: ', Adecay[i])
            frozen_lake_env = gym.make('CustomRewardedFrozenLake-v0',desc=my_map, map_name='25x25', rewarding=True, step_reward=-0.1, hole_reward=holerewards[i], is_slippery=False).unwrapped
            # frozen_lake_env.render()
            frozen_lake = MDP(env_name=mapname, environment=frozen_lake_env, convergence_threshold=0.0001, grid=True, max_iterations=1000)
            # print('Frozen Lake ', n, ' x ', n)
            if vpq_flag[0]:
                frozen_lake.value_iteration(iterations_to_save=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], visualize=True) # converges at 34
            if vpq_flag[1]:
                frozen_lake.policy_iteration(iterations_to_save=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],visualize=True) # converged at 14 here.
            if vpq_flag[2]:
                # frozen_lake.Q_learning(num_episodes=100000, learning_rate_decay=0.99, epsilon_decay=0.95, visualize=True)
                frozen_lake.Q_learning(num_episodes=num_ep[i], learning_rate_decay=Adecay[i], epsilon_decay=Edecay[i], visualize=True)

    if gamble_flag:
        vpq_flag = [False, False, True]
        print('Gambler')
        if vpq_flag[2]:
            Edecay = [.9]#[.99, .99, .999, .95, .9, .8, .99, .99, .99, .99]
            Adecay = [.9999]#[.99, .99, .99, .99, .99, .99, .999, .95, .9, .8]
            num_ep = 5000000
            for i in range(len(Adecay)):
                env_name = 'Gambler_experiment' + str(i)
                print('-'*5, env_name, ' | eps decay: ', Edecay[i], ' | learn rate decay: ', Adecay[i])
                gamble_env = gym.make('Gambler-v0', size=100).unwrapped
                gambler = MDP(env_name=env_name, environment=gamble_env, convergence_threshold=0.0001, grid=False, max_iterations=1e4)
                gambler.Q_learning(num_episodes=num_ep, learning_rate_decay=Adecay[i], epsilon_decay=Edecay[i], visualize=True)
        else:
            gamble_env = gym.make('Gambler-v0', size=100).unwrapped
            gambler = MDP(env_name='Gambler-v0', environment=gamble_env, convergence_threshold=0.0001, grid=False, max_iterations=1e4)
            if vpq_flag[0]:
                gambler.value_iteration(iterations_to_save=[1, 2, 4, 6, 8, 10, 16, 32, 64, 100, 200, 400, 800, 1600, 3200, 6400, 10000,12800], visualize=True)
            if vpq_flag[1]:
                gambler.policy_iteration(iterations_to_save=[1, 2, 4, 6, 8, 10, 16, 32, 64, 100, 128, 256, 512, 1024, 2048, 4096, 8192,16384, 32768], visualize=True)