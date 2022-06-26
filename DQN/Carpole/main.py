import time
import numpy as np
import torch
import gym
from DQN import DQN_Agent,ReplayBuffer,device
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from utils import evaluate_policy,str2bool

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=250*1000, help='which model to load')

parser.add_argument('--seed', type=int, default=532, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=1e6, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=5e4, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=1e3, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=3e3, help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
parser.add_argument('--DDQN', type=str2bool, default=True, help='True:DDQN; False:DQN')
opt = parser.parse_args()
print(opt)


def main():
    EnvName = ['CartPole-v1','LunarLander-v2']
    BriefEnvName = ['CPV1', 'LLdV2']
    Env_With_DW = [True, True] #DW: Die or Win
    EnvIdex = opt.EnvIdex
    env_with_dw = Env_With_DW[EnvIdex]
    env = gym.make(EnvName[EnvIdex])
    eval_env = gym.make(EnvName[EnvIdex])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_e_steps = env._max_episode_steps


    #Use DDQN or DQN
    if opt.DDQN: algo_name = 'DDQN'
    else: algo_name = 'DQN'

    seed = opt.seed
    torch.manual_seed(seed)
    env.seed(seed)
    eval_env.seed(seed)
    np.random.seed(seed)

    print('Algorithm:',algo_name,'  Env:',BriefEnvName[EnvIdex],'  state_dim:',state_dim,'  action_dim:',action_dim,'  Random Seed:',seed, '  max_e_steps:',max_e_steps)
    print('\n')

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}_{}'.format(algo_name,BriefEnvName[EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    kwargs = {
        "env_with_dw":env_with_dw,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": opt.gamma,
        "hid_shape": (opt.net_width,opt.net_width),
        "lr": opt.lr,
        "batch_size":opt.batch_size,
        "exp_noise":opt.exp_noise,
        "double_dqn": opt.DDQN
    }
    if not os.path.exists('model'): os.mkdir('model')
    model = DQN_Agent(**kwargs)
    if opt.Loadmodel: model.load(algo_name,BriefEnvName[EnvIdex],opt.ModelIdex)

    buffer = ReplayBuffer(state_dim, max_size=int(1e6))

    if opt.render:
        score = evaluate_policy(eval_env, model, True, 20)
        print('EnvName:', BriefEnvName[EnvIdex], 'seed:', seed, 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, done, ep_r, steps = env.reset(), False, 0, 0
            while not done:
                steps += 1  #steps in current episode
                if buffer.size < opt.random_steps:
                    a = env.action_space.sample()
                else:
                    a = model.select_action(s, deterministic=False)
                s_prime, r, done, info = env.step(a)

                '''Avoid impacts caused by reaching max episode steps'''
                if (done and steps != max_e_steps):
                    if EnvIdex == 1:
                        if r <= -100: r = -10  # good for LunarLander
                    dw = True  # dw: dead and win
                else:
                    dw = False

                buffer.add(s, a, r, s_prime, dw)
                s = s_prime
                ep_r += r

                '''update if its time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        model.train(buffer)

                '''record & log'''
                if (total_steps) % opt.eval_interval == 0:
                    model.exp_noise *= opt.noise_decay
                    score = evaluate_policy(eval_env, model, render=False)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('noise', model.exp_noise, global_step=total_steps)
                    print('EnvName:',BriefEnvName[EnvIdex],'seed:',seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)
                total_steps += 1

                '''save model'''
                if (total_steps) % opt.save_interval == 0:
                    model.save(algo_name,BriefEnvName[EnvIdex],total_steps)
    env.close()

if __name__ == '__main__':
    main()

# Train from scratch
# run 'python main.py', where the default enviroment is CartPole-v1.
#
# Play with trained model
# run 'python main.py --write False --render True --Loadmodel True --ModelIdex 50000'
#
# Change Enviroment
# If you want to train on different enviroments, just run 'python main.py --EnvIdex 1'.
# The --EnvIdex can be set to be 0 and 1, where
# '--EnvIdex 0' for 'CartPole-v1'
# '--EnvIdex 1' for 'LunarLander-v2'
#
# Visualize the training curve
# You can use the tensorboard to visualize the training curve. History training curve is saved at '\runs'