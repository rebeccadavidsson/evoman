################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment

experiment_name = 'test'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment in default mode, with human player and static enemy using random controllers
env = Environment(experiment_name=experiment_name,
                  enemymode='static',
                  playermode='human')
env.play()

