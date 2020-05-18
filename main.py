#!/usr/bin/env python 

import sys, time, random, csv
import numpy as np
import matplotlib.pyplot as plt
from QTable import *
from Qapprox import *

def help(text):
	print(text)
	print('\n-----RL Truck delivery Problem Help-----\n')
	print('Arguments : [TruckCapacity] [RoadLength] [TruckStartPenalty] [TimeSteps] [Table(T) or Neural Network(N) for Q function]')
	print('TruckCapacity\t\t: Capacity of the truck')
	print('RoadLength\t\t: Road Length')
	print('TruckStartPenalty\t: Penalty for starting the truck')
	print('TimeSteps\t\t: Timesteps to run the training for')
	print('\n-----End Help-----\n')
	sys.exit()

if __name__ == "__main__" :
	if len(sys.argv) != 6: 
		help("ERROR : Incorrect number of arguments")

	else : 
		T = truck(int(sys.argv[1]))
		W = warehouse()
		if sys.argv[5] == 'T' :
			Q = QTable_learn(T,W,int(sys.argv[2]),float(sys.argv[3]),1.0,0.999)
			Q.showConfig()
			Q.learner(int(sys.argv[4]))
			Q.showPolicy()

		elif sys.argv[5] == 'N' :
			Q = Qapprox_learn(T,W,int(sys.argv[2]),float(sys.argv[3]),1.0,0.995)
			Q.showConfig()
			Q.learner(int(sys.argv[4]))
			Q.showPolicy()

		print("Testing the learned policy for 10000 timesteps")
		T.packages = []
		T.location = 0
		W.packages = []
		W.prob = 0.15
		Q.eps = Q.eps*0.0
		Q.alpha = 0.0
		Q.reward = 0.0
		Q.overallReward = 0.0
		Q.overallRewardRec = []
		Q.time = 0
		Q.learner(10000)
		print("Total reward received =",Q.overallReward)
		print("Average reward per timestep =",Q.overallReward/10000)

		fig, axs = plt.subplots(1,1,figsize=(8, 8))
		axs.plot(np.array(Q.overallRewardRec)[:,0],np.array(Q.overallRewardRec)[:,1])
		axs.set_title('Reward received over time with the learned policy')
		axs.set_xlabel('Timesteps')
		axs.set_ylabel('Reward')
		plt.show()