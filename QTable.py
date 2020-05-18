import sys, time, random, csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T    
from collections import namedtuple
from itertools import count

np.set_printoptions(linewidth=np.inf,suppress=True,threshold=sys.maxsize,precision=5)

class warehouse:
	def __init__(self):
		self.packages = []
		self.prob = 0.15
		self.probMin = 0.05
		self.probMax = 0.25
		self.dprob = 0.02

	def createPackage(self,roadLength,time):
		create = np.random.choice([1,0],1,p=[self.prob,1-self.prob])
		if create == 1:
			houseNo = np.random.randint(1,roadLength+1)
			self.packages.append([houseNo,time])
			self.prob = min(self.prob+self.dprob , self.probMax)
		else:
			self.prob = max(self.prob-self.dprob , self.probMin)

class truck:
	def __init__(self,cap):
		self.packages = []
		self.capacity = cap
		self.location = 0

	def deliver(self):
		nDelivered = 0
		while (len(self.packages) > 0 and self.packages[0][0] == self.location):
			nDelivered += 1
			self.packages.pop(0)

		return(nDelivered)

class QTable_learn:
	def __init__(self,t,w,roadLen,stPen,e,eDec):
		self.time = 0
		self.T = t
		self.W = w

		self.roadLength = roadLen
		self.dircretizeCapacity = 15
		self.dircretizeCapacity = min(self.dircretizeCapacity,self.T.capacity)
		self.CapRange = float(self.T.capacity) / float(self.dircretizeCapacity)
		self.dircretizeRoadLen = 5
		self.dircretizeRoadLen = min(self.dircretizeRoadLen,self.roadLength)
		self.RoadLenRange = float(self.roadLength) / float(self.dircretizeRoadLen)

		self.Q = np.zeros([self.dircretizeCapacity+1,self.dircretizeRoadLen+1,2])
		self.alpha = 0.01
		self.gamma = 0.95
		self.eps = np.ones([self.dircretizeCapacity+1,self.dircretizeRoadLen+1])*e
		# self.eps = e
		self.epsDecay = eDec
		self.startPenalty = stPen
		self.deliverReward = 30*self.roadLength
		self.action = 0
		self.reward = 0
		self.overallReward = 0
		self.overallRewardRec = []

		self.ctrStates = np.zeros([self.dircretizeCapacity+1,self.dircretizeRoadLen+1,2])
		self.lookup = ["W","D"]

	def showConfig(self):
		print("-----Initial Configuration")
		print("Truck Capacity \t\t:",self.T.capacity)
		print("Road Length \t\t:",self.roadLength)
		print("Truck Start Penalty \t:",self.startPenalty)
		print("-----")

	def showPolicy(self):
		print("-----Trained Policy")
		print("Rows : Packages in the truck")
		print("Cols : Maximum house number of packages to be delivered\n")
		row_format ="{:^7}" * (self.dircretizeRoadLen + 3)
		
		ColHeader = [" ","0"]
		for i in range(self.dircretizeRoadLen):
			ColHeader.append("<="+str(int((i+1)*self.RoadLenRange)))
		print(row_format.format("", *ColHeader))
		
		RowHeader = ["0"]
		for i in range(self.dircretizeCapacity):
			RowHeader.append("<="+str(int((i+1)*self.CapRange)))

		ctrStatesSum = np.sum(self.ctrStates,2)
		# percentStates = ctrStatesSum/np.sum(Q.ctrStates)*100

		for i in range(self.dircretizeCapacity+1):
			Row = []
			Row.append(RowHeader[i])
			for j in range(self.dircretizeRoadLen+1):
				if (i==0 and j==0) or (i!=0 and j!=0):
					if self.eps[i,j] <= 0.25 or ctrStatesSum[i,j] > 100:
						Row.append(self.lookup[np.argmax(self.Q[i,j,:])])
					else:
						Row.append("?"+self.lookup[np.argmax(self.Q[i,j,:])])
				else:
					Row.append("*")
			print(row_format.format("", *Row))

		print("\nW -> Wait, D -> Deliver, ? -> Not enough data, * -> Not applicable")
		# print("Packages in warehouse =",len(W.packages))
		# print(self.Q)
		# print("Final epsilon value :\n",np.round(self.eps,4))
		print("-----")

	def discretizeState(self):
		if len(self.T.packages) == 0:
			return([0,0])
		
		state = [len(self.T.packages),np.amax(np.array(self.T.packages)[:,0])]
		# print(state)
		
		discreteState = [0,0]
		for i in range(1,self.dircretizeCapacity+1):
			if state[0] <= i*self.CapRange:
				discreteState[0] = i
				break

		for i in range(1,self.dircretizeRoadLen+1):
			if state[1] <= i*self.RoadLenRange:
				discreteState[1] = i
				break

		return(discreteState)

	def learner(self,nIterations):
		for iterations in range(nIterations):
			if iterations % (nIterations/10) == 0:
				print(round(100*iterations/nIterations,0),"%","completed")
			# Update time
			self.time += 1

			# Create Package
			self.W.createPackage(self.roadLength,self.time)

			# Do the following only when its at warehouse
			# 	Load package
			#		Update current state
			#   Update Q table for previous state
			#		Choose an action
			if self.T.location == 0:
				# Load package
				while(len(self.W.packages) > 0 and len(self.T.packages) <  self.T.capacity):
					self.T.packages.append(self.W.packages.pop(0))

				# Update current state
				currState = self.discretizeState()

				# Update Q table for previous state
				if iterations > 0:
					QPrev = self.Q[prevState[0],prevState[1],self.action]
					QCurrMax = max(self.Q[currState[0],currState[1],:])
					self.Q[prevState[0],prevState[1],self.action] = \
								QPrev + self.alpha*(self.reward + self.gamma*QCurrMax - QPrev)

				# Record the reward and reset it to 0
				self.overallReward += self.reward
				self.overallRewardRec.append([self.time-1,self.overallReward])
				self.reward = 0

				# Choose an action basen on exploration or exploitation
				explore = np.random.choice([0,1],1,p=[1-self.eps[currState[0],currState[1]],self.eps[currState[0],currState[1]]])
				# Select action based on the mode
				if explore == 1:	# Exploration
					self.action = np.random.choice([0,1],1)
				else:							# Exploitation
					self.action = np.argmax(self.Q[currState[0],currState[1],:])

				# if len(self.T.packages) == self.T.capacity:
				# 	self.action = 1

				# Update the tracking counter
				self.ctrStates[currState[0],currState[1],self.action] += 1

				# Update epsilon
				if np.sum(self.ctrStates[currState[0],currState[1],:]) > 50:
					self.eps[currState[0],currState[1]] *= self.epsDecay

			# Add the waiting time penalty for pacakges in warehouse and truck
			for each in self.W.packages:
				self.reward -= (self.time - each[1])

			for each in self.T.packages:
				self.reward -= (self.time - each[1])

			# Apply the action
			if self.action == 1:
				# Add penalty for starting the truck
				if self.T.location == 0:
					self.reward += self.startPenalty
					# Sort packages based on house number to simplify delivery process
					if len(self.T.packages) != 0:
						self.T.packages = sorted(self.T.packages,key=lambda l:l[0])
					# print("Delivery started",self.time,len(T.packages),T.packages)
				
				# Move forward till all packages are delivered
				if len(self.T.packages) != 0:
					self.T.location += 1
					nDelivered = self.T.deliver()
					self.reward += nDelivered*self.deliverReward
					# print(self.T.location,"Delivered = ",nDelivered," Reward received =",self.reward)

				# Return to warehouse after delivery
				elif len(self.T.packages) == 0 and self.T.location > 0:
					self.T.location -= 1

			prevState = currState.copy()

