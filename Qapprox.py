#!/usr/bin/env python 
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

class DQN(nn.Module) :
	def __init__(self,input_size, output_size) :
		super().__init__()

		self.fc1 = nn.Linear(in_features = input_size, out_features=4)
		self.fc2 = nn.Linear(in_features = 4, out_features = 4)
		self.out = nn.Linear(in_features = 4, out_features = output_size)

	def forward(self,t) :
		t = t.float()
		t = F.logsigmoid(self.fc1(t))
		t = F.logsigmoid(self.fc2(t))
		t = self.out(t)
		return t

Experience = namedtuple('Experience', ('state','action','next_state','reward'))

class ReplayMemory():
	def __init__(self,capacity) : 
		self.capacity = capacity
		self.memory = []
		self.push_count = 0

	def push(self,experience) :
		if len(self.memory) < self.capacity :
			self.memory.append(experience)
		else :
			self.memory[self.push_count % self.capacity] = experience
		self.push_count += 1

	def sample(self,batch_size) :
		return random.sample(self.memory, batch_size)

	def can_provide_sample(self,batch_size) :
		return len(self.memory) >= batch_size

class QValues():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	@staticmethod
	def get_current(policy_net,states,actions):
		states = torch.squeeze(states)
		return policy_net(states).gather(dim=1,index=actions)

	@staticmethod        
	def get_next(target_net, next_states):                
		batch_size = next_states.shape[0]
		values = torch.zeros(batch_size).to(QValues.device)
		values = target_net(next_states).max(dim=1)[0]
		return values

class Qapprox_learn:
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
		# self.alphaMin = 0.05
		# self.alphaDecay = 1.0
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

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     

		self.memory_size = 1000000
		self.batch_size = 100
		self.memory = ReplayMemory(self.memory_size)
		self.policy_net = DQN(2,2).to(self.device)
		self.target_net = DQN(2,2).to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.target_net.eval()
		self.optimizer = optim.Adam(params = self.policy_net.parameters(), lr = 0.0001, weight_decay=0.01)
		self.target_net_update = 500
		self.last_target_net_update = 0

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
				state = torch.tensor([i,j], device= self.device)
				if (i==0 and j==0) or (i!=0 and j!=0):
					if self.eps[i,j] <= 0.25 or ctrStatesSum[i,j] > 100:
						# Row.append(self.lookup[np.argmax(self.Q[i,j,:])])
						Row.append(self.lookup[self.policy_net(state).argmax()])
					else:
						Row.append("?"+self.lookup[self.policy_net(state).argmax()])
				else:
					Row.append("*")
			print(row_format.format("", *Row))

		print("\nW -> Wait, D->Deliver, ?->Not enough data, *-> Not applicable")
		print("Packages in warehouse =",len(self.W.packages))
		# print(self.Q)
		print("Final epsilon value :\n",np.round(self.eps,4))
		print(self.ctrStates)
		print("-----")

	def discretizeState(self):
		if len(self.T.packages) == 0:
			return([0,0])
		
		state = [len(self.T.packages),np.amax(self.T.packages,0)[0]]
		
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


	def extract_tensors(self,experiences) :
		batch = Experience(*zip(*experiences))
		t1 = torch.stack(batch.state)
		t2 = torch.stack(batch.action)
		t3 = torch.stack(batch.reward)
		t4 = torch.stack(batch.next_state)
		return (t1,t2,t3,t4)


	def learner(self,nIterations):

		for iterations in range(nIterations):
			if iterations % (nIterations/10) == 0:
				print(round(100*iterations/nIterations,0),"%","completed")
				
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
					previous_state_tensor = torch.tensor(prevState, dtype=torch.long, device = self.device)
					action_tensor = torch.tensor(self.action, dtype=torch.long, device = self.device)
					current_state_tensor = torch.tensor(currState, dtype=torch.long, device = self.device)
					reward_tensor = torch.tensor(self.reward,dtype=torch.long, device = self.device)
					self.memory.push(Experience(previous_state_tensor,action_tensor, current_state_tensor,reward_tensor))

				self.overallReward += self.reward
				self.overallRewardRec.append([self.time-1,self.overallReward])
				self.reward = 0

				# Choose an action basen on exploration or exploitation
				explore = np.random.choice([0,1],1,p=[1-self.eps[currState[0],currState[1]],self.eps[currState[0],currState[1]]])
				# Select action based on the mode
				if explore == 1:	# Exploration
					self.action = np.random.choice([0,1],1)
				else:							# Exploitation
					state_tensor = torch.tensor(currState, device = self.device)
					self.action = np.asarray([self.policy_net(state_tensor).argmax().item()])

				# if len(self.T.packages) == self.T.capacity:
				# 	self.action = 1

				# Update the tracking counter
				self.ctrStates[currState[0],currState[1],self.action] += 1

				# Update epsilon
				if np.sum(self.ctrStates[currState[0],currState[1],:]) > 50:
					self.eps[currState[0],currState[1]] *= self.epsDecay
				
			if self.memory.can_provide_sample(self.batch_size):
				experiences = self.memory.sample(self.batch_size)
				states, actions,rewards,next_states = self.extract_tensors(experiences)
				# indices0, indices1 = states[:,0]/100, states[:,1]/100
				# self.T.state_count[indices0, indices1,actions] += 1

				current_q_values = QValues.get_current(self.policy_net,states,actions)
				next_q_values = QValues.get_next(self.target_net,next_states)
				# print(current_q_values.shape)
				# print(next_q_values.unsqueeze(1).shape)
				target_q_values = (next_q_values * self.gamma) + rewards

				# print(current_q_values.shape)
				# print(target_q_values.unsqueeze(1).shape)
				loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				
				if self.time - self.last_target_net_update > self.target_net_update :
					self.target_net.load_state_dict(self.policy_net.state_dict())
					self.last_target_net_update = self.time

				# Record the reward and reset it to 0


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

		# self.showPolicy(self.policy_net)


# if __name__ == "__main__" :
# 	if len(sys.argv) != 5: 
# 		help("ERROR : Incorrent number of arguments")

# 	else : 
# 		T = truck(int(sys.argv[1]))
# 		W = warehouse()
# 		Q = Q_learn(T,W,int(sys.argv[2]),float(sys.argv[3]),1.0,0.995)
# 		Q.showConfig()
# 		Q.learner(int(sys.argv[4]))
# 		Q.showPolicy()

# 		print("Testing the learned policy for 10000 timesteps")
# 		T.packages = []
# 		T.location = 0
# 		W.packages = []
# 		W.prob = 0.15
# 		Q.eps = Q.eps*0.0
# 		Q.alpha = 0.0
# 		Q.reward = 0.0
# 		Q.overallReward = 0.0
# 		Q.overallRewardRec = []
# 		Q.time = 0
# 		Q.learner(10000)
# 		# Q.showPolicy()
# 		print("Total reward received =",Q.overallReward)

# 		fig, axs = plt.subplots(1,1,figsize=(8, 8))
# 		axs.plot(np.array(Q.overallRewardRec)[:,0],np.array(Q.overallRewardRec)[:,1])
# 		axs.set_title('Reward received over time with the learned policy')
# 		axs.set_xlabel('Timesteps')
# 		axs.set_ylabel('Reward')
# 		plt.show()