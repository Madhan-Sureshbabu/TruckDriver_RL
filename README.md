## Robotic Truck Driver -- Q-learning

	[Problem Statement](https://docs.google.com/document/d/113ojO8I44v5r96taZW5eKPBlE2nNwCP21CJwEgtKrFo/edit?usp=sharing)  
	
	[Report](https://docs.google.com/document/d/1zltb8kvZDBRrhLYiEg6sQSOBq6Z_Tw83eeF9PTLiLXY/edit?usp=sharing)  

### Usage

Arguments : [TruckCapacity] [RoadLength] [TruckStartPenalty] [TimeSteps] [Table(T) or Neural Network(N) for Q function]  

```bash
python3 main.py 30 25 -250 500000 N  
```
This learns the Q-function using a neural network  

```bash
python3 main.py 30 25 -250 500000 T  
```
This learns the Q-function using a Q-table  



