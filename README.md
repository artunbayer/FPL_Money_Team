# FPL_Money_Team

Money Team is an AI based Fantasy Premier League Soccer team manager. It uses data on players' statistics from 2015-2020 to form a strong team to enter the season, and also make player transfers along the way. The AI manager has two main components to it: a combinatorial optimizer, and a trained Neural Network.

The optimizer uses the data from previous years to come up with the best team within the budget and player constraints to enter the season. It then uses the Neural Network's predictions about how each player might perform to make a player transfer if it is feasible. 

train2.py contains the code to train the Neural Network.

