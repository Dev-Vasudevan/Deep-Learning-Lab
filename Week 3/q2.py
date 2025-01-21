from q1_q3 import Linear_Regressor
import torch
import matplotlib.pyplot as plt


x = torch.tensor([2.0,4.0])
y = torch.tensor([20.0,40.0])

model = Linear_Regressor(x,y,lr=0.001)
model.fit(100,info=True)
