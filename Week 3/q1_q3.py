import torch
import matplotlib.pyplot as plt
from apt.auth import update

epochs = 100
class Linear_Regressor :
    def __init__(self,x,y,lr=0.001,w=torch.tensor(1.0, requires_grad=True ),b=torch.tensor(1.0, requires_grad=True ) ,func = None):
        self.x = x
        self.y = y
        self.lr = lr
        self.w=w
        self.b = b
        if func is None :
            self.criterion = lambda pred,y: (pred-y)**2
        else :
            self.criterion = func
    def forward(self):
        return self.x * self.w + self.b
    def update(self):
        with torch.no_grad():
            self.w -= self.lr * self.w.grad
            self.b -= self.lr * self.b.grad
    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()
    def test(self,x):
        return self.w* x + self.b
    def train(self,epochs, info=False):
        loss_list = [ ]
        for _ in range(epochs):
            ypred = self.forward()
            loss =  0.0
            for i,item in enumerate(ypred):
                loss+= self.criterion(item,y[i])
            loss /= len(self.x)
            loss_list.append(loss.item())
            loss.backward()
            self.update()

            if info :
                print(f"Epoch : {_+1} => Loss {loss} , w : {self.w}  , b : {self.b} ")
            self.reset_grad()

        self.w = w
        self.b = b
        print(loss_list)
        plt.plot(loss_list)
        plt.show()

if __name__ == "__main__":
    x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                      19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])

    y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                      16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])

    lr = 0.001
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    model = Linear_Regressor(x,y)
    model.train(10,info=True)