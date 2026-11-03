import torch 

x= torch.tensor(5,dtype=torch.float32, requires_grad=True)
b= torch.tensor(3, dtype=torch.float32, requires_grad=True)
w= torch.tensor(4,dtype=torch.float32, requires_grad=True)

loss_fn = torch.nn.MSELoss()
c_ground = torch.tensor(20,dtype=torch.float32, requires_grad=True)
c= w*x+b

loss = loss_fn(c, c_ground)

loss.backward()

print(w.grad)
print(x.grad)
print(b.grad)


