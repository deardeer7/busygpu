import torch
import time

device = torch.device('cuda') 

model = torch.nn.Linear(1024, 1024).to(device)
optimizer = torch.optim.Adam(model.parameters())

while True:
    inputs = torch.randn(4096, 1024, device=device) 
    targets = torch.randn(4096, 1024, device=device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    optimizer.step()
    
    time.sleep(0.01)