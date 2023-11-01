import torch
import time

device = torch.device('cuda')

model = torch.nn.Linear(1024, 1024).to(device)
model = model.half() # 用半精度加速
optimizer = torch.optim.Adam(model.parameters()) 

torch.backends.cudnn.benchmark = True

batch_size = 32768

while True:
    inputs = torch.randn(batch_size, 1024, dtype=torch.float16, device=device)
    targets = torch.randn(batch_size, 1024, dtype=torch.float16, device=device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize() 
    time.sleep(0.0005)