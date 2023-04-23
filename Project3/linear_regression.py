import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Build the dataset with sk-learn
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
x_numpy_f32 = x_numpy.astype(np.float32)
y_numpy_f32 = y_numpy.astype(np.float32)
x = torch.from_numpy(x_numpy_f32)
y = torch.from_numpy(y_numpy_f32)
y = y.view(y.shape[0],1)  # to transform y from 1D to 2D tensor  
n_samples, n_features = x.shape

# Create the model with nn
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# Define the optimizer and the loss
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()

# Training loop
n_epochs = 500
for epoch in range(n_epochs):
    predicted_y = model(x)
    loss = criterion(predicted_y, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch%10==0:
        print(f"epoch:{epoch+1}, loss = {loss.item()}:.4f")

# Plot
y_predicted_numpy = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, y_predicted_numpy, 'b')
plt.show()


