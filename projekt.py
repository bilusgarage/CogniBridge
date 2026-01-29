import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor

# 1. Prepare simple synthetic data: y = 2x + 1
def get_data(num, w=2.0, b=1.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)

# 2. Define the Network
class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1) # 1 input, 1 output

    def construct(self, x):
        return self.fc(x)

# 3. Training logic
def train():
    # Initialize network, loss function, and optimizer
    net = LinearNet()
    loss_fn = nn.MSELoss()
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)

    # Define forward function for automatic differentiation
    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss

    # Get gradient function
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    # Simple training loop
    data_gen = get_data(200)
    print("Starting training...")

    for x, y in data_gen:
        x_tensor, y_tensor = Tensor(x), Tensor(y)
        loss, grads = grad_fn(x_tensor, y_tensor)
        optimizer(grads)

    # 4. Check results
    weight = net.fc.weight.asnumpy()
    bias = net.fc.bias.asnumpy()
    print(f"Result: y = {weight[0][0]:.2f}x + {bias[0]:.2f}")

if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    train()