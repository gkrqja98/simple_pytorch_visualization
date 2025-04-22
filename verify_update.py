import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=2, bias=False)
        self.pool1 = nn.MaxPool2d(2, stride=1)
        self.fc    = nn.Linear(4, 2)

        # 고정된 초기값
        self.conv1.weight.data = torch.tensor([[[[1.0, 0.5],
                                                 [0.5, 1.0]]]])
        self.fc.weight.data   = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                              [0.4, 0.3, 0.2, 0.1]])
        self.fc.bias.data     = torch.tensor([0.1, -0.1])

    def forward(self, x):
        return self.fc(self.pool1(F.relu(self.conv1(x))).view(x.size(0), -1))

if __name__ == "__main__":
    lr = 0.1
    model = SimpleCNN()

    # 더 비교하기 쉽게 파라미터 복사
    old_conv_w = model.conv1.weight.data.clone()
    old_fc_w   = model.fc.weight.data.clone()
    old_fc_b   = model.fc.bias.data.clone()

    # 입력과 타깃
    x = torch.arange(16., dtype=torch.float32).reshape(1,1,4,4)
    target = torch.tensor([1])

    # 순전파-역전파
    conv_out = model.conv1(x);     conv_out.retain_grad()
    relu_out = F.relu(conv_out);   relu_out.retain_grad()
    pool_out = model.pool1(relu_out); pool_out.retain_grad()
    flatten  = pool_out.view(1, -1); flatten.retain_grad()
    fc_out   = model.fc(flatten);  fc_out.retain_grad()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    optimizer.zero_grad()
    loss = criterion(fc_out, target)
    loss.backward()
    
    # 1) FC 출력에 대한 gradient (δy)
    print("fc_out.grad (∂L/∂y):\n", fc_out.grad)

    # 2) flatten에 대한 gradient (∂L/∂x_flatten)
    print("\nflatten.grad (∂L/∂flatten):\n", flatten.grad)

    # 3) 기대하는 gradient: δy @ W (shape [1,2] @ [2,4] → [1,4])
    expected_flat_grad = torch.matmul(fc_out.grad, model.fc.weight.data)
    print("\nexpected flatten grad (δy·W):\n", expected_flat_grad)

    # 4) pool_out에 대한 gradient (reshape 전)
    print("\npool_out.grad (reshaped back):\n", pool_out.grad)

    # 5) flatten.grad을 pool_out 모양으로 다시 변환해 보기
    print("\nflatten.grad.view_as(pool_out):\n", flatten.grad.view_as(pool_out))

    # 파라미터 업데이트
    optimizer.step()

    # 그래디언트와 실제 업데이트 비교
    def compare(name, old, grad, new):
        expected_delta = -lr * grad
        actual_delta   = new - old
        print(f"--- {name} ---")
        print("grad:\n", grad)
        print("expected Δ (−lr·grad):\n", expected_delta)
        print("actual Δ (new − old):\n", actual_delta)
        print("difference (actual − expected):\n", actual_delta - expected_delta, "\n")

    compare("conv1.weight",
            old_conv_w,
            model.conv1.weight.grad,
            model.conv1.weight.data)

    compare("fc.weight",
            old_fc_w,
            model.fc.weight.grad,
            model.fc.weight.data)

    compare("fc.bias",
            old_fc_b,
            model.fc.bias.grad,
            model.fc.bias.data)
