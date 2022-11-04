import torch
import gpu_rpma
from network import SBNetwork

def main():
    m, n = 128, 128
    n_inputs, batch_size, n_epochs = 256, 16, 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(19260817)
    input_data = torch.randn((n_inputs, 1, m, n))
    labels = torch.randn((n_inputs, 10)).to(device=device)
    loss_fn = torch.nn.SmoothL1Loss()
    model = SBNetwork(m, n)
    model = model.to(device=device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # A fake training process, 这P也学不到, 还是SB
    gpu_rpma.print_network(model.state_dict())
    for epoch in range(n_epochs):
        print("="*50)
        for i in range(n_inputs//batch_size):
            print(f"iteration {i}")
            optimizer.zero_grad()
            data_batch = input_data[i*batch_size : (i + 1)*batch_size]
            pred = model(data_batch)
            tgt = labels[i*batch_size : (i + 1)*batch_size]
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    main()
