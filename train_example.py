import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model import BoTModel

if __name__ == '__main__':

    ####train_test####
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    d_model, head, embodiment_num, layer_num = 512, 2, 6, 6
    batch_size = 16
    epochs = 10
    total_data_size = 10000
    learning_rate = 0.001

    model = BoTModel(d_model=d_model, head=head, layer_num=layer_num, embodiment_num=embodiment_num, mode='Hard').to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ce = nn.CrossEntropyLoss()

    inputs_body = torch.randn(total_data_size, 1).to(device)  # 몸통 입력
    inputs_arm = torch.randint(0, 4, (total_data_size, 5)).to(device)  # 팔다리 입력
    expert_actions = torch.randint(0, 2, (total_data_size, 5), dtype=torch.long).to(device)  # 타겟행동

    dataset = TensorDataset(inputs_body, inputs_arm, expert_actions)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mask_base = torch.tensor([
        [0, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0]
    ], dtype=torch.float32)
    identity = torch.eye(mask_base.size(-1))
    mask = (mask_base + identity).unsqueeze(0).unsqueeze(0).expand(batch_size, head, -1, -1).to(device)

    # 훈련 루프 시작
    for epoch in range(epochs):
        total_loss_epoch = 0
        for body_sensor, arm_sensor, expert_action in tqdm(data_loader, desc=f"epoch={epoch}"):
            body_sensor, arm_sensor = body_sensor.to(device), arm_sensor.to(device)
            expert_action = expert_action.to(device)

            current_mask = mask[:body_sensor.size(0)]

            outputs = model(body_sensor, arm_sensor, current_mask)
            classification_outputs = torch.stack(outputs[0][1:], dim=1)

            loss_classification = sum(
                ce(classification_outputs[:, i], expert_action[:, i]) for i in range(5)
            )

            optimizer.zero_grad()
            loss_classification.backward()
            optimizer.step()

            total_loss_epoch += loss_classification.item()

        avg_loss = total_loss_epoch / len(data_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.4f}")

    print("Imitation learning training completed.")
