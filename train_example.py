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
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    inputs_body = torch.randn(total_data_size, 1).to(device) # 몸통 입력
    inputs_arm = torch.randint(0, 4, (total_data_size, 5)).to(device) # 팔다리 입력
    target_body = torch.randn(total_data_size, 1).to(device) # 몸통 타겟
    target_arm = torch.randint(0, 2, (total_data_size, 5, 2), dtype=torch.float32).to(device) # 팔 타겟

    dataset = TensorDataset(inputs_body, inputs_arm, target_body, target_arm)
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

    for epoch in range(epochs):
        total_loss_epoch = 0
        for body_sensor, arm_sensor, target_body, target_arm in tqdm(data_loader,desc="epoch={}".format(epoch)):
            body_sensor, arm_sensor = body_sensor.to(device), arm_sensor.to(device)
            target_body, target_arm = target_body.to(device), target_arm.to(device)

            # Generate mask for the encoder
            current_mask = mask[:body_sensor.size(0)]

            # Forward pass
            outputs = model(body_sensor, arm_sensor, current_mask)

            regression_output = outputs[0][0]
            classification_outputs = torch.stack(outputs[0][1:], dim=1)

            # Calculate losses
            loss_regression = mse(regression_output, target_body)
            loss_classification = sum(
                ce(classification_outputs[:, i], target_arm[:, i]) for i in range(4)
            )
            total_loss = loss_regression + loss_classification

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.item()

        avg_loss = total_loss_epoch / len(data_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.4f}")

    print("Training completed.")
