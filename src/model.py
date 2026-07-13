import torch
import torch.nn as nn
# Step 5: Define CNN model

class CNN(nn.Module):

    def __init__(self):

        super(CNN,self).__init__()


        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )


        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )


        self.pool = nn.MaxPool2d(2,2)


        self.fc1 = nn.Linear(
            32*7*7,
            128
        )


        self.fc2 = nn.Linear(
            128,
            10
        )



    def forward(self,x):

        x = torch.relu(self.conv1(x))

        x = self.pool(x)


        x = torch.relu(self.conv2(x))

        x = self.pool(x)


        x = x.view(
            x.size(0),
            -1
        )


        x = torch.relu(self.fc1(x))

        x = self.fc2(x)


        return x



model = CNN().to(device)


print(model)
