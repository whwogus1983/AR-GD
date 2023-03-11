import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼파라미터 설정
batch_size = 32
learning_rate = 0.001
num_epochs = 10
leaky_slope = 0.01 # LeakyReLU의 기울기

# 데이터셋 불러오기
train_dataset = datasets.ImageFolder(root='train/', 
                                      transform=transforms.ToTensor())
val_dataset = datasets.ImageFolder(root='val/', 
                                    transform=transforms.ToTensor())

# 데이터 로더 생성
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                         batch_size=batch_size, 
                                         shuffle=False)

# 모델 생성
model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        ).to(device)

# 손실 함수와 최적화 알고리즘 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # GPU 사용 가능 시 GPU로 데이터 이동
        images = images.to(device)
        labels = labels.to(device)

        # 모델 예측
        outputs = model(images)

        # 손실 계산
        loss = criterion(outputs, labels)

        # 기울기 초기화 후 역전파 및 가중치 갱신
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 검증 데이터를 사용하여 모델 정확도 계산
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            # GPU 사용 가능 시 GPU로 데이터 이동
            images = images.to(device)
            labels = labels.to(device)

            # 모델 예측
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # 정확도 계산
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 모델 정확도 출력
    print('Epoch [{}/{}], Validation Accuracy: {} %'.format(epoch+1, num_epochs, 100 * correct / total))

    # Re
