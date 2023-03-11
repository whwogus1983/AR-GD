import torch.optim as optim
import torch.nn.functional as F

def train(model, optimizer, train_loader, test_loader, num_epochs):
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(num_epochs):
        # train mode
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # eval mode
        model.eval()
        test_loss = 0.0
        test_total = 0
        test_correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100 * test_correct / test_total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # print progress
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%, Test Loss: {:.4f}, Test Acc: {:.2f}%'
              .format(epoch+1, num_epochs, train_loss, train_acc, test_loss, test_acc))

    return train_losses, train_accs, test_losses, test_accs

num_epochs = 10
learning_rate = 0.001

model = Net()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses, train_accs, test_losses, test_accs = train(model, optimizer, train_loader, test_loader, num_epochs)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

'''
이 코드는 10 에폭(epoch) 동안 데이터셋을 학습시키고, 
학습 중에는 최적화 알고리즘으로 확률적 경사하강법(SGD)을 사용합니다. 
학습 속도(learning rate)는 0.001로 설정하고, 모멘텀(momentum)은 0.9로 설정합니다.

매 에폭마다 모델이 데이터셋을 한 번 학습하는데, 
각 미니배치(minibatch)에서 입력 데이터와 레이블을 가져와서 경사하강법 단계를 실행합니다.
이 단계에서는 다음과 같은 일이 일어납니다.

optimizer.zero_grad()를 호출하여 갱신할 가중치의 변화도를 0으로 설정합니다.
net(inputs)를 호출하여 모델의 예측값을 계산합니다.
criterion(outputs, labels)를 호출하여 예측값과 레이블 간의 손실을 계산합니다.
loss.backward()를 호출하여 손실에 대한 모델의 변화도를 계산합니다.
optimizer.step()을 호출하여 가중치를 갱신합니다.
마지막으로, 2000 미니배치마다 현재 손실을 출력합니다. 
10 에폭 동안 학습을 마친 후, "Finished Training"이라는 메시지가 출력됩니다.
'''
