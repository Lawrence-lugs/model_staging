import numpy as np 
import datasets
from torchvision import transforms
import torch
from models import mbv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def t_cropflip_augment_apply(examples):
    examples['img'] = [t_cropflip_augment(image) for image in examples['img']]
    return examples
    
def t_normalize_apply(examples):
    examples['img'] = [t_normalize(image) for image in examples['img']]
    return examples

t_cropflip_augment = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

t_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = datasets.load_dataset("uoft-cs/cifar10", split="train")
test_set = datasets.load_dataset("uoft-cs/cifar10", split="test")

train_set.set_transform(t_cropflip_augment_apply)
test_set.set_transform(t_normalize_apply)


def sup_train(model, train_loader, test_loader, test_set, optimizer, criterion, learning_epochs):
    for epoch in range(learning_epochs):
        model.train()
        epochscore = 0
        runloss = 0
        for indic in test_loader:
            model.train()
            inputs = indic["img"].to(device)
            labels = indic["label"].to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs,labels)

            loss.backward()
            optimizer.step()

            runloss += loss
            
            _, preds = torch.max(outputs,1)
            epochscore += torch.sum(preds == labels.data)

            model.eval()
            
        with torch.no_grad():
            _,accuracy = test(model, test_loader, criterion, test_set)
            print("Epoch: ", epoch, " Loss: ", runloss, " Accuracy: ", accuracy)

def test(model, test_loader, criterion, test_set):       
    ''' Tests the accuracy of the dp model on testset ''' 
    model.eval()
    loss = 0
    total_correct=0
    for indic in test_loader:
        inputs = indic["img"].to(device)
        labels = indic["label"].to(device)
        outputs = model(inputs)
        loss += criterion(outputs, labels).item()
        _, preds = torch.max(outputs,1)
        total_correct += torch.sum(preds == labels.data)
    accuracy = total_correct/len(test_set)
    return loss,accuracy


learning_epochs = 100
batch_size = 64

model = mbv2.MobileNetV2(
    num_classes=10,
    width_mult=1.0
).to(device)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False
)

sup_train(
    model=model,
    train_loader=train_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    criterion=torch.nn.CrossEntropyLoss(),
    learning_epochs=learning_epochs,
    test_loader=test_loader,
    test_set = test_set
)

torch.save(model.state_dict(), "saved_models/mbv2_cifar10.pt")