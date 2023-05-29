import torch

def MOCCT_fit(epoch, model, trainloader, testloader, optim, loss_fn, scheduler):
    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x, x, x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = (y_pred > 0.5).type(torch.int32)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    scheduler.step()

    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x, x, x)
            loss = loss_fn(y_pred, y)
            y_pred = (y_pred > 0.5).type(torch.int32)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    if epoch % 500 == 0:
        print('epoch: ', epoch,
              'loss： ', round(epoch_loss, 3),
              'accuracy:', round(epoch_acc, 3),
              'test_loss： ', round(epoch_test_loss, 3),
              'test_accuracy:', round(epoch_test_acc, 3)
              )
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc


def BP_fit(epoch, model, trainloader, testloader, optim, loss_fn, scheduler):
        correct = 0
        total = 0
        running_loss = 0
        
        for x, y in trainloader:
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            with torch.no_grad():
                y_pred = (y_pred > 0.5).type(torch.int32)
                correct += (y_pred == y).sum().item()
                total += y.size(0)
                running_loss += loss.item()

        scheduler.step()

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = correct / total

        test_correct = 0
        test_total = 0
        test_running_loss = 0

        with torch.no_grad():
            for x, y in testloader:
                if torch.cuda.is_available():
                    x, y = x.to('cuda'), y.to('cuda')
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                y_pred = (y_pred > 0.5).type(torch.int32)
                test_correct += (y_pred == y).sum().item()
                test_total += y.size(0)
                test_running_loss += loss.item()

        epoch_test_loss = test_running_loss / len(testloader.dataset)
        epoch_test_acc = test_correct / test_total

        if epoch % 500 == 0:
            print('epoch: ', epoch,
                  'loss： ', round(epoch_loss, 3),
                  'accuracy:', round(epoch_acc, 3),
                  'test_loss： ', round(epoch_test_loss, 3),
                  'test_accuracy:', round(epoch_test_acc, 3)
                  )
        return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc