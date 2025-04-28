import torch

def train(model, loader, optimizer, criterion, device, print_every=500):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (X_batch, y_batch) in enumerate(loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        c = (predicted == y_batch).sum().item()
        correct += c

        if i % print_every == 0:
            print(f"[Batch {i}/{len(loader)}] Train Loss: {loss.item():.4f} | Train Accuracy: {c/y_batch.size(0):.2f}")
                    
    avg_loss = running_loss / len(loader.dataset)
    avg_accuracy = correct / total

    return avg_loss, avg_accuracy



def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()
    return running_loss / len(loader.dataset), correct / total