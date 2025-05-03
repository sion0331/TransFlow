import torch

def train_validate(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './outputs/transLOB/best_model.pth')
            print(f"✅ Saved best model at epoch {epoch+1} with Val Acc {val_acc:.4f}")

    return history


def train(model, loader, optimizer, criterion, device, print_every=500):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (X_batch, y_batch) in enumerate(loader):
        X_batch, y_batch = X_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.int64) 
        # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
            print(f"[Batch {i}/{len(loader)}] Train Loss: {loss.item():.4f} | Train Accuracy: {c/y_batch.size(0):.4f}")
                    
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
            X_batch, y_batch = X_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.int64) 
            # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()
    return running_loss / len(loader.dataset), correct / total