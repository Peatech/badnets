import torch
from tqdm import tqdm
from sklearn.metrics import classification_report


def train(model, data_loader, criterion, optimizer):
    """
    Training step for the model.
    
    Args:
        model (torch.nn.Module): The model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        
    Returns:
        float: Average training loss over the dataset.
    """
    running_loss = 0.0
    model.train()

    for step, (batch_img, batch_label) in enumerate(tqdm(data_loader, desc="Training")):
        # Move data to the same device as the model
        batch_img, batch_label = batch_img.to(model.device), batch_label.to(model.device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_img)

        # Compute loss
        loss = criterion(outputs, batch_label)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Return average loss
    return running_loss / len(data_loader)


def eval(model, test_loader, report=True):
    """
    Evaluation step for the model.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        report (bool): Whether to print a classification report.
        
    Returns:
        float: Accuracy over the test dataset.
    """
    correct = 0
    total = 0
    preds = []
    gt = []

    with torch.no_grad():
        model.eval()

        for batch_img, batch_label in tqdm(test_loader, desc="Evaluating"):
            # Move data to the same device as the model
            batch_img, batch_label = batch_img.to(model.device), batch_label.to(model.device)

            # Forward pass
            outputs = model(batch_img)

            # Predicted labels
            predicted = torch.argmax(outputs, dim=1)

            # If labels are one-hot encoded, convert to indices
            if batch_label.dim() > 1:
                batch_label = torch.argmax(batch_label, dim=1)

            # Append predictions and ground truths for report
            preds.append(predicted)
            gt.append(batch_label)

            # Count correct predictions
            correct += (predicted == batch_label).sum().item()
            total += batch_label.size(0)

        # Print classification report
        if report:
            gt = torch.cat(gt).cpu()  # Concatenate ground truths and move to CPU
            preds = torch.cat(preds).cpu()  # Concatenate predictions and move to CPU
            print(classification_report(gt, preds))

    # Compute and return accuracy
    return correct / total
