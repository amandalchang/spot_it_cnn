import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from model import CLASS_LABEL_DICT


def show_loss_curves(train_loss_csv, test_loss_csv):
    df = pd.read_csv(train_loss_csv)
    train_losses = df[df.columns[0]].values.tolist()
    train_losses.insert(0, float(list(df)[0]))

    df = pd.read_csv(test_loss_csv)
    test_losses = df[df.columns[0]].values.tolist()
    test_losses.insert(0, float(list(df)[0]))

    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(test_losses, label="Test Loss", color="orange")
    plt.title("Training and Test Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()


def calculate_confusion_matrix(predictions, gt_labels, num_classes):
    """
    Calculate the confusion matrix based on predictions and ground truth labels.

    Args:
        predictions (list): List of predicted labels.
        gt_labels (list): List of ground truth labels.
        num_classes (int): The number of classes/labels.

    Returns:
        list: A 2D list representing the confusion matrix.
    """
    # Initialize the confusion matrix with zeros
    # The matrix is a 2D list where rows represent the true classes/labels
    # and columns represent the predicted classes/labels
    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    # Iterate over the pairs of predictions and ground truth labels
    for pred, true in zip(predictions, gt_labels):
        confusion_matrix[true][pred] += 1

    # Return the populated confusion matrix
    return confusion_matrix


def get_predictions_and_labels(model, loader):
    """
    Get predictions and ground truth labels from the model using the data loader.

    Args:
        model (torch.nn.Module): The trained model for making predictions.
        loader (DataLoader): The data loader providing test data in batches.

    Returns:
        tuple: A tuple containing two lists:
            - all_predictions (list): List of predicted labels.
            - all_labels (list): List of ground truth labels.
    """
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model.cpu()(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_predictions, all_labels


def calculate_accuracy(confusion_matrix):
    """
    Calculate the accuracy from the confusion matrix.

    Args:
        confusion_matrix (list): A 2D list representing the confusion matrix.

    Returns:
        float: The accuracy as a value between 0 and 1.
    """
    correct_predictions = sum(
        confusion_matrix[i][i] for i in range(len(confusion_matrix))
    )
    total_predictions = sum(sum(row) for row in confusion_matrix)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy


def calculate_confusion_matrix_details(confusion_matrix):
    """
    Calculate true positives, false positives, false negatives, and true negatives
    for each class from the confusion matrix.

    Args:
        confusion_matrix (list): A 2D list representing the confusion matrix.

    Returns:
        dict: A dictionary containing metrics for each class.
    """
    # Get the number of classes from the confusion matrix's size
    num_classes = len(confusion_matrix)
    details = defaultdict(dict)

    # Iterate over each class to calculate TP, FP, FN, and TN
    for i in range(num_classes):
        TP = confusion_matrix[i][i]
        FP = sum(confusion_matrix[j][i] for j in range(num_classes)) - TP
        FN = sum(confusion_matrix[i]) - TP
        TN = sum(
            sum(confusion_matrix[j][k] for k in range(num_classes) if k != i)
            for j in range(num_classes)
            if j != i
        )
        details[i]["TP"] = TP
        details[i]["FP"] = FP
        details[i]["FN"] = FN
        details[i]["TN"] = TN
    return details


def get_worst_performance(details):
    """
    Get the three worst performing icons and return their index value

    Args:
        details: A dictionary containing metrics for each class, returned from calculate_confusion_matrix_details
    """
    worst_performance = [0, 0, 0]
    first = 0
    second = 0
    third = 0
    for i in details:
        metrics = details[i]
        false_positives = metrics["FP"]
        false_negatives = metrics["FN"]
        total_wrong = false_positives + false_negatives
        if total_wrong > first:
            worst_performance[0] = i
            first = total_wrong
        elif total_wrong > second:
            worst_performance[1] = i
            second = total_wrong
        elif total_wrong > third:
            worst_performance[2] = i
            third = total_wrong
    return worst_performance


def calculate_precision(confusion_matrix):
    """
    Calculate the precision for each class based on the confusion matrix.

    Args:
        confusion_matrix (list): A 2D list representing the confusion matrix.

    Returns:
        dict: A dictionary containing precision scores for each class.
    """
    details = calculate_confusion_matrix_details(confusion_matrix)
    precision_scores = defaultdict(float)

    for class_index, metrics in details.items():
        TP = metrics["TP"]
        FP = metrics["FP"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        precision_scores[class_index] = precision
    return precision_scores


def calculate_recall(confusion_matrix):
    """
    Calculate the recall for each class based on the confusion matrix.

    Args:
        confusion_matrix (list): A 2D list representing the confusion matrix.

    Returns:
        dict: A dictionary containing recall scores for each class.
    """
    details = calculate_confusion_matrix_details(confusion_matrix)
    recall_scores = defaultdict(float)

    for class_index, metrics in details.items():
        TP = metrics["TP"]
        FN = metrics["FN"]
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        recall_scores[class_index] = recall
    return recall_scores


def show_breakdown(confusion_matrix):
    """
    Display the confusion matrices, precision, and recall of the three most error-prone icons

    Args:
        confusion_matrix (list): A 2D list representing the confusion matrix.
    """
    precision = calculate_precision(confusion_matrix)
    recall = calculate_recall(confusion_matrix)
    breakdown = calculate_confusion_matrix_details(confusion_matrix)

    worst_performance = get_worst_performance(breakdown)
    for i in worst_performance:
        icon_name = CLASS_LABEL_DICT[i]
        metrics = breakdown[i]

        # Create the confusion matrix
        matrix = np.array(
            [[metrics["TP"], metrics["FP"]], [metrics["FN"], metrics["TN"]]]
        )

        # Define row and column labels
        row_labels = ["Actual Positive", "Actual Negative"]
        col_labels = ["Predicted Positive", "Predicted Negative"]

        # Create a heatmap using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Greys",
            xticklabels=col_labels,
            yticklabels=row_labels,
            cbar=False,
            vmin=0,
            vmax=0,
            linewidths=1,
            linecolor="black",
        )

        # Add labels and title
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title(f"Confusion Matrix for {icon_name}")

        # Display the heatmap
        plt.show()
        print(
            f"Precision: {round(precision[i], 4) *100}% \nRecall: {round(recall[i], 4) * 100}%"
        )
