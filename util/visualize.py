import matplotlib.pyplot as plt
def plot_training_results(losses, val_acc, test_acc, save_path):
    epochs = range(1, len(losses) + 1)

    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Plot validation and test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='blue')
    plt.plot(epochs, test_acc, label='Test Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation and Test Accuracy over Epochs')
    plt.legend()

    # Save image
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()