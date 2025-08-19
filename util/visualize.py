import matplotlib.pyplot as plt
def plot_training_results(losses, val_acc, test_acc, save_path):
    epochs = range(1, len(losses) + 1)

    # 绘制训练损失
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # 绘制验证和测试准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='blue')
    plt.plot(epochs, test_acc, label='Test Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation and Test Accuracy over Epochs')
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()