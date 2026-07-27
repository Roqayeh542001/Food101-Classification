import matplotlib.pyplot as plt

def plot_results(results):
    epochs = range(1, len(results["train_loss"]) + 1)

    plt.figure(figsize=(12,5))

    # Loss
    plt.subplot(1,2,1)

    plt.plot(epochs,
             results["train_loss"],
             label="Train")
    
    plt.plot(epochs,
             results["test_loss"],
             label="Test")
    
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)

    plt.plot(epochs,
             results["train_acc"],
             label="Train")
    
    plt.plot(epochs,
             results["test_acc"],
             label="Test")
    
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/training_curve.png")

    plt.show()