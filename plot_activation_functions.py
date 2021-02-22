import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


if __name__ == "__main__":
    xs = np.linspace(-10, 10, 1000)
    colors = sns.color_palette()
    
    plt.figure(figsize=(6,4))
    plt.plot(xs, sigmoid(xs), label='Sigmoid')
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.tight_layout()
    plt.savefig("backtheory_activation_sigmoid.pdf")
    plt.savefig("backtheory_activation_sigmoid.png", dpi=250)

    plt.figure(figsize=(6,4))
    plt.plot(xs, relu(xs), color=colors[1], label='ReLU')
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.tight_layout()
    plt.savefig("backtheory_activation_relu.pdf")
    plt.savefig("backtheory_activation_relu.png", dpi=250)

    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.plot(xs, sigmoid(xs), label='Sigmoid')
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(xs, relu(xs), color=colors[1], label='ReLU')
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.tight_layout()
    plt.savefig("backtheory_activation_comb.pdf")
    plt.savefig("backtheory_activation_comb.png", dpi=250)

    delta = 0.000001
    derivative10 = (sigmoid(10+delta) - sigmoid(10-delta))/(delta)
    print(derivative10)