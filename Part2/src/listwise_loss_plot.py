import matplotlib.pyplot as plt

# Data for plotting
epochs = list(range(1, 11))
losses = [
    13.62137222290039, 72.11636352539062, 10.750556945800781,
    261.01751708984375, 245.66067504882812, 10.750556945800781,
    -0.0, 5.545177459716797, 23.02585220336914, 23.02585220336914
]

# Creating the plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
