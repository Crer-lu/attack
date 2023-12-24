import torchvision

save_dir = "datasets"
train_dataset = torchvision.datasets.MNIST(root=save_dir, train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root=save_dir, train=False, download=True)
