import torch
import numpy as np
import matplotlib.pyplot as plt

# Generate sample input (4x4 image)
def generate_sample_input():
    # Create a simple 4x4 pattern
    input_data = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [9.0, 10.0, 11.0, 12.0],
                               [13.0, 14.0, 15.0, 16.0]]]], dtype=torch.float32)
    
    # Target label (class 0)
    target = torch.tensor([0], dtype=torch.long)
    
    return input_data, target

# Visualize the input
def visualize_input(input_data):
    plt.figure(figsize=(6, 6))
    plt.imshow(input_data[0, 0].numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Sample Input (4x4)')
    plt.savefig('input_visualization.png')
    plt.close()

if __name__ == "__main__":
    input_data, target = generate_sample_input()
    visualize_input(input_data)
    print("Sample input shape:", input_data.shape)
    print("Sample input values:\n", input_data)
    print("Target:", target.item())
