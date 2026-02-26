import numpy as np
import matplotlib.pyplot as plt
import os

# Create an instance of the dataset class mainly for visualization 
from src.datasets.shapes import BinaryShapesDataset

def main():
    # Attempt to load the dataset
    data_path = os.path.join("data", "shapes_5x5.npy")
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find dataset at {data_path}")
        return
        
    print(f"Loading dataset from {data_path}...")
    dataset = np.load(data_path)
    
    n_examples = len(dataset)
    print(f"Loaded {n_examples} total shape configurations.")
    
    # Initialize a Dataset object just to use its helpful visualize method
    # Note: 5x5 grid shape is hardcoded here corresponding to the loaded dataset
    ds = BinaryShapesDataset(grid_shape=(5, 5))
    
    # Randomly select 5 indices
    rng = np.random.default_rng()
    random_indices = rng.choice(n_examples, size=5, replace=False)
    
    print(f"Plotting 5 randomly chosen examples (Indices: {random_indices})...")
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for ax_idx, sample_idx in enumerate(random_indices):
        sample = dataset[sample_idx]
        
        # Use the class's visualize method but point it to our subplot axes
        ds.visualize(sample, ax=axes[ax_idx])
        axes[ax_idx].set_title(f"Sample #{sample_idx}")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()