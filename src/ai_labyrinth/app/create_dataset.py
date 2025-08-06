import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import random
from queue import Queue
import tensorflow as tf 

DIMENSION_MULTIPLIER = 2
OFFSET_IMPAIR = 1

def create_maze(dim: int) -> np.ndarray:
    # Create a grid filled with walls
    dimension = dim * DIMENSION_MULTIPLIER + OFFSET_IMPAIR
    maze = np.ones((dimension, dimension))

    # Define the starting point
    x, y = (0, 0)
    maze[2*x+1, 2*y+1] = 0

    # Initialize the stack with the starting point
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        # Define possible directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < dim and ny < dim and maze[2*nx+1, 2*ny+1] == 1:
                maze[2*nx+1, 2*ny+1] = 0
                maze[2*x+1+dx, 2*y+1+dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()
            
    # Create an entrance and an exit
    maze[1, 0] = 2
    maze[-2, -1] = 3

    return maze

def draw_maze(maze_array: np.ndarray) -> None:
    """
    Visualizes a labyrinth array with four distinct categories.
    0: Path (white)
    1: Wall (black)
    2: Start (green)
    3: End (red)
    """
    # Define a custom colormap for the four categories
    # The colors are specified in the order of the values (0, 1, 2, 3)
    cmap = ListedColormap(['white', 'black', 'green', 'red'])

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))

    # Display the maze using the custom colormap
    ax.imshow(maze_array, cmap=cmap)

    # Set ticks and labels to improve readability (optional)
    ax.set_xticks(np.arange(-.5, maze_array.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, maze_array.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Generated Labyrinth")

    # Show the plot
    plt.savefig('my_plot.png')


def create_dataset(maze_dim: int, dataset_width: int) -> None:
    dim = maze_dim * DIMENSION_MULTIPLIER + OFFSET_IMPAIR
    X = np.ones((dataset_width, 32, 32), dtype=np.uint8)
    Y_labels = np.zeros((dataset_width, 32, 32), dtype=np.uint8)

    for i in range (0, dataset_width):
        labyrinth = create_maze(maze_dim)
        pad_total = 32 - dim # 32 - 21 = 11
        pad_width = ((pad_total // 2, pad_total - pad_total // 2), 
                     (pad_total // 2, pad_total - pad_total // 2))
        labyrinth_padded = np.pad(labyrinth, pad_width=pad_width, mode='constant', constant_values=1)
        Y_labels[i] = labyrinth_padded


    num_classes = 4
    Y_encoded = tf.keras.utils.to_categorical(Y_labels, num_classes=num_classes) # one hot encode for output of training with 4 labels

    np.savez("dataset_labyrinthes.npz", inputs=X, outputs=Y_encoded)

# maze = create_maze(16)
# draw_maze(maze)

create_dataset(10, 10)
data = np.load('dataset_labyrinthes.npz')
X = data['inputs']
Y = data['outputs']

