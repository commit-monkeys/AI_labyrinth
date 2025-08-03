import matplotlib.pyplot as plt
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

def draw_maze(maze:list[list[int]]) -> None:
    fig, ax = plt.subplots(figsize=(10,10))
    
    # Set the border color to white
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    
    ax.set_xticks([])
    ax.set_yticks([])
      
    plt.show()


def create_dataset(maze_dim: int, dataset_width: int) -> None:
    dim = maze_dim * DIMENSION_MULTIPLIER + OFFSET_IMPAIR
    X = np.ones((dataset_width, dim, dim), dtype=np.uint8)
    Y_labels = np.zeros((dataset_width, dim, dim), dtype=np.uint8)

    for i in range (0, dataset_width):
        labyrinth = create_maze(maze_dim)
        Y_labels[i] = labyrinth

    num_classes = 4 
    Y_encoded = tf.keras.utils.to_categorical(Y_labels, num_classes=num_classes) # one hot encode for output of training with 4 labels

    np.savez("dataset_labyrinthes.npz", inputs=X, outputs=Y_encoded)

# create_dataset(10, 10)

data = np.load('dataset_labyrinthes.npz')
X = data['inputs']
Y = data['outputs']
print (X[1].shape)
print (Y[1].shape)