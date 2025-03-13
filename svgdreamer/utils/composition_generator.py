import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.ndimage import gaussian_filter

def generate_golden_spiral_image(size=600, num_arcs=10, dpi=300):
    """
    Generates an image of the golden spiral and returns it as a NumPy ndarray in grayscale.

    Parameters:
        size (int): The width and height of the image in pixels.
        num_arcs (int): The number of quarter-circle arcs to draw.
        dpi (int): The resolution of the figure.

    Returns:
        np.ndarray: The image of the golden spiral in grayscale format.
    """
    # Define golden ratio
    phi = (1 + sqrt(5)) / 2
    const = 500  # Base unit size
    altura = const
    
    # Initialize the figure
    # fig, ax = plt.subplots(figsize=(int(size * phi / dpi), size / dpi), dpi=dpi)
    fig, ax = plt.subplots(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Initial reference point
    x0, y0 = (phi * -const) / 2, -const / 2  
    arc_radius = altura
    angle_start = np.pi  # Start at 180 degrees

    # Define center offsets for each quarter-circle
    offsets = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    for k in range(num_arcs):
        theta = np.linspace(angle_start, angle_start - np.pi / 2, 50)  # Quarter-circle
        dx, dy = offsets[k % 4]
        center_x, center_y = x0 + dx * arc_radius, y0 + dy * arc_radius

        ax.plot(center_x + arc_radius * np.cos(theta), 
                center_y + arc_radius * np.sin(theta), 'k', lw=3)

        x0, y0 = center_x + np.cos(theta[-1]) * arc_radius, \
                 center_y + np.sin(theta[-1]) * arc_radius  
        arc_radius /= phi  
        angle_start -= np.pi / 2  

    # Render figure to buffer
    fig.canvas.draw()

    # Convert to numpy array and extract grayscale (R channel) for heatmap
    image_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3].mean(-1).astype(float) / 255.0  # Extract the R channel for grayscale

    plt.close(fig)  # Close the figure to free memory

    return image_array

def generate_equal_lateral_triangle(size=600, dpi=300):
    """
    Generates an equilateral triangle and returns it as a NumPy ndarray in grayscale.

    Parameters:
        size (int): The width and height of the image in pixels.
        dpi (int): The resolution of the figure.

    Returns:
        np.ndarray: The image of the equilateral triangle in grayscale format.
    """
    # Define the height of the equilateral triangle
    height = size * np.sqrt(3) / 2
    half_base = size / 2

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(size / dpi, height / dpi), dpi=dpi)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Coordinates of the triangle's vertices
    x = [-half_base, 0, half_base]
    y = [0, height, 0]

    # Draw the triangle contour (only the outline)
    ax.plot([x[0], x[1]], [y[0], y[1]], 'k', lw=3)  # Side 1
    ax.plot([x[1], x[2]], [y[1], y[2]], 'k', lw=3)  # Side 2
    ax.plot([x[2], x[0]], [y[2], y[0]], 'k', lw=3)  # Side 3

    # Render figure to buffer
    fig.canvas.draw()

    # Convert to numpy array and extract grayscale (R channel)
    image_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3].mean(-1).astype(float) / 255.0

    plt.close(fig)

    return image_array

def generate_diagonal_line(size=600, dpi=300):
    """
    Generates a diagonal line and returns it as a NumPy ndarray in grayscale.

    Parameters:
        size (int): The width and height of the image in pixels.
        dpi (int): The resolution of the figure.

    Returns:
        np.ndarray: The image of the diagonal line in grayscale format.
    """
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Plot a diagonal line from top-left to bottom-right (only the contour)
    ax.plot([0, size], [size, 0], 'k', lw=3)

    # Render figure to buffer
    fig.canvas.draw()

    # Convert to numpy array and extract grayscale (R channel)
    image_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3].mean(-1).astype(float) / 255.0

    plt.close(fig)

    return image_array

def generate_l_shape_line(size=600, dpi=300):
    """
    Generates an L-shape line and returns it as a NumPy ndarray in grayscale.

    Parameters:
        size (int): The width and height of the image in pixels.
        dpi (int): The resolution of the figure.

    Returns:
        np.ndarray: The image of the L-shape line in grayscale format.
    """
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Plot an L-shape line (horizontal and vertical segments)
    ax.plot([0, size], [0, 0], 'k', lw=3)  # Horizontal line
    ax.plot([0, 0], [0, size], 'k', lw=3)  # Vertical line

    # Render figure to buffer
    fig.canvas.draw()

    # Convert to numpy array and extract grayscale (R channel)
    image_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3].mean(-1).astype(float) / 255.0
    

    plt.close(fig)

    return image_array


if __name__ == '__main__':
    # Example usage of different shapes
    golden_spiral_image = generate_golden_spiral_image()
    triangle_image = generate_equal_lateral_triangle()
    diagonal_line_image = generate_diagonal_line()
    l_shape_image = generate_l_shape_line()

    # Apply Gaussian blur to the shapes
    sigma = 50.
    blurred_golden_spiral = gaussian_filter(golden_spiral_image, sigma=sigma)
    blurred_triangle = gaussian_filter(triangle_image, sigma=sigma)
    blurred_diagonal_line = gaussian_filter(diagonal_line_image, sigma=sigma)
    blurred_l_shape = gaussian_filter(l_shape_image, sigma=sigma)

    # Plot the shapes and their blurred versions
    plt.figure(figsize=(12, 8))

    # Original and blurred golden spiral
    plt.subplot(4, 2, 1)
    plt.imshow(golden_spiral_image, cmap="hot")
    plt.title("Golden Spiral")
    plt.axis("off")

    plt.subplot(4, 2, 2)
    plt.imshow(blurred_golden_spiral, cmap="hot")
    plt.title("Blurred Golden Spiral")
    plt.axis("off")

    # Original and blurred triangle
    plt.subplot(4, 2, 3)
    plt.imshow(triangle_image, cmap="hot")
    plt.title("Equilateral Triangle")
    plt.axis("off")

    plt.subplot(4, 2, 4)
    plt.imshow(blurred_triangle, cmap="hot")
    plt.title("Blurred Triangle")
    plt.axis("off")

    # Original and blurred diagonal line
    plt.subplot(4, 2, 5)
    plt.imshow(diagonal_line_image, cmap="hot")
    plt.title("Diagonal Line")
    plt.axis("off")

    plt.subplot(4, 2, 6)
    plt.imshow(blurred_diagonal_line, cmap="hot")
    plt.title("Blurred Diagonal Line")
    plt.axis("off")

    # Original and blurred L-shape line
    plt.subplot(4, 2, 7)
    plt.imshow(l_shape_image, cmap="hot")
    plt.title("L-shape Line")
    plt.axis("off")

    plt.subplot(4, 2, 8)
    plt.imshow(blurred_l_shape, cmap="hot")
    plt.title("Blurred L-shape Line")
    plt.axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig('composition_generator.png')
