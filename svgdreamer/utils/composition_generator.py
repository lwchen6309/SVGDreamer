import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.ndimage import gaussian_filter
from PIL import Image


def formatter(fig, size):
    # Convert to numpy array and extract grayscale for heatmap
    image_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3].mean(-1).astype(float) / 255.0  # Extract the R channel for grayscale
    image_array = np.array(Image.fromarray(image_array).resize((size, size), Image.NEAREST))
    image_array = (1 - image_array) * 255
    image_array = image_array / 255.0
    return image_array

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
    image_array = formatter(fig, size)
    plt.close(fig)
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
    image_array = formatter(fig, size)
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
    image_array = formatter(fig, size)
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
    image_array = formatter(fig, size)
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

    # Function to add a bounding box and hide ticks
    def add_bounding_box(ax):
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        rect = plt.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    # Original and blurred golden spiral
    ax1 = plt.subplot(4, 2, 1)
    ax1.imshow(golden_spiral_image, cmap="hot")
    ax1.set_title("Golden Spiral")
    add_bounding_box(ax1)

    ax2 = plt.subplot(4, 2, 2)
    ax2.imshow(blurred_golden_spiral, cmap="hot")
    ax2.set_title("Blurred Golden Spiral")
    add_bounding_box(ax2)

    # Original and blurred triangle
    ax3 = plt.subplot(4, 2, 3)
    ax3.imshow(triangle_image, cmap="hot")
    ax3.set_title("Equilateral Triangle")
    add_bounding_box(ax3)

    ax4 = plt.subplot(4, 2, 4)
    ax4.imshow(blurred_triangle, cmap="hot")
    ax4.set_title("Blurred Triangle")
    add_bounding_box(ax4)

    # Original and blurred diagonal line
    ax5 = plt.subplot(4, 2, 5)
    ax5.imshow(diagonal_line_image, cmap="hot")
    ax5.set_title("Diagonal Line")
    add_bounding_box(ax5)

    ax6 = plt.subplot(4, 2, 6)
    ax6.imshow(blurred_diagonal_line, cmap="hot")
    ax6.set_title("Blurred Diagonal Line")
    add_bounding_box(ax6)

    # Original and blurred L-shape line
    ax7 = plt.subplot(4, 2, 7)
    ax7.imshow(l_shape_image, cmap="hot")
    ax7.set_title("L-shape Line")
    add_bounding_box(ax7)

    ax8 = plt.subplot(4, 2, 8)
    ax8.imshow(blurred_l_shape, cmap="hot")
    ax8.set_title("Blurred L-shape Line")
    add_bounding_box(ax8)

    plt.tight_layout()
    # plt.show()
    plt.savefig('composition_generator.png')