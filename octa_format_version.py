from lxml import etree as ET
import numpy as np
import os
import re
import os
from glob import glob


# Function to apply rotation transformation to points
def rotate_point(x, y, angle, cx, cy):
    """Rotate point (x, y) around center (cx, cy) by angle (in degrees)."""
    theta = np.radians(angle)
    x_new = (x - cx) * np.cos(theta) - (y - cy) * np.sin(theta) + cx
    y_new = (x - cx) * np.sin(theta) + (y - cy) * np.cos(theta) + cy
    return round(x_new, 3), round(y_new, 3)


# Function to apply transformation (rotation) to path coordinates
def apply_transform_to_points(points, transform):
    """Apply rotation transformation directly to a list of (x, y) points."""
    match = re.search(r'rotate\((-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', transform)
    if match:
        angle, cx, cy = map(float, match.groups())
        transformed_points = [rotate_point(x, y, angle, cx, cy) for x, y in points]
        return transformed_points
    return points  # Return original points if no transformation was found


# Function to approximate an ellipse (or circle) using six Bézier segments
def approximate_ellipse_six_segments(cx, cy, rx, ry):
    """Approximates an ellipse using exactly six cubic Bézier curves with Z to close the path."""
    # k = 0.55191502449  # Bézier approximation constant for smoothness
    k = 0.35191502449  # Bézier approximation constant for smoothness

    # Angles for six-segment division (360° / 6 = 60° per segment)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points, without repeating the start

    # Compute control points for each segment
    bezier_points = []
    for i in range(6):
        theta1 = angles[i]
        theta2 = angles[(i + 1) % 6]

        # Start and end points of the segment
        x1, y1 = cx + rx * np.cos(theta1), cy + ry * np.sin(theta1)
        x2, y2 = cx + rx * np.cos(theta2), cy + ry * np.sin(theta2)

        # Control points
        dx1, dy1 = -np.sin(theta1) * k * rx, np.cos(theta1) * k * ry
        dx2, dy2 = np.sin(theta2) * k * rx, -np.cos(theta2) * k * ry

        c1x, c1y = x1 + dx1, y1 + dy1
        c2x, c2y = x2 + dx2, y2 + dy2

        # Append to list
        bezier_points.append((x1, y1, c1x, c1y, c2x, c2y, x2, y2))

    return bezier_points


def remove_first_rectangle(svg_path, output_path):
    """Removes the first large rectangle (background) from the SVG file based on 75% threshold."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Get SVG dimensions
    svg_width = float(root.get("width", 0))
    svg_height = float(root.get("height", 0))

    # Define the 75% threshold
    width_threshold = 0.75 * svg_width
    height_threshold = 0.75 * svg_height

    # Iterate through elements and find the first large rectangle
    for element in root.iter():
        tag = element.tag.split("}")[-1]  # Extract the tag name without namespace

        if tag == "rect":
            width = float(element.get("width", 0))
            height = float(element.get("height", 0))
            
            # Check if this rect is likely the background (using 75% of SVG dimensions)
            if width >= width_threshold and height >= height_threshold:
                element.getparent().remove(element)
                break  # Remove only the first rectangle

        elif tag == "path":
            d = element.get("d", "")
            if "L" in d and "Z" in d:  # Check for a closed rectangular path
                commands = d.split()
                if len(commands) == 10:  # Typical rectangular path with "M x y L x y L x y L x y Z"
                    element.getparent().remove(element)
                    break  # Remove only the first detected path-based rectangle

    # Save the modified SVG
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Background rectangle removed. Saved as {output_path}")


# Function to convert supported SVG shapes into paths
def convert_to_path(element):
    """Convert rect, polygon, polyline, ellipse, and circle to paths with transformations applied."""
    tag = element.tag.split("}")[-1]  # Extract the tag name without namespace
    transform = element.get("transform", None)  # Get transform if available

    if tag == "rect":
        x, y = float(element.get("x", 0)), float(element.get("y", 0))
        width, height = float(element.get("width", 0)), float(element.get("height", 0))
        rect_path = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        if transform:
            rect_path = apply_transform_to_points(rect_path, transform)
        d = f"M {rect_path[0][0]} {rect_path[0][1]} L {rect_path[1][0]} {rect_path[1][1]} " \
            f"L {rect_path[2][0]} {rect_path[2][1]} L {rect_path[3][0]} {rect_path[3][1]} Z"
        return d

    elif tag in ["polygon", "polyline"]:
        points = element.get("points", "").strip().split()
        formatted_points = [(float(p.split(',')[0]), float(p.split(',')[1])) for p in points]
        if transform:
            formatted_points = apply_transform_to_points(formatted_points, transform)
        d = f"M {formatted_points[0][0]} {formatted_points[0][1]} " + \
            " ".join(f"L {p[0]} {p[1]}" for p in formatted_points[1:])
        if tag == "polygon":
            d += " Z"
        return d
    
    elif tag in ["ellipse", "circle"]:
        if tag == "ellipse":
            cx, cy = float(element.get("cx", 0)), float(element.get("cy", 0))
            rx, ry = float(element.get("rx", 0)), float(element.get("ry", 0))
        else:  # tag == "circle"
            cx, cy = float(element.get("cx", 0)), float(element.get("cy", 0))
            r = float(element.get("r", 0))
            rx, ry = r, r  # Circle is a special case of an ellipse

        # Approximate the ellipse/circle using six cubic Bézier curves
        bezier_segments = approximate_ellipse_six_segments(cx, cy, rx, ry)

        if transform:
            transformed_segments = []
            for (x1, y1, c1x, c1y, c2x, c2y, x2, y2) in bezier_segments:
                transformed_points = apply_transform_to_points([(x1, y1), (c1x, c1y), (c2x, c2y), (x2, y2)], transform)
                transformed_segments.append((*transformed_points[0], *transformed_points[1], *transformed_points[2], *transformed_points[3]))
            bezier_segments = transformed_segments

        # Convert the six-segment Bézier approximation into an SVG path
        d = f"M {bezier_segments[0][0]} {bezier_segments[0][1]} "
        for x1, y1, c1x, c1y, c2x, c2y, x2, y2 in bezier_segments:
            d += f"C {c1x} {c1y}, {c2x} {c2y}, {x2} {y2} "

        d += "Z"  # Close the path
    return d


# Function to remove unnecessary attributes and groups
def clean_svg(root):
    transform_data = extract_global_transform(root)
    
    # Remove viewBox attribute
    if 'viewBox' in root.attrib:
        del root.attrib['viewBox']
    
    # Remove unnecessary group transformations
    for g in root.findall('.//{http://www.w3.org/2000/svg}g'):
        transform = g.get('transform', '')
        if 'translate' in transform or 'rotate' in transform or 'scale' in transform:
            del g.attrib['transform']
    
    # Apply global transformation if available
    if transform_data:
        angle, cx, cy = transform_data
        apply_global_transform(root, angle, cx, cy)
    
    return root


def remove_first_rectangle(root):
    """Removes the first large rectangle (background) from the SVG file based on 75% threshold."""
    svg_width = float(root.get("width", 0))
    svg_height = float(root.get("height", 0))

    # Define the 75% threshold
    width_threshold = 0.75 * svg_width
    height_threshold = 0.75 * svg_height

    for element in root.iter():
        tag = element.tag.split("}")[-1]  # Extract tag without namespace

        if tag == "rect":
            width = float(element.get("width", 0))
            height = float(element.get("height", 0))
            
            if width >= width_threshold and height >= height_threshold:
                element.getparent().remove(element)
                print("Removed background rectangle.")
                return  # Remove only the first large rectangle

        elif tag == "path":
            d = element.get("d", "")
            if "L" in d and "Z" in d:  # Check if it's a closed rectangular path
                commands = d.split()
                if len(commands) == 10:  # Typical rect path: "M x y L x y L x y L x y Z"
                    element.getparent().remove(element)
                    print("Removed background path rectangle.")
                    return  # Remove only the first detected path-based rectangle


def duplicate_paths(root, required_paths):
    """Duplicates existing paths until the required number of paths is reached."""
    paths = [elem for elem in root.iter() if elem.tag.endswith('path')]  # Extract only <path> elements
    num_existing_paths = len(paths)

    if num_existing_paths == 0:
        print("No paths found to duplicate.")
        return

    if num_existing_paths >= required_paths:
        print(f"Already have {num_existing_paths} paths, no duplication needed.")
        return

    current_path_count = num_existing_paths  # Use a counter instead of iterating each time
    index = 0

    while current_path_count < required_paths:
        original_path = paths[index % num_existing_paths]  # Cycle through existing paths
        new_path = ET.Element("path", attrib=original_path.attrib)  # Clone path with attributes
        root.append(new_path)  # Append the new path to the root
        current_path_count += 1  # Increment counter
        index += 1

    print(f"Duplicated paths to reach {required_paths} total paths from {num_existing_paths}.")


def convert_svg(input_svg, output_svg, required_paths=256):
    tree = ET.parse(input_svg)
    root = tree.getroot()
    remove_first_rectangle(root)

    # Iterate through elements and convert them
    for element in root.iter():
        tag = element.tag.split("}")[-1]  # Extract the tag name without namespace
        if tag in ["rect", "polygon", "polyline", "circle", "ellipse"]:
            path_d = convert_to_path(element)
            if path_d:
                path_element = ET.Element("path", d=path_d)
                
                # Copy over attributes like fill, stroke, etc.
                for attr in ["fill", "stroke", "stroke-width", "opacity"]:
                    if element.get(attr):
                        path_element.set(attr, element.get(attr))
                
                # Replace shape with its new path
                parent = element.getparent()
                parent.replace(element, path_element)

    duplicate_paths(root, required_paths)

    # Save the new SVG with paths
    tree.write(output_svg, encoding="utf-8", xml_declaration=True)
    print(f"Converted SVG saved as {output_svg}")


if __name__ == '__main__':
    # Load the SVG file
    octa_dir = 'octa_examples'
    output_dir = 'init_target/demo'
    
    # shapes = ['circle', 'hexagon', 'rectangle', 'triangle']
    shapes = ['demo']
    for shape in shapes:
        input_svgs = glob(os.path.join(octa_dir, shape, '*.svg'))
        for input_svg in input_svgs:
            filename = os.path.basename(input_svg)
            print(filename)
            filename = filename.replace('.svg', f'_{shape}.svg')
            output_svg = os.path.join(output_dir, filename)
            convert_svg(input_svg, output_svg, required_paths=512)
