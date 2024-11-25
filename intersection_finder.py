import cv2
import numpy as np
import heapq


def find_grid_intersections(image_path, min_pixels=5, border_margin=10, max_density=0.3, vicinity_size=10, min_distance=10):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to load image")

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    output_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    height, width = binary.shape

    raw_intersections = []

    for y in range(border_margin, height - border_margin):
        for x in range(border_margin, width - border_margin):
            if binary[y, x] == 255:
                # Check for continuous white pixels in each direction
                up = np.all(binary[max(0, y - min_pixels):y, x] == 255)
                down = np.all(binary[y + 1:min(height, y + min_pixels + 1), x] == 255)
                left = np.all(binary[y, max(0, x - min_pixels):x] == 255)
                right = np.all(binary[y, x + 1:min(width, x + min_pixels + 1)] == 255)

                conditions = [up, down, left, right]
                if sum(conditions) >= 3:
                    vicinity = binary[max(0, y - vicinity_size):min(height, y + vicinity_size + 1),
                                      max(0, x - vicinity_size):min(width, x + vicinity_size + 1)]
                    white_density = np.sum(vicinity == 255) / (vicinity.size)
                    if white_density > max_density:
                        continue

                    raw_intersections.append((x, y))

    suppressed_intersections = []
    for (x, y) in raw_intersections:
        if all(np.sqrt((x - px)**2 + (y - py)**2) >= min_distance for (px, py) in suppressed_intersections):
            suppressed_intersections.append((x, y))
            cv2.circle(output_image, (x, y), 3, (0, 0, 255), -1)

    return suppressed_intersections, output_image, binary


def a_star_path(binary_image, start, end, intersections, directional_weight=(1, 1)):
    def heuristic(a, b):
        return abs(a[0] - b[0]) * directional_weight[0] + abs(a[1] - b[1]) * directional_weight[1]

    # Create a set for fast lookup of other intersections
    intersection_set = set(intersections)
    intersection_set.discard(start)  # Remove the starting point, since it's allowed
    intersection_set.discard(end)    # Remove the ending point, since it's allowed

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}

    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        # If the current node is the end, reconstruct and return the path
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse to get the path from start to end

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[1] < binary_image.shape[0] and 0 <= neighbor[0] < binary_image.shape[1]:
                if binary_image[neighbor[1], neighbor[0]] == 255:
                    if neighbor in intersection_set:
                        continue

                    # Tentative g_score
                    tentative_g_score = g_score[current] + 1

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # No path found


if __name__ == "__main__":
    image_path = "processed.png"
    intersections, result_image, binary = find_grid_intersections(
        image_path, min_pixels=5, border_margin=0, max_density=0.3, vicinity_size=10, min_distance=10
    )

    print("Detected intersections:", intersections)

    max_neighbor_distance = 60
    visited_intersections = set()

    for i, start in enumerate(intersections):
        for j, end in enumerate(intersections):
            if i != j:
                distance = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
                if distance <= max_neighbor_distance:
                    path = a_star_path(binary, start, end, list(visited_intersections), directional_weight=(1, 1))
                    if path:
                        print(f"Path found between {start} and {end}")

                        visited_intersections.add(start)
                        visited_intersections.add(end)

                        for point in path:
                            cv2.circle(result_image, point, 1, (0, 255, 0), -1)
    cv2.imshow("Intersections and Paths", result_image)
    cv2.imwrite("intersections_and_paths.png", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
