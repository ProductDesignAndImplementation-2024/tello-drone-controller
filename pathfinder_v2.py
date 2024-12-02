from collections import deque

# A helper function to perform BFS to find the optimal path
def bfs_find_path(matrix2, start, end):
    # Convert start and end to tuples to ensure they are hashable
    start = tuple(start)
    end = tuple(end)

    # If start and end are the same, no need to do BFS
    if start == end:
        return [start]
    
    # Initialize visited set and the queue
    visited = set()
    queue = deque([(start, [start])])  # Queue holds tuples of (current_node, path_to_current_node)
    
    while queue:
        current_node, path = queue.popleft()
        
        # Skip this node if we've already visited it
        if current_node in visited:
            continue
        
        # Mark the node as visited
        visited.add(current_node)
        
        # Check all the neighbors of the current node
        for neighbor in matrix2[current_node[0]][current_node[1]]:
            # Convert neighbor to a tuple to ensure it's hashable
            neighbor = tuple(neighbor)
            
            if neighbor not in visited:
                # If the neighbor is the end node, return the full path
                if neighbor == end:
                    return path + [neighbor]
                
                # Otherwise, continue to explore this neighbor
                queue.append((neighbor, path + [neighbor]))
    
    # If no path is found, return None
    return None

def print_path_directions(path):
    if not path:
        print("Error: No path provided or path is None")
        return
    
    # Ensure the path starts at (4, 4)
    if path[0] != (4, 4):
        print("Error: Path did not start at (4, 4)")
        return

    def direction(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return (dx, dy)

    def get_move_direction(prev, current, last_direction):
        dx = current[0] - prev[0]
        dy = current[1] - prev[1]

        # Determine the direction
        if dx > 0:  # Moving down
            return 's'
        elif dx < 0:  # Moving up
            return 'n'
        elif dy > 0:  # Moving right
            return 'r'
        elif dy < 0:  # Moving left
            return 'l'

        return ''  # Invalid move

    directions = []
    last_direction = (0, 1)  # Default initial direction

    for i in range(1, len(path)):
        move_direction = get_move_direction(path[i - 1], path[i], last_direction)
        directions.append(move_direction)
        last_direction = direction(path[i - 1], path[i])
    
    print("Path Directions:", directions)



# Now we use this BFS-based function inside `find_path`
def find_path(grid, start, end):
    # Call bfs_find_path to get the optimal path from start to end
    path = bfs_find_path(grid, start, end)
    
    # Print the result
    if path:
        print(f"Optimal path from {start} to {end}: {path}")
    else:
        print(f"No path found from {start} to {end}")

    return path