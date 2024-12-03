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

# Now we use this BFS-based function inside `find_path`
def find_path(grid, start, end, Debbugger=False):
    # Call bfs_find_path to get the optimal path from start to end
    path = bfs_find_path(grid, start, end)
    if path:
        # Print the result
        if Debbugger:
            print(f"Optimal path from {start} to {end}: {path}")
        return path
    else:
        # Print the result
        if Debbugger:
            print(f"No path found from {start} to {end}")
        return path