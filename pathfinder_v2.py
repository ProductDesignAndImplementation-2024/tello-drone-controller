from collections import deque

def bfs_find_path(matrix2, start, end):
    start = tuple(start)
    end = tuple(end)

    if start == end:
        return [start]
    
    visited = set()
    queue = deque([(start, [start])])
    
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

def find_path(grid, start, end, Debbugger=False):
    path = bfs_find_path(grid, start, end)
    
    if Debbugger:
        if path:
            print(f"Optimal path from {start} to {end}: {path}")
        else:
            print(f"No path found from {start} to {end}")
    
    return path
