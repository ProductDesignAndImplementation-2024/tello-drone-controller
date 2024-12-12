moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Only right, down, left, and up
visited = []
max_x = 0
max_y = 0

def get_directions(path):
    directions = []

    # Function to determine the direction between two points
    def direction(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return (dx, dy)
    
    # Initial direction
    current_dir = direction(path[0], path[1])
    for i in range(1, len(path) - 1):
        next_dir = direction(path[i], path[i + 1])
        if current_dir == next_dir:
            directions.append("f")
        else:
            if current_dir[0] != 0:  # Moving horizontally
                if current_dir[0] == next_dir[1]:
                    directions.append("r")
                else:
                    directions.append("l")
            elif current_dir[1] != 0:  # Moving vertically
                if current_dir[1] == next_dir[0]:
                    directions.append("l")
                else:
                    directions.append("r")
        
        current_dir = next_dir
    
    return directions

def __fms(grid, position, end, path): 
    global visited
    if (position) == (end): 
        print (path)
        return path
    visited.append(position)
    possible_moves = []
    for move_x, move_y in moves:
        next_x, next_y = position[0] + move_x, position[1] + move_y
        if 0 <= next_x < max_x and 0 <= next_y < max_y:
            if (grid[next_x][next_y] == '-'):
                next_y += move_y
                next_x += move_x
                if [next_x,next_y] not in visited:
                    possible_moves.append([next_x, next_y])
    if not possible_moves:
        path.remove(position)
        return None
    
    for i in possible_moves:
        path.append(i)
        result = __fms(grid, i, end, path)
        if result:
            return result
        
def find_path (grid, start, end):
    global max_y 
    max_y = len(grid)
    global max_x 
    max_x = len(grid[0])
    global visited 
    visited = []
    bad_list = __fms(grid, start, end, [start])
    bad_list.append([10,0])
    bad_list.insert(0, [10,8])
    return get_directions(bad_list)
