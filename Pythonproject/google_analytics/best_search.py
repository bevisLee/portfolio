


### graph search
def search(problem) :
    node = node(problem.initial)
    if problem.goal_test(node.state) :
        return node
    frontier = ProperDataStructure()
    frontier.append(node)
    explored = set()
    while frontier :
        node = frontier.pop()
        explored.add(node.state)
        for child in node.expand(problem) :
            if child.state not in explored and child not in frontier :
                if problem.goal_test(child.state) :
                    return child
                frontier.append(child)
    return None

### Breadth-first search
def breadth_first_search(problem) :
    node = Node(problem.initial)
    if problem.goal_test(node.state) :
        return node
    frontier = Queue()
    frontier.append(node)
    explored = set()
    while frontier :
        node = frontier.pop()
        explored.add(node.state)
        for child in node.expand(problem) :
            if child.state not in explored and child not in frontier :
                if problem.goal_test(child.state) :
                    return child
                frontier.append(child)
    return None

### Breadth-first search
def depth_first_search(problem) :
    node = Node(problem.initial)
    if problem.goal_test(node.state) :
        return node
    frontier = Stack()
    frontier.append(node)
    explored = set()
    while frontier :
        node = frontier.pop()
        explored.add(node.state)
        for child in node.expand(problem) :
            if child.state not in explored and child not in frontier :
                if problem.goal_test(child.state) :
                    return child
                frontier.append(child)
    return None

### Best-first search
def best_first_search(problem) :
    node = Node(problem.initial)
    if problem.goal_test(node.state) :
        return node
    frontier = PriorityQueue()
    frontier.append(node)
    explored = set()
    while frontier :
        node = frontier.pop()
        explored.add(node.state)
        for child in node.expand(problem) :
            if child.state not in explored and child not in frontier :
                if problem.goal_test(child.state) :
                    return child
                frontier.append(child)
    return None

