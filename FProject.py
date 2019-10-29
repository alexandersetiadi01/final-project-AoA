import heapq
import numpy as np
import matplotlib.pyplot as plt

grid1 = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])

grid2 = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

grid3 = np.array([
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

grid4 = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])

A = (17, 3)
B = (17, 16)
C = (1, 18)
D = (0, 0)


def oneMap(grid, city):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(grid, cmap=plt.cm.Dark2)
    ax.scatter(A[1], A[0], label="A", color="red", s=200)
    ax.scatter(B[1], B[0], label="B", color="blue", s=200)
    ax.scatter(C[1], C[0], label="C", color="black", s=200)
    ax.scatter(D[1], D[0], label="D", color="yellow", s=200)
    plt.title("city {}".format(city))
    plt.legend()
    plt.show()


def showAll(A, B, C, D):
    # plot map and path
    fig = plt.figure(figsize=(50,50));plt.clf()
    ax1 = fig.add_subplot(2, 2, 1)
    plt.title("city 1")
    ax1.imshow(grid1, cmap=plt.cm.Dark2)
    ax1.scatter(A[1], A[0], label="A", color="red", s=200)
    ax1.scatter(B[1], B[0], label="B", color="blue", s=200)
    ax1.scatter(C[1], C[0], label="C", color="pink", s=200)
    ax1.scatter(D[1], D[0], label="D", color="yellow", s=200)

    ax2 = fig.add_subplot(2, 2, 2)
    plt.title("city 2")
    ax2.imshow(grid2, cmap=plt.cm.Dark2)
    ax2.scatter(A[1], A[0], label="A", color="red", s=200)
    ax2.scatter(B[1], B[0], label="B", color="blue", s=200)
    ax2.scatter(C[1], C[0], label="C", color="pink", s=200)
    ax2.scatter(D[1], D[0], label="D", color="yellow", s=200)

    ax3 = fig.add_subplot(2, 2, 3)
    plt.title("city 3")
    ax3.imshow(grid3, cmap=plt.cm.Dark2)
    ax3.scatter(A[1], A[0], label="A", color="red", s=200)
    ax3.scatter(B[1], B[0], label="B", color="blue", s=200)
    ax3.scatter(C[1], C[0], label="C", color="pink", s=200)
    ax3.scatter(D[1], D[0], label="D", color="yellow", s=200)

    ax4 = fig.add_subplot(2, 2, 4)
    plt.title("city 4")
    ax4.imshow(grid4, cmap=plt.cm.Dark2)
    ax4.scatter(A[1], A[0], label="A", color="red", s=200)
    ax4.scatter(B[1], B[0], label="B", color="blue", s=200)
    ax4.scatter(C[1], C[0], label="C", color="pink", s=200)
    ax4.scatter(D[1], D[0], label="D", color="yellow", s=200)
    plt.show()


def heuristic(a, b):
    return abs((b[0] - a[0]) + (b[1] - a[1]))


def astar(array, start, goal):

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] #movement

    # from internet(line 157 - 202)
    close_set = set()
    came_from = {}
    gscore = {start: 0}    #movement cost from the starting point to our current point/potential neighbors
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        # if 0 <= goal[0] < array.shape[0]:
        #   if 0 <= goal[1] < array.shape[1]:
        #       if array[goal[0]][goal[1]] == 1:
        #           print("error")
        #           break

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))


def travel(grid):

    while True:
        print("traveling \n"
              "start from: \n1.A \n2.B \n3.C \n4.D \n5.cancel \nchoose: ")
        start = input().upper()
        if start == "1" or start == "A":
            start = A
            startPost = "A"

        elif start == "2" or start == "B":
            start = B
            startPost = "B"

        elif start == "3" or start == "C":
            start = C
            startPost = "C"

        elif start == "4" or start == "D":
            start = D
            startPost = "D"

        elif start == "5" or start == "cancel":
            break

        else:
            print("invalid input\n\n")
            continue

        print("destination: \n1.A \n2.B \n3.C \n4.D \n5.cancel \nchoose: ")
        destination = input().upper()
        if destination == "1" or destination == "A":
            goal = A
            destinationPost = "A"

        elif destination == "2" or destination == "B":
            goal = B
            destinationPost = "B"

        elif destination == "3" or destination == "C":
            goal = C
            destinationPost = "C"

        elif destination == "4" or destination == "D":
            goal = D
            destinationPost = "D"

        elif destination == "5" or destination == "cancel":
            break

        else:
            print("invalid input\n\n")
            continue

        if start == goal:
            print("your destination can not be the same as the starting point")
            continue
        else:
            print("from {} to {} ?".format(startPost, destinationPost))
            print("confirm(Y/N): ")
            confirmation = input().upper()
            if confirmation == "Y":
                route = astar(grid, start, goal)
                #print("coordinates : ", route[::-1])
                # route
                x_coords = []
                y_coords = []

                for i in (range(0, len(route))):
                    x = route[i][0]
                    y = route[i][1]
                    x_coords.append(x)
                    y_coords.append(y)

                # plot map and path
                fig, ax = plt.subplots(figsize=(20, 20))
                ax.imshow(grid, cmap=plt.cm.Dark2)
                ax.scatter(A[1], A[0], label="A", color="red", s=200)
                ax.scatter(B[1], B[0], label="B", color="blue", s=200)
                ax.scatter(C[1], C[0], label="C", color="pink", s=200)
                ax.scatter(D[1], D[0], label="D", color="yellow", s=200)
                ax.plot(y_coords, x_coords, color="black")
                plt.legend()
                plt.show()
                break

            elif confirmation == "N":
                continue

            else:
                print("please enter Y/N")
                continue


def mainMenu():
    while True:
        print("main menu")
        print("1.show all cities \n2.choose a city \n3.exit\nchoose: ")
        choice = input()
        if choice == "1":
            showAll(A, B, C, D)

        elif choice == "3":
            print("have a nice day (>_<)")
            exit()

        elif choice == "2":
            while(True):
                print("cities option")
                print("choose your city: \n1.city 01 \n2.city 02 \n3.city 03 \n4.city 04"
                      "\n5.back \nchoose: ")
                city = input()

                if city == "1":
                    print("do you want to see the map again?(Y/N)  ")
                    again = input().upper()
                    if again == "Y":
                        oneMap(grid1, city)
                    elif again == "N":
                        pass
                    else:
                        print("invalid input\n")
                        continue

                    print("confirm city {} (Y/N): ".format(city))
                    confirmation = input().upper()
                    if confirmation == "Y":
                        travel(grid1)
                    elif confirmation == "N":
                        continue

                    else:
                        print("please enter Y/N")
                        continue

                elif city == "2":
                    print("do you want to see the map again?(Y/N)  ")
                    again = input().upper()
                    if again == "Y":
                        oneMap(grid2, city)
                    elif again == "N":
                        pass
                    else:
                        print("invalid input\n")
                        continue

                    print("confirm city {} (Y/N): ".format(city))
                    confirmation = input().upper()
                    if confirmation == "Y":
                        travel(grid2)
                    elif confirmation == "N":
                        continue

                    else:
                        print("please enter Y/N")
                        continue

                elif city == "3":
                    print("do you want to see the map again?(Y/N)  ")
                    again = input().upper()
                    if again == "Y":
                        oneMap(grid3, city)
                    elif again == "N":
                        pass
                    else:
                        print("invalid input\n")
                        continue

                    print("confirm city {} (Y/N): ".format(city))
                    confirmation = input().upper()
                    if confirmation == "Y":
                        travel(grid3)
                    elif confirmation == "N":
                        continue

                    else:
                        print("please enter Y/N")
                        continue

                elif city == "4":
                    print("do you want to see the map again?(Y/N)  ")
                    again = input().upper()
                    if again == "Y":
                        oneMap(grid4, city)
                    elif again == "N":
                        pass
                    else:
                        print("invalid input\n")
                        continue

                    print("confirm city {} (Y/N): ".format(city))
                    confirmation = input().upper()
                    if confirmation == "Y":
                        travel(grid4)
                    elif confirmation == "N":
                        continue

                    else:
                        print("please enter Y/N")
                        continue

                elif city == "5":
                    break

                else:
                    print("invalid input\n\n")
                    continue
        else:
            print("invalid input\n\n")


mainMenu()