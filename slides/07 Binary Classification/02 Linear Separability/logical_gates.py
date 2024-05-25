import matplotlib.pyplot as plt



def and_gate():
    plt.xlabel('x')
    plt.ylabel('y')

    x_pts = [0, 0, 1]
    y_pts = [0, 1, 0]

    plt.plot(x_pts, y_pts, 'o', color = 'b', label="0")

    x_pts = [1]
    y_pts = [1]

    plt.plot(x_pts, y_pts, '^', color = 'r', label="1")


    plt.title("AND gate")
    plt.legend(loc="center")
    plt.grid()
    plt.show()


def or_gate():
    plt.xlabel('x')
    plt.ylabel('y')

    x_pts = [0]
    y_pts = [0]

    plt.plot(x_pts, y_pts, 'o', color = 'b', label="0")

    x_pts = [0, 1, 1]
    y_pts = [1, 0, 1]

    plt.plot(x_pts, y_pts, '^', color = 'r', label="1")


    plt.title("OR gate")
    plt.legend(loc="center")
    plt.grid()
    plt.show()


def xor_gate():
    plt.xlabel('x')
    plt.ylabel('y')

    x_pts = [0, 1]
    y_pts = [0, 1]

    plt.plot(x_pts, y_pts, 'o', color = 'b', label="0")

    x_pts = [0, 1]
    y_pts = [1, 0]

    plt.plot(x_pts, y_pts, '^', color = 'r', label="1")


    plt.title("XOR gate")
    plt.legend(loc="center")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    #and_gate()
    #or_gate()
    xor_gate()