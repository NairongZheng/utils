"""
    author:damonzheng
    function:plot piecewise function
    edition:1.0
    date:20220228
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation



def main():
    """
        主函数
    """
    def update_points(num):
        '''
        更新数据点
        '''
        point_ani.set_data(x[num], y[num])
        return point_ani,
    
    # 画上下两个弧线
    x_up = np.linspace(0, 2*np.pi, 100)
    x_down = np.linspace(2*np.pi, 0, 100)
    
    
    # 画折线
    x_2 = np.linspace(0, 2*np.pi, 100)
    y_p_1 = np.array([-1 if (i < 2) else 0 for i in x_2])
    y_p_2 = np.array([i - 3 if (2 < i < 4) else 0 for i in x_2])
    y_p_3 = np.array([1 if (i > 4) else 0 for i in x_2])
    y_p = y_p_1 + y_p_2 + y_p_3

    # 过度
    x_guodu = np.linspace(0, 0.001, 50)
    y_guodu = np.linspace(0, -1, 50)


    x = np.hstack((x_up, x_down, x_guodu, x_2))
    
    y_up = np.sin(x / 2)[0:100]
    y_down = -np.sin(x / 2)[100:200]

    y = np.hstack((y_up, y_down, y_guodu, y_p))
    
    fig = plt.figure(tight_layout=True)
    plt.plot([2, 4], [0, 0], 'o')
    plt.plot(x_2, y_p)
    plt.plot(x, y)
    point_ani, = plt.plot(x[0], y[0], "ro")
    plt.xticks(np.arange(0, 6, 1))
    plt.yticks(np.arange(-2, 2, 1))

    ani = animation.FuncAnimation(fig, update_points, np.arange(0, 350), interval=20, blit=True)
    # plt.show()
    ani.save('xiaoche.gif', writer='imagemagick', fps=30)
    pass

if __name__ == '__main__':
    main()
