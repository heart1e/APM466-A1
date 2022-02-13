import xlrd
from pip._vendor.rich.jupyter import display
from sympy import symbols, solve, lambdify
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import derivative # to solve for the derivatives automatically
import datetime
from scipy.optimize import fsolve


def read_file(file_name):
    wb = xlrd.open_workbook(file_name)
    sheet = wb.sheet_by_index(0)
    # the collections of data needed to plot
    coupon_name = []
    coupon_rate = []
    n_values = []
    dates = []
    close_price = []
    period = []

    for c in range(10, 20):
        dates.append(sheet.cell_value(1, c))     # date info

    for row in range(2, sheet.nrows):
        temp = []   # for each coupon
        for col in range(11, 21):
            temp.append(float(sheet.cell_value(row, col)))     # close price
        close_price.append(temp)

    for i in range(2, sheet.nrows):
        coupon_name.append(sheet.cell_value(i, 0))  # coupon name
        coupon_rate.append(float(sheet.cell_value(i, 1)))  # coupon rate data
        n_values.append(int(sheet.cell_value(i, 10)))    # n_value
        period.append(int(sheet.cell_value(i, 21)))     # period
    return coupon_name, coupon_rate, n_values, dates, close_price, period


def compute_dirty_price(coupon_rate, n_values, close_price):
    res = []
    for i in range(len(close_price)):
        temp = []
        for j in range(len(close_price[0])):
            temp.append((n_values[i]+j)/365 * 100*coupon_rate[i] + close_price[i][j])
        res.append(temp)
    return res


# def solve_YTM(coupon_rate, dirty_price, n_values):
#     ytm = symbols('ytm')
#     sol = []
#     N = len(dirty_price)
#     M = len(dirty_price[0])
#     for i in range(N):
#         temp = []
#         for j in range(M):
#             l_expr = coupon_rate[i]*100/2 * ((1 - 1/(1+ytm)**(2*(n_values[i] + j)/365))/ytm)
#             r_expr = 100 * 1/((1+ytm)**(2*(n_values[i] + j)/365))
#             expr = l_expr + r_expr - dirty_price[i][j]
#             print(expr)
#             temp.append(solve(expr))
#         sol.append(temp)
#     return sol


def YTM_func(price, face, coupon, periods):
    def bond_price_calc(ytm):
        return ((coupon*((1-(1+ytm)**-periods))/ytm)+((face)/(1+ytm)**periods))-price
    return bond_price_calc


def newton(f, x_0, epsilon, max_iter):
    xn = x_0
    for n in range(0, max_iter):
        fxn = f(xn)
        df.loc[len(df)] = [n, xn, fxn]
        if abs(fxn) < epsilon:
            return xn
        Dfxn = derivative(f, xn, dx=1e-10)
        if Dfxn == 0:
            return None
        xn = xn - fxn/Dfxn
    return None


def solve_spot_rate(coupon, dirty_price, n):
    r = symbols('r')
    left_sum = 0
    right_sum = (100 + 1/2 * coupon) * np.power(np.e, -1*r)
    for _ in range(1, n):
        left_sum += 1/2 * coupon * np.power(np.e, -1*r)

    return solve(left_sum + right_sum - dirty_price)[0]


def solve_spot_rate2(coupon, dirty_price, n):
    r = symbols('r')
    left_sum = 0
    right_sum = (100 + 1/2 * coupon) * np.power(np.e, -1*r)
    for k in range(1, n):
        left_sum += 1/2 * coupon * np.power(np.e, -1*k*r)
    func_np = lambdify(r, left_sum + right_sum - dirty_price, modules=['numpy'])
    return fsolve(func_np, np.array([0.0]))


def solve_forward(spot):
    res = []
    for i in range(len(spot)):
        temp = []
        for j in range(1, len(spot[0])):
            top = spot[i][j] * 1/2 * (j + 1) - spot[i][0] * 1/2
            bottom = 1/2 * (j + 1) - 1/2
            temp.append((top/bottom)[0])
        res.append(temp)
    return res


def compute_forward_matrix(forward):
    res_matrix = np.array(forward).T
    matrix = []
    for row in range(res_matrix.shape[0]):
        if row % 2 == 1:
            row_temp = []
            for col in range(res_matrix.shape[1] - 1):
                row_temp.append(np.log(res_matrix[row][col+1] / res_matrix[row][col]))
            matrix.append(row_temp)
    return np.array(matrix)


def solve_forward2(spot):
    res = []
    for i in range(len(spot)):
        temp = []
        for j in range(1, len(spot[0])):
            temp.append(spot[i][j] * 1/2 * (j + 1) - spot[i][0] * 1/2)
        res.append(temp)
    return res


def compute_ytm_matrix(d_p, c_r, p):
    res_data = []
    for i in range(len(d_p[0])):
        temp = []
        for j in range(len(d_p)):
            if newton(f=YTM_func(price=d_p[j][i],face=100,coupon=c_r[j]*100,periods=p[j]), x_0=0.1, epsilon=1e-10, max_iter=100) == None:
                break
            temp.append(2*newton(f=YTM_func(price=d_p[j][i],face=100,coupon=c_r[j]*100/2,periods=p[j]), x_0=0.1, epsilon=1e-10, max_iter=100))
        res_data.append(temp)
    # print(res_data)
    # print(len(res_data))
    # print(len(res_data[0]))
    res_matrix = np.array(res_data).T
    # print(res_matrix.shape[0])
    # print(res_matrix.shape[1])
    # print(res_matrix)
    matrix = []
    for row in range(res_matrix.shape[0]):
        if row % 2 == 1:
            row_temp = []
            for col in range(res_matrix.shape[1] - 1):
                row_temp.append(np.log(res_matrix[row][col+1] / res_matrix[row][col]))
            matrix.append(row_temp)
    return np.array(matrix)


if __name__ == '__main__':
    df = pd.DataFrame(columns=['iteration', "x_value", "function_value"])
    xn = 0

    # call read file to gather the data
    # c_n, c_r, n, d, c_p, p = read_file("Book3.xlsx")
    # print(c_p)
    # d_p = compute_dirty_price(c_r, n, c_p)
    # print(d_p)

    # matrix = compute_ytm_matrix(d_p, c_r, p)
    # print(matrix)
    # cov = []
    # for row1 in matrix:
    #     temp = []
    #     for row2 in matrix:
    #         temp.append(np.cov(row1, row2)[0][1])
    #     cov.append(temp)
    # print(cov)
    # w, v = np.linalg.eig(np.array(cov))
    # print(w)
    # print(v)


    # print(c_r)
    # print(n)
    # print(c_p)
    # print(p)
    # print(d_p)
    # print(newton(f=YTM_func(price=100.53219178082192,face=100,coupon=0.005*100,periods=1), x_0=0.1, epsilon=1e-10, max_iter=100))
    # get all sport rate
    # s_r = []
    # for x in range(len(d_p[0])):
    #     temp = []
    #     for y in range(len(d_p)):
    #         temp.append(solve_spot_rate2(c_r[y]*100, d_p[y][x], p[y]))
    #     s_r.append(temp)
    #
    # forward = solve_forward(s_r)
    # print(forward)
    # forward_matrix = compute_forward_matrix(forward)
    # print(forward_matrix)
    # forward_res = []
    # for row1 in forward_matrix:
    #     temp = []
    #     for row2 in forward_matrix:
    #         temp.append(np.cov(row1, row2)[0][1])
    #     forward_res.append(temp)
    # print(forward_res)
    # w, v = np.linalg.eig(np.array(forward_res))
    # print(w)
    # print(v)

    #
    # date = [datetime.date(2022, 1, 10), datetime.date(2022, 1, 11),
    #         datetime.date(2022, 1, 12), datetime.date(2022, 1, 13),
    #         datetime.date(2022, 1, 14), datetime.date(2022, 1, 17),
    #         datetime.date(2022, 1, 18), datetime.date(2022, 1, 19),
    #         datetime.date(2022, 1, 20), datetime.date(2022, 1, 21)]
    # for i in range(len(date)):
    #     plt.plot(p, s_r[i], label=date[i])
    # plt.xlabel('Number of terms(semiannual)')
    # plt.ylabel('Spot Rate(%)')
    # plt.title('Spot Rates of 10 bonds')
    # plt.legend()
    # plt.show()

    # compute forward rate per day
    # forward_rate = solve_forward2(s_r)
    #
    # date = [datetime.date(2022, 1, 10), datetime.date(2022, 1, 11),
    #         datetime.date(2022, 1, 12), datetime.date(2022, 1, 13),
    #         datetime.date(2022, 1, 14), datetime.date(2022, 1, 17),
    #         datetime.date(2022, 1, 18), datetime.date(2022, 1, 19),
    #         datetime.date(2022, 1, 20), datetime.date(2022, 1, 21)]
    # for i in range(len(date)):
    #     plt.plot(p[1:], forward_rate[i], label=date[i])
    # plt.xlabel('Terms range from 0.5yr')
    # plt.ylabel('Forward Rate(%)')
    # plt.title('Forward Rates of 10 bonds')
    # plt.legend()
    # plt.show()


    # res_data = []
    # for i in range(len(d_p[0])):
    #     temp = []
    #     for j in range(len(d_p)):
    #         if newton(f=YTM_func(price=d_p[j][i],face=100,coupon=c_r[j]*100,periods=p[j]), x_0=0.1, epsilon=1e-10, max_iter=100) == None:
    #             print("i = {}, j = {}".format(i, j))
    #             print(c_p[i][j], c_r[i], p[i])
    #             break
    #         temp.append(2*newton(f=YTM_func(price=d_p[j][i],face=100,coupon=c_r[j]*100/2,periods=p[j]), x_0=0.1, epsilon=1e-10, max_iter=100))
    #     res_data.append(temp)
    # # rearrange the data
    # #
    # date = [datetime.date(2022, 1, 10), datetime.date(2022, 1, 11),
    #         datetime.date(2022, 1, 12), datetime.date(2022, 1, 13),
    #         datetime.date(2022, 1, 14), datetime.date(2022, 1, 17),
    #         datetime.date(2022, 1, 18), datetime.date(2022, 1, 19),
    #         datetime.date(2022, 1, 20), datetime.date(2022, 1, 21)]
    # for i in range(len(date)):
    #     plt.plot(p, res_data[i], label=date[i])
    # plt.xlabel('Number of terms(semiannual)')
    # plt.ylabel('YTM(%)')
    # plt.title('Yield to Maturity of 10 bonds')
    # plt.legend()
    # plt.show()
