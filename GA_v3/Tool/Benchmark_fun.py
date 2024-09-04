import numpy as np


def fun_bound(fun_name_fb):
    if fun_name_fb == 'ellipsoid':
        return -5.12, 5.12
    elif fun_name_fb == 'rosenbrock':
        return -2.048, 2.048
    elif fun_name_fb == 'ackley':
        return -32.768, 32.768
    elif fun_name_fb == 'griewank':
        return -600, 600
    elif fun_name_fb == 'rastrigin':
        return -5.12, 5.12


def result_calculation(fun_name_rc, x_rc):
    if fun_name_rc == 'ellipsoid':
        return np.array(ellipsoid(x_rc))
    elif fun_name_rc == 'rosenbrock':
        return np.array(rosenbrock(x_rc))
    elif fun_name_rc == 'ackley':
        return np.array(ackley(x_rc))
    elif fun_name_rc == 'griewank':
        return np.array(griewank(x_rc))
    elif fun_name_rc == 'rastrigin':
        return np.array(rastrigin(x_rc))


def ellipsoid(pop_ell):  # x:[-5.12, 5.12]
    if np.ndim(pop_ell) == 1:
        pop_ell = pop_ell[np.newaxis, :]
    num_pop_ell = pop_ell.shape[0]  # 个体的数量
    dim_pop_ell = pop_ell.shape[1]  # 个体的维度（变量个数）
    out_ell = np.zeros(num_pop_ell)
    for i_ell in range(num_pop_ell):
        pop_i_ell = pop_ell[i_ell, :]
        for j_ell in range(dim_pop_ell):
            out_ell[i_ell] += (j_ell + 1) * pop_i_ell[j_ell] ** 2
    return out_ell


def rosenbrock(pop_ros):  # x:[-2.048, 2.048]
    if np.ndim(pop_ros) == 1:
        pop_ros = pop_ros[np.newaxis, :]

    num_pop_ros = pop_ros.shape[0]  # 个体的数量
    dim_pop_ros = pop_ros.shape[1]  # 个体的维度（变量个数）
    out_ros = np.zeros(num_pop_ros)
    for i_ros in range(num_pop_ros):
        pop_i_ros = pop_ros[i_ros, :]
        for j_ros in range(dim_pop_ros - 1):
            out_ros[i_ros] += (100 * np.square(pop_i_ros[j_ros + 1] - np.square(pop_i_ros[j_ros])) + np.square(pop_i_ros[j_ros] - 1))
    return out_ros


def ackley(pop_ack):  # x:[-32.768, 32.768]
    if np.ndim(pop_ack) == 1:
        pop_ack = pop_ack[np.newaxis, :]

    num_pop_ack = pop_ack.shape[0]  # 个体的数量
    dim_pop_ack = pop_ack.shape[1]  # 个体的维度（变量个数）
    out_ack = np.zeros(num_pop_ack)
    for i_ack in range(num_pop_ack):
        pop_i_ack = pop_ack[i_ack, :]
        sum_1_ack = np.sum(pop_i_ack ** 2)
        sum_2_ack = np.sum(np.cos(2 * np.pi * pop_i_ack))
        out_ack[i_ack] = -20 * np.exp(-0.2 * np.sqrt(sum_1_ack / dim_pop_ack)) - np.exp(sum_2_ack / dim_pop_ack) + 20 + np.exp(1)
    return out_ack


def griewank(pop_gri):  # x:[-600, 600]
    if np.ndim(pop_gri) == 1:
        pop_gri = pop_gri[np.newaxis, :]

    num_pop_gri = pop_gri.shape[0]  # 个体的数量
    dim_pop_gri = pop_gri.shape[1]  # 个体的维度（变量个数）
    out_gri = np.zeros(num_pop_gri)
    sum_2_gri = 1.0
    for i_gri in range(num_pop_gri):
        pop_i_gri = pop_gri[i_gri, :]
        sum_1_gri = np.sum(pop_i_gri ** 2) / 4000
        for j_gri in range(dim_pop_gri):
            sum_2_gri *= np.cos(pop_i_gri[j_gri] / np.sqrt(j_gri + 1))
            out_gri[i_gri] = sum_1_gri - sum_2_gri + 1
    return out_gri


def rastrigin(pop_ras):  # x:[-5.12, 5.12]
    if np.ndim(pop_ras) == 1:
        pop_ras = pop_ras[np.newaxis, :]

    num_pop_ras = pop_ras.shape[0]  # 个体的数量
    dim_pop_ras = pop_ras.shape[1]  # 个体的维度（变量个数）
    out_ras = np.zeros(num_pop_ras)
    for i_ras in range(num_pop_ras):
        pop_i_ras = pop_ras[i_ras, :]
        sum_1_ras = np.sum(pop_i_ras ** 2) / 4000
        sum_2_ras = np.sum(np.cos(2*np.pi*pop_i_ras))
        out_ras[i_ras] = 10*dim_pop_ras + sum_1_ras - 10*sum_2_ras

    return out_ras
