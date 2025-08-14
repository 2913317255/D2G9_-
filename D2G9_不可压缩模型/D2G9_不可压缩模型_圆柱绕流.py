import taichi as ti
import tifffile
import numpy as np
import math
ti.init(arch=ti.cpu)  # 优先GPU
from PIL import Image
# D2Q9模型
D = 2
Q = 9

NX,NY = 1280,360

# ----参数以及物理量----
tau = 0.6           # 决定单组分粘度

u_in = 0.0001

print("仿真范围：",NX,NY)

omega = ti.field(ti.f32, shape=Q)   # 每个方向上的权重
ecx = ti.field(ti.i32, shape=Q)     # 格子速度
ecy = ti.field(ti.i32, shape=Q)     # 格子速度
opp = ti.field(ti.i32, shape=Q)     # 离散速度的相反序号

old_omega = [4.0 / 9.0,
             1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
             1.0 / 36.0, 1.0 / 36.0,1.0 / 36.0, 1.0 / 36.0]
old_ecx = [0, 1, 0, -1, 0, 1, -1, -1, 1]
old_ecy = [0, 0, 1, 0, -1, 1, 1, -1, -1]
old_opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# 初始filed
for i in range(Q):
    omega[i] = old_omega[i]
    ecx[i] = old_ecx[i]
    ecy[i] = old_ecy[i]
    opp[i] = old_opp[i]

# 组分的物理量
rho = ti.field(ti.f32, shape=(NX, NY))  # 组分的密度（不可压缩模型，所以这个是不是真正的密度，我在写代码的时候也没弄清楚）
P = ti.field(ti.f32, shape=(NX, NY))    # 压力
u = ti.field(ti.f32, shape=(NX, NY))  # 组分的x分量速度
v = ti.field(ti.f32, shape=(NX, NY))  # 组分的y分量速度
real_u = ti.field(ti.f32, shape=(NX, NY))
boundary_types = ti.field(ti.f32, shape=(NX, NY))

# 输出的PNG图片
out_png_field = ti.Vector.field(n=3, dtype=int, shape=(NX, NY))

# 分布函数
gin = ti.field(ti.f32, shape=(NX, NY, Q))  # 第一组分布函数
gout = ti.field(ti.f32, shape=(NX, NY, Q))  # 第二组分布函数

# 默认的一些参数
"""
c = 1
d0,d1,d2 = 5/12,1/3,1/12
"""

@ti.func
def feq(u_f: float, v_f: float, P_f :float ,q_f: int) -> float:
    """
    c = 1
    计算得到Feq（分布函数非平衡部分）
    :param u_f: 速度u
    :param v_f: 速度v
    :param w_f: 速度w
    :param rho_f: 密度
    :param q_f: 离散的速度方向
    :return: Feq
    """
    eu = u_f * ecx[q_f] + v_f * ecy[q_f]
    u2 = u_f * u_f + v_f * v_f
    out_feq = omega[q_f] * (3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
    if q_f == 0:
        out_feq = out_feq - 5.0 /3.0 * P_f

    if q_f == 1 or q_f == 2 or q_f == 3 or q_f == 4:
        out_feq = 1.0/3.0 * P_f + out_feq

    if q_f == 5 or q_f == 6 or q_f == 7 or q_f == 8:
        out_feq = 1.0/12.0 * P_f + out_feq

    return out_feq


@ti.func
def get_s_alpha_u(u_f: float, v_f: float,  q_f: int) -> float:
    eu = u_f * ecx[q_f] + v_f * ecy[q_f]
    u2 = u_f * u_f + v_f * v_f
    return omega[q_f] * (3.0 * eu + 4.5 * eu * eu - 1.5 * u2)


@ti.kernel
def init():
    for i, j in ti.ndrange(NX, NY):
        # 初始化一个圆柱
        if (i-NY/2)**2 + (j-NY/2)**2 <= 36**2:
            boundary_types[i, j] = 1

        # 流体域初始化
        if boundary_types[i, j] == 0:
            rho[i, j] = 1.0
            P[i, j] = 3.0 / 5.0
        else:
            rho[i, j] = 0.000001
        u[i, j] = 0.0
        v[i, j] = 0.0

        for q in ti.ndrange(Q):
            gin[i, j,q] = feq(u[i, j],v[i, j],P[i, j],q)


@ti.kernel
def collision():
    """
    碰撞步，对应公式2.12右端
    :return:
    """
    for i, j, q in ti.ndrange(NX,NY, Q):
        if boundary_types[i, j] == 0:
            gout[i, j,  q] = gin[i, j,  q] - (gin[i, j,  q] - feq(u[i, j], v[i, j], P[i, j], q)) / tau


@ti.kernel
def boundary():
    """
    边界条件
    :return:
    """
    for i, j in ti.ndrange(NX,NY):
        # 如果是固体
        if boundary_types[i, j] == 1:
            for q in ti.ndrange(Q):
                gout[i, j, q] = gout[i+ecx[q], j+ecy[q], opp[q]]

@ti.kernel
def boundary_new():
    """
    进出口压差
    :return:
    """
    for j in ti.ndrange(NY):
        for q in ti.ndrange(Q):
            # 左边定压
            gin[0, j, q] = feq(u_in, 0.0, 1.2 * 3.0 / 5.0, q)
            # 右边定压
            gin[NX - 1, j, q] = feq(u[NX - 1, j], v[NX - 1, j],  0.8 * 3.0 / 5.0, q)


@ti.kernel
def stream():
    """
    迁移，对应公式2.12左端
    这里我用的是全周期边界，进出口压差边界条件在boundary_new()中覆盖了
    :return:无返回参数
    """
    for i, j in ti.ndrange(NX,NY):
        # # 非固体域执行迁移
        if boundary_types[i, j] == 0:
            for q in ti.ndrange(Q):
                new_i = (i - ecx[q] + NX) % NX
                new_j = (j - ecy[q] + NY) % NY
                gin[i, j, q] = gout[new_i, new_j, q]

@ti.kernel
def statistics():
    """
    统计宏观参数
    :return:
    """
    for i, j in ti.ndrange(NX,NY):
        # 流体域
        if boundary_types[i, j] == 0:
            # 统计更新密度，速度
            rho[i, j] = 0.0
            u[i, j] = 0.0
            v[i, j] = 0.0
            for q in ti.ndrange((1,Q)):
                rho[i, j] += gin[i, j, q]
                u[i, j] += gin[i, j, q] * ecx[q]
                v[i, j] += gin[i, j, q] * ecy[q]
            P[i, j] = 3.0/5.0 *(rho[i, j] + get_s_alpha_u(u[i, j],v[i, j],0))
            real_u[i, j] = ti.sqrt(u[i, j]**2 + v[i, j]**2)


@ti.func
def float_to_rgb(in_float:ti.f32,in_min:ti.f32,in_max:ti.f32):
    """
    将输入的数据转化为三维RGB值--实现彩色color map，去除了紫色
    :param in_float: 输入的数据
    :param in_min: 输入的数据的最小参考值
    :param in_max: 输入的数据的最大参考值
    :return: RGB
    """
    cha = (in_max - in_min)/6.0
    r = 0.0
    g = 0.0
    b = 0.0
    if in_float < in_min:
        r = 255
        g = 0
        b = 0
    # 1红--橙
    elif in_min <=  in_float <= in_min+cha:
        r = 255
        g = 152*(in_float-in_min)/cha
        b = 0
    # 2橙--黄
    elif in_min+cha <= in_float <= in_min + 2.0*cha:
        r = 255
        g = 152 + 103*(in_float - in_min - cha) / cha
        b = 0
    # 3黄--绿
    elif in_min+2.0*cha <= in_float <= in_min + 3.0*cha:
        r = 255 - 255*(in_float - in_min - 2.0*cha) / cha
        g = 255
        b = 0
    # 4绿--青
    elif in_min+3.0*cha <= in_float <= in_min + 4.0*cha:
        r = 0
        g = 255
        b = 255*(in_float - in_min-3.0*cha)/cha
    # 5青--蓝
    elif in_min+4.0*cha <= in_float <= in_min + 5.0*cha:
        r = 0
        g = 255-(in_float - in_min-4.0*cha)/cha
        b = 255
    # 6蓝--紫
    elif in_min+5.0*cha <= in_float <= in_min + 6.0*cha:
        r = 150*(in_float - in_min-5.0*cha) / cha
        g = 0
        b = 255

    elif in_float > in_max:
        r = 150
        g = 0
        b = 255
    return int(r),int(g),int(b)

@ti.kernel
def RGB_PNG():
    """
    填充PNG
    :return:
    """
    black_rgb = ti.Vector([255,255,255])
    for i, j in ti.ndrange(NX,NY):
        out_png_field[i, j] = float_to_rgb(real_u[i, j],0.0,0.4)
        if (i - NY / 2) ** 2 + (j - NY / 2) ** 2 <= 36 ** 2:
            out_png_field[i, j] = black_rgb


def out_png(filename):
    """
    输出PNG
    :param filename: 文件路径
    :return:
    """
    png_np = out_png_field.to_numpy()
    img_data = png_np.astype(np.uint8)
    if img_data.shape[0] == NX and img_data.shape[1] == NY:
        # 转置为 (高度, 宽度, 通道)
        img_data = img_data.transpose(1, 0, 2)
    # 创建图像对象
    img = Image.fromarray(img_data, 'RGB')
    img.save(filename)


# @ti.func
# @ti.kernel
# 目前似乎无法把这一部分代码放到ti.kernel内核或者ti.func中，所以输出文件这一操作可能会比较占用时间
def out_tiff(in_iter):
    """
    直接输出tiff文件
    :param in_iter: 步数
    :return:无返回值，仅生成tiff文件
    """
    rho_arr = rho.to_numpy()
    tifffile.imwrite('OutPut/' + str(in_iter) + 'rho.tiff', rho_arr)  # 将numpy数组切片后输出tiff文件

    P_arr = P.to_numpy()
    tifffile.imwrite('OutPut/' + str(in_iter) + 'P.tiff', P_arr)

    u_arr = u.to_numpy()
    tifffile.imwrite('OutPut/' + str(in_iter) + 'u.tiff', u_arr)  # 将numpy数组切片后输出tiff文件

    v_arr = v.to_numpy()
    tifffile.imwrite('OutPut/' + str(in_iter) + 'v.tiff', v_arr)  # 将numpy数组切片后输出tiff文件

def lbm_main():
    """
    LBM计算的主程序
    :return: 无返回值
    """

    print("----Start initialization----")
    init()          # 初始化
    print("----End of initialization----")

    # 计算总循环
    print("----Start the calculation----")
    for iter in range(60001):
        # evolution（求解） 碰撞->更新边界->迁移->宏观统计更新->更新实际速度->计算判断->选择数据输出
        collision()
        boundary()
        stream()
        boundary_new()
        statistics()

        if iter > NX and iter % 100 == 0:
            print("step:", iter)
            # out_tiff(iter)
            RGB_PNG()
            if iter < 10000:
                out_png('OutPut/0' + str(iter)+'.png')
            else:
                out_png('OutPut/' + str(iter)+'.png')


if __name__ == '__main__':
    lbm_main()
