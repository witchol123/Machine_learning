import numpy as np

# 输入矩阵
X = np.array([[1,1],[1,0],[0,1],[0,0]])
# 权重向量
W = np.array([0,0])
# 期望输出矩阵
D = np.array([1,1,1,0])
# 学习率
a = 0.1
# 期望误差
expect_E = 0.005
# 最大尝试次数
max_Count = 20

# 硬限幅函数
def sgn(v):
    if v > 0:
        return 1
    else:
        return 0

# 读取实际输出=权重矩阵的转置 * 输入矩阵的每个元素
def get_V(W,x):
    return sgn(np.dot(W.T,x))

# 读取误差=期望值-实际输出值
def get_E(W,x,d):
    return d-get_V(W,x)

# 权重计算w(n+1)=w(n)+a*x(n)*e
def neww(oldw,d,x,a):
    e=get_E(oldw,x,d)
    return(oldw+a*x*e,e)


count = 0
while True:
    i = 0
    err = 0
    for xn in X:
        W,e=neww(W,D[i],xn,a) 
        i+=1 
        err += pow(e,2)
    err /= float(i)
    count+=1
    print("第%d次调整后的权重为："%count)
    print(W)
    print("第%d次调整后的误差为："%count)
    print(err)
    if err < expect_E or count >= max_Count:
        break

print("开始验证结果：")
for xn in X:
    print("D:%s and W:%s -->%d"%(xn,W.T,get_V(W,xn)))


    





