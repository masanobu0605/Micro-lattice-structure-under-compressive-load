import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##--先端たわみ角からpの算出--
phi_0_data = np.loadtxt("phi_0.csv" ,delimiter=",", encoding="shift-jis") #先端たわみ角のimport
phi_0_data = phi_0_data * np.pi / 180 #先端たわみ角をラジアン表記に直す


##--変数定義．大文字のみ定義すればいい--
L = 10e-3
D = L/10 #円柱断面形状を想定している．
E = 200e9
Eh = E/100

I = np.pi*(D**4)/64
ei = Eh * I

sigma_y = 100 * 10**6
Zp = D**3/6 #全塑性モーメントの計算に使う．円柱断面形状です
Mp = Zp*sigma_y

alpha = 30 #degで入力していいよ
alpha = alpha * np.pi /180

nn = len(phi_0_data)

p = np.zeros(nn)
for i in range(nn):
    p[i] = np.sin((alpha + phi_0_data[i]) * 0.5)
print("進捗度10%|pの導出完了")

##!--mの算出(mは行う積分の下限になる．0 -> m -> pi/2 の順番で大きい) 0.5がどこに来るのかによって答えがだいぶ違う
m = np.zeros(nn)
for i in range(nn):
    m[i] = np.arcsin(np.sin(alpha* 0.5)  / p[i])
print("進捗度20%|m の導出完了")


##--関数の準備F(start_point,stop_point,m)------------------a1,a2が積分区間
def Finte(a1,a2,p):
    n = 500
    d_theta = (a2 - a1) / n
    theta_before = a1
    Fanswer = 0
    
    for i in range(n):
        theta_after = theta_before + d_theta
        Fanswer = Fanswer  + (((1 - (p**2)*(np.sin(theta_before))**2)**(-0.5)) + ((1 - (p**2)*(np.sin(theta_after))**2)**(-0.5)) )*d_theta*0.5
        theta_before = theta_after
    return Fanswer
##--関数の準備E(start_point,stop_point,m)------------------a1,a2が積分区間
def Einte(a1,a2,p):
    n = 500
    d_theta = (a2 - a1) / n
    theta_before = a1
    Eanswer = 0
    
    for i in range(n):
        theta_after = theta_before + d_theta
        Eanswer = Eanswer + (((1 - (p**2)*(np.sin(theta_before))**2)**(0.5)) + ((1 - (p**2)*(np.sin(theta_after))**2)**(0.5)))*d_theta*0.5
        theta_before = theta_after
    return Eanswer
##--関数の準備f(p,m)------------------mが積分区間
def finte(p,m):
    finte_answer = 0
    finte_answer = Finte(m,np.pi * 0.5,p) - Finte(0,np.pi * 0.5,p) + 2*(Einte(0,np.pi * 0.5,p) - Einte(m,np.pi * 0.5,p))
    return finte_answer


##--lambdaの導出/
def lambda_definition(lll):
    ll =  lll - np.sqrt(Eh * I /(Mp * L)) * (Finte(m[i],np.pi*0.5,p[i])) * np.sqrt(lll) - 1/np.sin(phi_0_data[i] + alpha)
    return ll

lambda_P = np.zeros(nn)
for i in range(nn):
    b = np.sqrt(Eh * I /(Mp * L)) * (Finte(m[i],np.pi*0.5,p[i]))
    c = 1/np.sin(phi_0_data[i] + alpha)
    lambda_root = (b + np.sqrt(b**2 + 4*c))/2
    lambda_P[i] = lambda_root**2
print("進捗度30%|lambdaの導出完了")

##--P_loadの導出/
P_load = np.zeros(nn)
for i in range(nn):
    P_load[i] = lambda_P[i] * Mp / L
print("進捗度40%|Pの導出完了")

##--aの導出/
a = np.zeros(nn)
for i in range(nn):
    a[i] = Mp/(P_load[i] * np.sin(phi_0_data[i] + alpha))
print("進捗度50%|aの導出完了")


##--点Dでの変位量計算/
u,v,U,V = np.zeros(nn),np.zeros(nn),np.zeros(nn),np.zeros(nn)
for i in range(nn):
    u[i] = L - a[i] - np.sqrt(Eh * I /P_load[i]) * (finte(p[i],m[i]) * np.cos(alpha) + 2 * p[i] * np.sin(alpha) * np.cos(m[i]))# > 0
    v[i] = np.sqrt(Eh * I /P_load[i]) * ( - finte(p[i],m[i]) * np.sin(alpha) + 2 * p[i] * np.cos(alpha) * np.cos(m[i]))# > 0
        
print("進捗度70%|点Dでの変位量の導出完了")

##--点C(自由端)での変位量計算/
for i in range(nn):
    U[i] = u[i] + a[i] * (1 - np.cos(phi_0_data[i]))
    V[i] = v[i] + a[i] * np.sin(phi_0_data[i])
"""    if v[i] < 0:
        u[i],v[i],U[i],V[i] = None,None,None,None
    if u[i] < 0:
        u[i],v[i],U[i],V[i] = None,None,None,None"""
print("進捗度90%|点Cでの変位量の導出完了")


##--データ整理・出力/
datasheet = np.ndarray((nn,11))
phi_0_data = np.loadtxt("phi_0.csv" ,delimiter=",", encoding="shift-jis")
for i in range(0,nn):
    datasheet[i,0] = phi_0_data[i]
    datasheet[i,1] = p[i]
    datasheet[i,2] = a[i]
    datasheet[i,3] = P_load[i]
    datasheet[i,4] = u[i]
    datasheet[i,5] = v[i]
    datasheet[i,6] = U[i]
    datasheet[i,7] = V[i]
    datasheet[i,8] = L - a[i] - u[i]
    datasheet[i,9] = L - U[i]   
    datasheet[i,10] = lambda_P[i]

def fig():
    ##---荷重による変位量を3種類プロット
    fig, ax1 = plt.subplots()
    x = datasheet[:,3]
    y1 = datasheet[:,4]
    y2 = datasheet[:,5]
    ax1.plot(x, y1, 'o', color='r', markersize=4)
    ax1.plot(x, y2, '*', color='b', markersize=4)
    ax1.legend(["x_displacement","y_displacement"],prop = {"family" : "MS Gothic"})
    ax1.set_title("荷重による点Dの変位量 | L = " + str(L) + "[m]",fontname = 'MS Gothic')
    ax1.set_xlabel("Load[N] | P")
    ax1.set_ylabel("Displacement[m]")

    plt.savefig("荷重による点Dの変位量")

    ##---変位量の推移
    fig, ax2 = plt.subplots()
    ax2.set_ylim(-max(V) -max(V)/50,0)
    x1 = datasheet[:,8]
    x2 = datasheet[:,9]
    y1 = -datasheet[:,5]
    y2 = -datasheet[:,7]
    ax2.plot(x1, y1, 'o', color='r', markersize=3)
    ax2.plot(x2, y2, '*', color='b', markersize=3)
    ax2.set_xlim(0,L)
    ax2.set_aspect('equal')

    ax2.set_title("荷重による点C・点Dの推移 | L = " + str(L) + "[m]",fontname = 'MS Gothic')
    ax2.legend(["点C","点D"],prop = {"family" : "MS Gothic"})
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    plt.savefig("荷重による点C・点Dの推移")

    print("進捗度100%|計算完了")
    print("画像のプロットを行います")
fig()

datasheet_pd = pd.DataFrame(datasheet,columns=["phi0","p","a","P_load","u","v","U","V","L-a-u","L-U","lambda_P"],)
datasheet_pd.to_csv('lattice_compressive_load_cluclate_data.csv')
plt.show()
