import numpy as np
from scipy import io
from quaternion import Quaternion
import math
import os
from scipy.linalg import sqrtm
from quaternion import Quaternion
from scipy.spatial.transform import Rotation as Rot

class UKF:
    def __init__(self,x0,P0,Q,R,dt,tol=0.001):
        self.x_hat_k=x0
        self.P_k=P0
        self.Q=Q
        self.R=R
        self.n=6
        self.chi=np.zeros((7,12))
        self.gamma=np.zeros((7,12))
        self.Z=np.zeros((6,12))
        self.dt=dt
        self.g=Quaternion(scalar=0,vec=[0, 0, 9.81])
        self.tol=tol
        self.zkm=np.zeros(6)
        self.x_bar=np.zeros(7)
        self.K=np.zeros((6,6))

    def generateSigmaPoints(self,P_km1):
        S=sqrtm(P_km1+self.Q)
        Wis=np.hstack((math.sqrt(self.n)*S,-math.sqrt(self.n)*S))
        # print("Wis: ",Wis)
        for i in range(2*self.n):
            wq=Wis[0:3,i]
            qw=self.quatFrom3D(wq)
            # if i==0: print(wq)
            q_km1=Quaternion(scalar=self.x_hat_k[0],vec=self.x_hat_k[1:4])
            q_temp=q_km1*qw
            self.chi[0:4,i]=q_temp.q
            self.chi[4:7,i]=Wis[3:6,i]+self.x_hat_k[4:7]
        # print('chi:\n',self.chi)
    def transformSigmaPoints(self):
        for i in range(2*self.n):
            self.A(i)
            self.H(i)
        # print( "gamma:\n",self.gamma)
        # print( 'Z:\n',self.Z[3:6,:])

    def A(self,i):
        xk=self.chi[:,i]
        w_kp1=xk[4:7]*self.dt
        quat_d=self.quatFrom3D(w_kp1)
        quat_k=Quaternion(scalar=xk[0],vec=xk[1:4])
        q_kp1=quat_k*quat_d #reversed order gives the correct product
        self.gamma[0:4,i]=q_kp1.q
        self.gamma[4:7,i]=xk[4:7]

    def H(self,i):
        xk=self.chi[:,i]
        quat_k=Quaternion(scalar=xk[0],vec=xk[1:4])
        g_p=quat_k.inv()*self.g*quat_k
        self.Z[3:6,i]=g_p.vec()
        self.Z[0:3,i]=xk[4:7]
        
    
    def computeMean(self):
        self.zkm=np.zeros(6)
        w_bar=np.zeros(3)
        for i in range(2*self.n):
            self.zkm+=self.Z[:,i]/(2*self.n)
            w_bar+=self.gamma[4:7,i]/(2*self.n)
        qt_bar=self.computeOrientationMean()
        self.x_bar[0:4]=qt_bar
        self.x_bar[4:7]=w_bar
        # print('zkm: ',self.zkm)
        # print('xkm: ',self.x_bar)
        # print(w_bar)
    
    def computeOrientationMean(self):
        qt_bar=Quaternion(scalar=1,vec=np.array([0,0,0]))
        while (True):
            e_bar_arr=np.zeros(3)
            for i in range(2*self.n):
                qi=Quaternion(scalar=float(self.gamma[0,i]),vec=self.gamma[1:4,i])
                ei=qi*qt_bar.inv()
                e_arr_i=self.quatTo3D(ei)
                e_bar_arr+=e_arr_i/(2*self.n)
            e_bar=self.quatFrom3D(e_bar_arr)
            # e_bar.normalize()
            q_new=e_bar*qt_bar
            dq=q_new.q-qt_bar.q
            # print(q_new)
            if np.linalg.norm(dq)<self.tol:
                break
            else:
                qt_bar=q_new
        # print(qt_bar)
        return qt_bar.q
    
    def computeCovAndUpdate(self,zk):
        Pkm=np.zeros((6,6))
        Pzz=np.zeros((6,6))
        Pxz=np.zeros((6,6))
        vk=zk-self.zkm
        for i in range(2*self.n):
            Wip=np.zeros(6)
            Wip[3:6]=self.gamma[4:7,i]-self.x_bar[4:7]
            qi=Quaternion(self.gamma[0,i],self.gamma[1:4,i])
            q_bar=Quaternion(scalar=float(self.x_bar[0]),vec=self.x_bar[1:4])
            q_rw=qi*q_bar.inv()
            Wip[0:3]=self.quatTo3D(q_rw)
            Pkm+=np.outer(Wip,Wip)/(2*self.n)

            Pzz+=np.outer(self.Z[:,i]-self.zkm,self.Z[:,i]-self.zkm)/(2*self.n)
            
            Pxz+=np.outer(Wip,self.Z[:,i]-self.zkm)/(2*self.n)
        Pvv=Pzz+self.R
        self.K=Pxz@np.linalg.inv(Pvv)
        temp=self.K@vk
        
        x_hat_k_rot=self.quatTo3D(Quaternion(self.x_bar[0],self.x_bar[1:4]))+temp[0:3]
        # print(x_hat_k_rot)
        new_q=self.quatVecFrom3D(x_hat_k_rot)
        newx=np.hstack((new_q,  np.array(self.x_bar[4:7]+temp[3:6])))
        self.x_hat_k=newx
        self.P_k=Pkm-self.K@Pvv@self.K.T
        # print('x_hat_rot: ',x_hat_k_rot,new_q,"new w: ",self.x_bar[4:7]+temp[3:6])
        # print('x_hat_rot: ',x_hat_k_rot,self.quatTo3D(Quaternion(self.x_bar[0],self.x_bar[1:4])))
        # print("vk: ",vk)
        # print("x_hat: ",self.x_hat_k)
        # print("Pvv: ",Pvv)
        # print("K: ",self.K)
        # print("Pkm: ",Pkm)
        # print("P_k: ",self.P_k)
    
    def update(self,zk):
        self.generateSigmaPoints(self.P_k)
        self.transformSigmaPoints()
        self.computeMean()
        self.computeCovAndUpdate(zk)
        q=Quaternion(self.x_hat_k[0],self.x_hat_k[1:4])
        euler=q.euler_angles()
        # print('euler: ',euler)
        return euler
        


    def quatFrom3D(self,wq):
        alpha_w=np.linalg.norm(wq)
        e_bar_w=wq/alpha_w
        qw=Quaternion(scalar=math.cos(alpha_w/2),vec=math.sin(alpha_w/2)*e_bar_w)
        return qw

    def quatVecFrom3D(self,wq):
        alpha_w=np.linalg.norm(wq)
        e_bar_w=wq/alpha_w
        qw=np.hstack((np.array([math.cos(alpha_w/2)]),  math.sin(alpha_w/2)*e_bar_w))
        return qw

    def quatTo3D(self,q):
        r = Rot.from_quat(np.hstack((q.vec(),np.array([q.scalar()]))))
        rot_v=r.as_rotvec()
        return rot_v




#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 3)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat(os.path.join(os.path.dirname(__file__) + '/imu/imuRaw' + str(data_num) + '.mat'))
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]

    # your code goes here
    Ts_imu= np.arange(T)
    test_beta=511
    test_beta_y=501
    test_alpha=303
    gyro_beta1=369.5
    gyro_beta2=373.5
    gyro_beta3=375.5
    gyro_alpha=34.5*57.32
    dt=(imu['ts'][0][T-1]-imu['ts'][0][0])/T


    Q=np.diag([0.001,0.001,0.001,100,100,100])*0.0001 #rotation + angularV
    R=np.diag([0.01,0.01,0.01,1,1,1]) # angularV + Accelerameter
    P_t=np.random.rand(6,6)
    P0=np.matmul(P_t,P_t.T)/100000


    x0=np.array([1,0,0,0,0,0,0])
    my_ukf=UKF(x0,P0,Q,R,dt)

    rolls_UKF=[]
    pitchs_UKF=[]
    yaws_UKF=[]
    # T=300
    for i in range(T):
        x_accel=(accel[0,i]-test_beta)*3300/1023/test_alpha*9.81
        y_accel=-(accel[1,i]-test_beta_y)*3300/1023/test_alpha*9.81
        z_accel=(accel[2,i]-test_beta)*3300/1023/test_alpha*9.81
    
        omega1=(gyro[0,i]-gyro_beta1)*3300/1023/gyro_alpha*9.81
        omega2=(gyro[1,i]-gyro_beta2)*3300/1023/gyro_alpha*9.81
        omega3=(gyro[2,i]-gyro_beta3)*3300/1023/gyro_alpha*9.81
        
        # print(np.array([omega1,omega3,omega2,x_accel,y_accel,z_accel]))
        # print('\n',i)
        euler=my_ukf.update(np.array([omega2,omega3,omega1,-x_accel,y_accel,z_accel]))
        rolls_UKF.append(euler[0])
        pitchs_UKF.append(euler[1])
        yaws_UKF.append(euler[2])
    # roll, pitch, yaw are numpy arrays of length T
    return rolls_UKF,pitchs_UKF,yaws_UKF
