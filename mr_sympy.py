from sympy import *

def VecToso3(omg):
    so3mat = Matrix(3,3,[0,-omg[2],omg[1],omg[2],0,-omg[0],-omg[1],omg[0],0])
    return so3mat
    
def so3ToVec(so3mat):
    """Converts an so(3) representation to a 3-vector
    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat
    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    """
    omg = Matrix(3,1,[so3mat[2,1], so3mat[0,2], so3mat[1,0]])
    return omg

def AxisAng3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into
    axis-angle form
    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle
    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    """
    return (expc3.normalized(), expc3.norm())

def MatrixExp3(so3mat):
    """Computes the matrix exponential of a matrix in so(3)
    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The matrix exponential of so3mat
    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [ 0.69297817,  0.6313497 ,  0.34810748]])
    """
    omgtheta = so3ToVec(so3mat)
    theta = AxisAng3(omgtheta)[1]
    omgmat = so3mat / theta
    return eye(3) + sin(theta) * omgmat + (1 - cos(theta)) * omgmat * omgmat

def RpToTrans(R,p):
    T = R.row_join(p).col_join(Matrix(1,4,[0,0,0,1]))
    return T

def TransToRp(T):
    R = T[0:3,0:3]
    p = T[0:3,3]
    return R,p

def VecTose3(V):
    se3mat = ((VecToso3(Matrix(3,1,[V[0],V[1],V[2]]))).row_join(Matrix(3,1,[V[3],V[4],V[5]]))).col_join(zeros(1,4))
    return se3mat

def Adjoint(T):
    R,p = TransToRp(T)
    AdT = (R.row_join(zeros(3,3))).col_join((VecToso3(p)*R).row_join(R))
    return AdT

def MatrixExp6(se3mat):
    """Computes the matrix exponential of an se3 representation of
    exponential coordinates
    :param se3mat: A matrix in se3
    :return: The matrix exponential of se3mat
    Example Input:
        se3mat = np.array([[0,          0,           0,          0],
                           [0,          0, -1.57079632, 2.35619449],
                           [0, 1.57079632,           0, 2.35619449],
                           [0,          0,           0,          0]])
    Output:
        np.array([[1.0, 0.0,  0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.0],
                  [0.0, 1.0,  0.0, 3.0],
                  [  0,   0,    0,   1]])
    """
    omgtheta = so3ToVec(se3mat[0: 3, 0: 3])
    theta = AxisAng3(omgtheta)[1]
    omgmat = se3mat[0: 3, 0: 3] / theta
    R = MatrixExp3(se3mat[0: 3, 0: 3])
    p = (eye(3) * theta + (1 - cos(theta)) * omgmat + (theta - sin(theta)) \
                * omgmat*omgmat) * se3mat[0: 3, 3] / theta
    T = (R.row_join(p)).col_join(Matrix(1,4,[0,0,0,1]))
    return T


# Chapter 4 Forward Kinematics
def FKinBody(M, Blist, thetalist):
    T = M.copy()
    for i in range(len(thetalist)):
        T = T * exp(VecTose3(Blist[:, i] * thetalist[i]))
    return T

def FKinSpace(M, Slist, thetalist):
    T = M.copy()
    for i in range(len(thetalist)-1,-1,-1):
        T = exp(VecTose3(Slist[:, i] * thetalist[i])) * T
    return T

# Chapter 5 Velocity Kinematics and Statics

def JacobianBody (Blist, thetalist):
    Jb = Blist.copy()
    T = eye(4)
    for i in range(len(thetalist) - 2, -1, -1):
        T = T * MatrixExp6(VecTose3(Blist[:, i + 1] \
                                         * -thetalist[i + 1]))
        Jb[:, i] = Adjoint(T) * Blist[:, i]
    return Jb

def JacobianSpace(Slist, thetalist):
    """Computes the space Jacobian for an open chain robot
    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: The space Jacobian corresponding to the inputs (6xn real
             numbers)
    Example Input:
        Slist = np.array([[0, 0, 1,   0, 0.2, 0.2],
                          [1, 0, 0,   2,   0,   3],
                          [0, 1, 0,   0,   2,   1],
                          [1, 0, 0, 0.2, 0.3, 0.4]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])
    Output:
        np.array([[  0, 0.98006658, -0.09011564,  0.95749426]
                  [  0, 0.19866933,   0.4445544,  0.28487557]
                  [  1,          0,  0.89120736, -0.04528405]
                  [  0, 1.95218638, -2.21635216, -0.51161537]
                  [0.2, 0.43654132, -2.43712573,  2.77535713]
                  [0.2, 2.96026613,  3.23573065,  2.22512443]])
    """
    Js = Slist.copy()
    T = eye(4)
    for i in range(1, len(thetalist)):
        T = T * MatrixExp6(VecTose3(Slist[:, i - 1] \
                                * thetalist[i - 1]))
        Js[:, i] = Adjoint(T) * Slist[:, i]
    return Js