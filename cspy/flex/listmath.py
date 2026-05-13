from __future__ import print_function

from cspy.flex.listmathfast import (
    rotation_by_q,
    list3norm,
    list3add,
    list3normalize,
    list3diff,
    list3dot,
    list3mag,
    list3cross,
    list3subtract,
    list3multiply,
    list3vecmatrix,
    list3_proj_pts_axis,
    list3delparallelvecs,
    quaternion_mult,
    quaternion_rotatn,
    list3matrixmatrix,
    list3perp_face_vecs,
    list3apply_sym_list_norm,
    list3apply_sym_list,
    list3edge_vecs,
    list3apply_sym,
    list3list_add,
    list3make_all_translations,
    list3cross_combinations,
    list3cross_comb_tri,
    list3required_dvec,
    list3find_overlap,
    list3list_dot,
    list3square_trace,
    test_listmathfast_pointer,
    flist3angle as list3angle,
    set_dihedral_st_vectors_fast,
    set_angle_st_vectors_fast,
    set_bond_st_vectors_fast,
    fdihedral_st_vectors_xyz as dihedral_st_vectors_xyz,
    flist3dihedral as list3dihedral,
    set_impropertorsion_st_vectors_fast,
)


def list3centroid(points):
    total = [0.0, 0.0, 0.0]
    for pt in points:
        total = list3add(total, pt)
    centroid = list3multiply(total, 1.0 / float(len(points)))
    return centroid


def make_quat_rot1(x3):
    import math

    pi = math.acos(-1.0)
    x0 = x3[0]
    x1 = x3[1]
    x2 = x3[2]
    quat_rot = [
        (1.0 - x0) ** 0.5 * math.sin(2.0 * pi * x1),
        (1.0 - x0) ** 0.5 * math.cos(2.0 * pi * x1),
        x0 ** 0.5 * math.sin(2.0 * pi * x2),
        x0 ** 0.5 * math.cos(2.0 * pi * x2),
    ]
    return quat_rot


def make_quat_rot2(theta, omega):
    import math

    quat_rot = [
        math.cos(theta / 2.0),
        omega[0] * math.sin(theta / 2.0),
        omega[1] * math.sin(theta / 2.0),
        omega[2] * math.sin(theta / 2.0),
    ]
    return quat_rot


def quaternion_to_matrix(quaternion):
    matrix = []
    matrix.append(quaternion_rotatn([1.0, 0.0, 0.0], quaternion))
    matrix.append(quaternion_rotatn([0.0, 1.0, 0.0], quaternion))
    matrix.append(quaternion_rotatn([0.0, 0.0, 1.0], quaternion))
    return matrix


def make_quat_rot_from_matrix(mat):
    """ the axis of rotation is parallel to an eigenvector of mat,                                                                                                               
        cos theta, not theta appears in general matrix rep, so watch the sign                                                                                                    
        I expect that there is a more general formula online which is robust 
        note always the direction in which rotation matrix acts (R) dhc 231015 """
    import numpy as np
    import math

    eta = 1.0e-7
    mat = np.array(mat)
    theta = math.acos((np.trace(mat) - 1.0) / 2.0)
    evals, evecs = np.linalg.eig(mat)
    if len([x for x in evals if (x < 1.0 + eta and x > 1.0 - eta)]) != 1:
        raise StandardError("error in make_quat_from_matrix")
    for i in range(3):
        if evals[i] > 1.0 - eta and evals[i] < 1 + eta:
            axis = evecs[i]
    # take care to ensure rotation is in correct direction
    quaternion = make_quat_rot2(theta, axis)
    if np.allclose(mat, np.array(quaternion_to_matrix(quaternion))):
        return quaternion
    else:
        print("Debug - Changing sign of theta in quaternion generation")
        theta *= -1.0
        quaternion = make_quat_rot2(theta, axis)
        if np.allclose(mat, np.array(quaternion_to_matrix(quaternion))):
            return quaternion
    if np.allclose(mat, np.array(quaternion_to_matrix(quaternion))):
        return quaternion
    else:
        print("Debug - Changing sign of theta in quaternion generation")
        theta *= -1.0
        quaternion = make_quat_rot2(theta, axis)
        if np.allclose(mat, np.array(quaternion_to_matrix(quaternion))):
            return quaternion
        raise StandardError("error in make_quat_from_matrix")


def list3zeros():
    return [0.0, 0.0, 0.0]


def list3matrixvec(
    m=[[None, None, None], [None, None, None], [None, None, None]], a=[None, None, None]
):
    v1 = list3dot(m[0], a)
    v2 = list3dot(m[1], a)
    v3 = list3dot(m[2], a)
    return [v1, v2, v3]


def rot_vec_around_dir(
    input_vec=[0.0, 0.0, 0.0], input_dir=[1.0, 0.0, 0.0], rot_angle=0.0
):
    """
    Rotation of a 3D-vector around another 3D-vector by angle theta
    
    Given an input direction (vx, vy, vz), first generates the normalized
    vector (ux, uy, uz):
    ux = vx / (vx**2 + vy**2 + vz**2)
    uy = vy / (vx**2 + vy**2 + vz**2)
    uz = vz / (vx**2 + vy**2 + vz**2)
    
    Then the rotation matrix around u = (ux, uy, uz) by an angle theta is given
    by:
    
    (ux*ux*(1-c)+c      ux*uy*(1-c)-uz*s   ux*uz*(1-c)+uy*s)
    (ux*uy*(1-c)+uz*s   uy*uy*(1-c)+c      uy*uz*(1-c)-ux*s) 
    (ux*uz*(1-c)-uy*s   uy*uz*(1-c)+ux*s   uz*uz*(1-c)+c   ) 
    
    with c = cos(theta) and s = sin(theta)
    
    Source:
    http://en.wikipedia.org/wiki/Rotation_matrix
    ['Rotation matrix from axis and angle']
    """
    from math import cos, sin
    from numpy import array, dot

    output_vec = [0.0, 0.0, 0.0]
    c = cos(rot_angle)
    s = sin(rot_angle)
    u = array(list3normalize(input_dir))
    M = array(
        [
            [
                u[0] * u[0] * (1.0 - c) + c,
                u[0] * u[1] * (1.0 - c) - u[2] * s,
                u[0] * u[2] * (1.0 - c) + u[1] * s,
            ],
            [
                u[0] * u[1] * (1.0 - c) + u[2] * s,
                u[1] * u[1] * (1.0 - c) + c,
                u[1] * u[2] * (1.0 - c) - u[0] * s,
            ],
            [
                u[0] * u[2] * (1.0 - c) - u[1] * s,
                u[1] * u[2] * (1.0 - c) + u[0] * s,
                u[2] * u[2] * (1.0 - c) + c,
            ],
        ]
    )
    output_vec = list(dot(M, array(input_vec)))
    return output_vec


def get_alignment_matrix(input_vec=[1.0, 0.0, 0.0], input_dir=[1.0, 0.0, 0.0]):
    """
    Calculates the transformation matrix that aligns a vector along a given 
    direction
    This matrix can be seen as a 3D-rotation using only two angles
    Freezes the remaining degree of freedom (third angle) by applying an 
    arbitrary constraint (one coefficient set to zero in the
    'construct_arbitrary_basis' routine)
    
    (1) Construction of the input and final bases
        
        v1 = normalized input vector
        V2 : v1*v2 = 0 and 1 coefficient is zero
        v3 = v1 ^ v2
        
        u1 = normalized input direction
        u2 : u1*u2 = 0 and 1 coefficient is zero
        u3 = u1 ^ u2
        
    (2) Solve the system:
    
       {m*v1 = u1
       {m*v2 = u2
       {m*v3 = u3
    
    where m is the transformation matrix

    scipy.linalg.solve() does that:
    http://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html
    """
    from numpy import array
    from scipy.linalg import solve as linsolve

    def construct_arbitrary_basis(vector):
        from math import sqrt

        tol = 1.0e-15
        v1 = list3normalize(vector)
        if abs(v1[0]) > tol:
            i = 0
            j = 1
        elif abs(v1[1]) > tol:
            i = 1
            j = 2
        elif abs(v1[2]) > tol:
            i = 2
            j = 0
        else:
            err_msg = (
                "The input vector used to construct a basis must not "
                "be the [0., 0., 0.] vector!\n "
                "value: {}".format(vector)
            )
            print(err_msg)
            raise StandardError(err_msg)
        v2 = array([0.0] * 3)
        v2[i] = -sqrt(v1[j] ** 2 / (v1[i] ** 2 + v1[j] ** 2))
        v2[j] = sqrt(v1[i] ** 2 / (v1[i] ** 2 + v1[j] ** 2))
        v3 = array(list3cross(v1, v2))
        basis = array([v1, v2, v3])
        return basis

    # (1) Construction of the u and v bases
    u = construct_arbitrary_basis(input_dir)
    v = construct_arbitrary_basis(input_vec)

    # (2) Solving the m*v = u system
    #
    # Let m be (9 variables):
    #
    #     (a b c)
    # m = (d e f)
    #     (g h i)
    #
    # Constraint matrix of the form (9 constraints):
    #
    # (v1x v1y v1z  0   0   0   0   0   0 ) a
    # ( 0   0   0  v1x v1y v1z  0   0   0 ) b
    # ( 0   0   0   0   0   0  v1x v1y v1z) c
    # (v2x v2y v2z  0   0   0   0   0   0 ) d
    # ( 0   0   0  v2x v2y v2z  0   0   0 ) e
    # ( 0   0   0   0   0   0  v2x v2y v2z) f
    # (v3x v3y v3z  0   0   0   0   0   0 ) g
    # ( 0   0   0  v3x v3y v3z  0   0   0 ) h
    # ( 0   0   0   0   0   0  v3x v3y v3z) j
    #
    # Target vector of the form (9 coordinates):
    #
    #   a   b   c   d   e   f   g   h   i
    # (u1x u1y u1z u2x u2y u2z u3x u3y u3z)
    #
    constraint_matrix = array([[0.0] * 9] * 9)
    target_vector = array([0.0] * 9)
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                constraint_matrix[a * 3 + b][b * 3 + c] = v[a][c]
                target_vector[a * 3 + c] = u[a][c]
    list_coeff = linsolve(constraint_matrix, target_vector)
    m = array([[0.0] * 3] * 3)
    for line_index in [0, 1, 2]:
        for column_index in [0, 1, 2]:
            m[line_index][column_index] = list_coeff[line_index * 3 + column_index]
    return m


def overlay_points_RMSD(set_pts1, set_pts2):
    """ This returns the optimal RMSD, and the quaternion to bring about rotation of set 2 onto set 1
    Assumes that both sets have mean point at origin
    See DOI 10.1002/jcc.20110 for theory and naming conventions
     """
    import numpy as np

    set_points1 = np.array(set_pts1)
    set_points2 = np.array(set_pts2)
    if False:
        matrix_R = np.array(list3list_dot(set_points1, set_points2))
    else:
        matrix_R = np.dot(np.transpose(set_points1), set_points2)
    matrix_F = np.array(
        [
            [
                matrix_R[0, 0] + matrix_R[1, 1] + matrix_R[2, 2],
                matrix_R[1, 2] - matrix_R[2, 1],
                matrix_R[2, 0] - matrix_R[0, 2],
                matrix_R[0, 1] - matrix_R[1, 0],
            ],
            [
                matrix_R[1, 2] - matrix_R[2, 1],
                matrix_R[0, 0] - matrix_R[1, 1] - matrix_R[2, 2],
                matrix_R[0, 1] + matrix_R[1, 0],
                matrix_R[0, 2] + matrix_R[2, 0],
            ],
            [
                matrix_R[2, 0] - matrix_R[0, 2],
                matrix_R[0, 1] + matrix_R[1, 0],
                -matrix_R[0, 0] + matrix_R[1, 1] - matrix_R[2, 2],
                matrix_R[1, 2] + matrix_R[2, 1],
            ],
            [
                matrix_R[0, 1] - matrix_R[1, 0],
                matrix_R[0, 2] + matrix_R[2, 0],
                matrix_R[1, 2] + matrix_R[2, 1],
                -matrix_R[0, 0] - matrix_R[1, 1] + matrix_R[2, 2],
            ],
        ]
    )

    eigenValues, eigenVectors = np.linalg.eig(matrix_F)
    tempObject = [[eigenValues[i], eigenVectors.transpose()[i]] for i in range(4)]
    tempObject = sorted(tempObject, key=lambda x: x[0])
    eigenVectors = [x[1] for x in tempObject]
    eigenValues = [x[0] for x in tempObject]

    def best_fit_MSD(l1, l2, max_lambda):
        return (
            sum([list3dot(x, x) for x in l1])
            + sum([list3dot(x, x) for x in l2])
            - 2.0 * max_lambda
        ) / l1.shape[0]

    MSD = best_fit_MSD(set_points1, set_points2, eigenValues[-1])
    if MSD > 0.0:
        return (MSD ** 0.5, tuple(eigenVectors[-1]))
    elif MSD < -0.1:
        print("MSD is negative, %s, in overlay of points" % (MSD))
        return (MSD, tuple(eigenVectors[-1]))
    else:
        return (0.0, tuple(eigenVectors[-1]))


def overlay_points_RMSD_quat(set_pts1, set_pts2, invert=False):
    """ This returns the optimal RMSD, and the quaternion to bring about rotation of set 2 onto set 1                                                                             
    Assumes that both sets have mean point at origin                                                                                                                              
    See DOI 10.1002/jcc.20110 for theory and naming conventions                                                                                                                  
    Contains automatic molecular inversion and returns different variables 
     """
    import numpy as np

    set_points1 = np.array(set_pts1)
    set_points2 = np.array(set_pts2)
    matrix_R = np.dot(
        np.transpose(set_points2), np.transpose(np.transpose(set_points1))
    )
    matrix_F = np.array(
        [
            [
                matrix_R[0, 0] + matrix_R[1, 1] + matrix_R[2, 2],
                matrix_R[1, 2] - matrix_R[2, 1],
                matrix_R[2, 0] - matrix_R[0, 2],
                matrix_R[0, 1] - matrix_R[1, 0],
            ],
            [
                matrix_R[1, 2] - matrix_R[2, 1],
                matrix_R[0, 0] - matrix_R[1, 1] - matrix_R[2, 2],
                matrix_R[0, 1] + matrix_R[1, 0],
                matrix_R[0, 2] + matrix_R[2, 0],
            ],
            [
                matrix_R[2, 0] - matrix_R[0, 2],
                matrix_R[0, 1] + matrix_R[1, 0],
                -matrix_R[0, 0] + matrix_R[1, 1] - matrix_R[2, 2],
                matrix_R[1, 2] + matrix_R[2, 1],
            ],
            [
                matrix_R[0, 1] - matrix_R[1, 0],
                matrix_R[0, 2] + matrix_R[2, 0],
                matrix_R[1, 2] + matrix_R[2, 1],
                -matrix_R[0, 0] - matrix_R[1, 1] + matrix_R[2, 2],
            ],
        ]
    )
    eigenValues, eigenVectors = np.linalg.eig(matrix_F)
    try:
        eigenVectors = [
            x for (y, x) in sorted(zip(eigenValues, eigenVectors.transpose()))
        ]
    except Exception as exc:
        print("Warning - Perhaps have degenerate eigenVectors")
        eigenVectors = [x for x in eigenVectors.transpose()]
    eigenValues = sorted(eigenValues)
    # the expression for the best-fit RMSD requires the the largest eigenvalue
    # in cases in which improper rotations are allowed we must also consider
    # the smallest eigenvaue and test if
    # -lamba_min > lambda_max
    # if so we need the quaternion associated with this eigenvalue (overall we need -U(q4))
    # this should allow the construction of this without the need to know we performed an inversion
    # Alternatively, q(4) is the q(1) of the inverted coordinates -r
    # we can use this fact to still use q*r*qc=(0,U(q)*(-r)), if we keep track of the inversion factor
    list(eigenValues)
    q = -1
    m_lambda = eigenValues[q]
    quaternion = tuple(eigenVectors[q])
    factor = +1.0
    # If inversion is allowed test for it:
    if invert == True:
        if -eigenValues[3] > eigenValues[0]:
            q = 0
            m_lambda = -eigenValues[q]
            quaternion = tuple(eigenVectors[q])
            factor = -1.0  # used to invert the coordinates of the original set_pts2

    def best_fit_MSD(l1, l2, max_lambda):
        return (
            sum([list3dot(x, x) for x in l1])
            + sum([list3dot(x, x) for x in l2])
            - 2.0 * max_lambda
        ) / l1.shape[0]

    MSD = best_fit_MSD(set_points1, set_points2, m_lambda)
    if MSD > 0.0:
        return (MSD ** 0.5, quaternion, factor)
    elif MSD < -0.1:
        print("MSD is negative, %s, in overlay of points" % (MSD))
        return (MSD, quaternion, factor)
    else:
        return (0.0, quaternion, factor)


def map_into_hypersphere(v1):  # r=1
    """ maps n dimensional v1 element of [0,1)^n into Real^n, with Norm < r=1
        this method should be attributed to Roger Stafford, but cannot find a paper,
        only seen on Matlab forums. Naming conventions from here -dhc 190115 """
    from scipy.stats import norm
    from scipy.special import gamma, gammainc

    n = float(len(v1))
    # norm.ppf is the percentile point function, i.e. generates normal distribution with mean = 0, sigma = 1
    #    print "before",v1
    v1 = map(norm.ppf, v1)
    #    print "after",v1
    s2 = sum([x * x for x in v1])
    inv_sqrt_s2 = s2 ** -0.5
    # gamma_inc is the (regularized??) incomplete lower Gamma function- don't divide by gamma (online scipy information is ambiguous, but this way works)
    scale = inv_sqrt_s2 * (gammainc(n / 2.0, s2 / 2.0)) ** (1.0 / n)
    return [x * scale for x in v1]
