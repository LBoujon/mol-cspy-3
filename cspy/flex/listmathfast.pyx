import math
from libc.stdlib cimport malloc, free
import cython
#from libc.math cimport sin, cos, acos

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.

#cdef extern from "math.h":
#
#cdef extern from "math.h":

def test_listmathfast_pointer():
    return True

cdef extern from "math.h":
    double acosf(double theta)
    double sinf(double theta)

cdef rot_c(double vecin0,double vecin1,double vecin2,
           double theta,
           double omega0,double omega1,double omega2):
    cdef double t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t12,t13,t14,t15,t16,t19,t21,t18,t30
    cdef double res0, res1, res2
    t1=omega0
    t2=t1*t1
    t3=omega1
    t4=t3*t3
    t5=omega2
    t6=t5*t5
    t7=t2+t4+t6
    t12=5.e-1*theta
    t13=sinf(t12)
    t14=t13*t13
    t15=1.0/t14
    t9=4.*t14
    t10=sinf(theta)
    t16=vecin0
    t19=vecin1
    t21=vecin2
    t8=2.5e-1/t7
    t18=t7**5.e-1
    t30=t19*t3
    res0=(t10*(t10*t7*t15*t16-4.*t18*(-t21*t3+t19*t5))+t9*(t16*t2+2.e0*t1*(t30+t21*t5)-t16*(t4+t6)))*t8
    res1=(t9*(-t19*t2+2.e0*t1*t16*t3+t19*(t4-t6)+2.*t21*t3*t5)+t10*(t10*t7*t15*t19+4.e0*t18*(-t1*t21+t16*t5)))*t8
    res2=(t10*(t10*t7*t15*t21-4.*t18*(-t1*t19+t16*t3))+t9*(-t21*(t2+t4-t6)+2.e0*(t1*t16+t30)*t5))*t8
    return [res0,res1,res2]

def rotation_by_q(vecin,theta,omega):
    try:
        return rot_c(vecin[0],vecin[1],vecin[2],theta,omega[0],omega[1],omega[2])
    except ZeroDivisionError:
        return vecin

#def rotation_by_q(vecin,theta,omega):
#    try:
#        res=[0,0,0]
#        t1=omega[0]
#        t2=t1*t1
#        t3=omega[1]
#        t4=t3*t3
#        t5=omega[2]
#        t6=t5*t5
#        t7=t2+t4+t6
#        t12=5.e-1*theta
#        t13=sinf(t12)
#        t14=t13*t13
#        t15=1.0/t14
#        t9=4.*t14
#        t10=sinf(theta)
#        t16=vecin[0]
#        t19=vecin[1]
#        t21=vecin[2]
#        t8=2.5e-1/t7
#        t18=t7**5.e-1
#        t30=t19*t3
#        res[0]=(t10*(t10*t7*t15*t16-4.*t18*(-t21*t3+t19*t5))+t9*(t16*t2+2.e0*t1*(t30+t21*t5)-t16*(t4+t6)))*t8
#        res[1]=(t9*(-t19*t2+2.e0*t1*t16*t3+t19*(t4-t6)+2.*t21*t3*t5)+t10*(t10*t7*t15*t19+4.e0*t18*(-t1*t21+t16*t5)))*t8
#        res[2]=(t10*(t10*t7*t15*t21-4.*t18*(-t1*t19+t16*t3))+t9*(-t21*(t2+t4-t6)+2.e0*(t1*t16+t30)*t5))*t8
#    except ZeroDivisionError:
#        return vecin
#    return res

cdef qmult(double a0 , double a1 , double a2 , double a3 , double b0 , double b1 , double b2 , double b3 ):
    return [a0*b0-a1*b1-a2*b2-a3*b3,a0*b1+a1*b0+a2*b3-a3*b2,a0*b2+a2*b0+a3*b1-a1*b3,a0*b3+a3*b0+a1*b2-a2*b1]

def quaternion_mult(a=[None,None,None,None],b=[None,None,None,None]):
    return qmult(a[0],a[1],a[2],a[3],b[0],b[1],b[2],b[3])

def quaternion_rotatn(v=[None,None,None],q=[None,None,None,None]):
    tempq = qmult(q[0],q[1],q[2],q[3],0.0,v[0],v[1],v[2])
    return qmult(tempq[0],tempq[1],tempq[2],tempq[3],q[0],-q[1],-q[2],-q[3])[1:4]

cdef l3n(double a0,double a1,double a2,
         double b0,double b1,double b2):
    return ((a0-b0)**2+(a1-b1)**2+(a2-b2)**2)**0.5

def list3norm(a=[None,None,None],b=[None,None,None]):
    return l3n(a[0],a[1],a[2],b[0],b[1],b[2])

cdef l3a(double a0,double a1,double a2,
         double b0,double b1,double b2):
    return [a0+b0,a1+b1,a2+b2]
def list3add(a=[None,None,None],b=[None,None,None]):
    return l3a(a[0],a[1],a[2],b[0],b[1],b[2])

cdef l3dn(double a0,double a1,double a2,
          double b0,double b1,double b2):
    cdef double c0, c1, c2, n
    c0 = a0-b0
    c1 = a1-b1
    c2 = a2-b2
  
    n=((c0)**2+(c1)**2+(c2)**2)**0.5

    return (c0/n,c1/n,c2/n)

def list3diffnormalize(tuple a, tuple b):
    return l3dn(a[0],a[1],a[2],b[0],b[1],b[2])


cdef l3d(double a0,double a1,double a2,
         double b0,double b1,double b2):
    return (a0-b0,a1-b1,a2-b2)

def list3diff(a=[None,None,None],b=[None,None,None]):
    return l3d(a[0],a[1],a[2],b[0],b[1],b[2])

cdef l3sub(double a0,double a1,double a2,
         double b0,double b1,double b2):
    return (a0-b0,a1-b1,a2-b2)

def list3subtract(a=[None,None,None],b=[None,None,None]):
    return l3sub(a[0],a[1],a[2],b[0],b[1],b[2])

#
#def list3add(a=[None,None,None],b=[None,None,None]):
#    return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]

cdef l3normalize(double a0,double a1,double a2):
    cdef double n
    n=((a0)**2+(a1)**2+(a2)**2)**0.5
    return (a0/n,a1/n,a2/n)

def list3normalize(a):
    return l3normalize(a[0],a[1],a[2])
#def list3normalize(a=[None,None,None]):
#    import math
#    n=list3norm(a,[0.0,0.0,0.0])
#    return [a[0]/n,a[1]/n,a[2]/n]

cdef l3cross(double a0,double a1,double a2,
         double b0,double b1,double b2):
     return (a1*b2 - a2*b1 , a2*b0 - a0*b2 , a0*b1 - a1*b0)

def list3cross(a=[None,None,None],b=[None,None,None]):
    return l3cross(a[0],a[1],a[2],b[0],b[1],b[2])

cdef l3dot(double a0,double a1,double a2,
         double b0,double b1,double b2):
     return a0*b0 + a1*b1 + a2*b2

def list3dot(a=[None,None,None],b=[None,None,None]):
    return l3dot(a[0],a[1],a[2],b[0],b[1],b[2])

cdef l3mag(double a0,double a1,double a2):
    return ((a0)**2+(a1)**2+(a2)**2)**0.5

def list3mag(a=[None,None,None]):
    return l3mag(a[0],a[1],a[2])

cdef l3mult(double a0,double a1,double a2 , double fac):
    return ( fac * a0 , fac * a1 , fac * a2 )

def list3multiply(a=[None,None,None],fac=None):
    return l3mult(a[0],a[1],a[2],fac)

cdef l3vecmat(double v0,double v1,double v2,double m0,double m1,double m2,double m3,double m4,double m5,double m6,double m7,double m8):
    return (m0*v0 + m3*v1 + m6*v2 , m1*v0 + m4*v1 + m7*v2 , m2*v0 + m5*v1 + m8*v2)

def list3vecmatrix(v=[None,None,None],m=[[None,None,None],[None,None,None],[None,None,None]]):
    return l3vecmat(v[0],v[1],v[2],m[0][0],m[0][1],m[0][2],m[1][0],m[1][1],m[1][2],m[2][0],m[2][1],m[2][2])

cdef l3matmat(double m0,double m1,double m2,double m3,double m4,double m5,double m6,double m7,double m8,double n0,double n1,double n2,double n3,double n4,double n5,double n6,double n7,double n8):
    return ((m0*n0+m1*n3+m2*n6,m0*n1+m1*n4+m2*n7,m0*n2+m1*n5+m2*n8),
            (m3*n0+m4*n3+m5*n6,m3*n1+m4*n4+m5*n7,m3*n2+m4*n5+m5*n8),
            (m6*n0+m7*n3+m8*n6,m6*n1+m7*n4+m8*n7,m6*n2+m7*n5+m8*n8))

def list3matrixmatrix(m=[[None,None,None],[None,None,None],[None,None,None]],n=[[None,None,None],[None,None,None],[None,None,None]]):
    return l3matmat(m[0][0],m[0][1],m[0][2],m[1][0],m[1][1],m[1][2],m[2][0],m[2][1],m[2][2],n[0][0],n[0][1],n[0][2],n[1][0],n[1][1],n[1][2],n[2][0],n[2][1],n[2][2])

#def list3make_all_translations( m=[[None,None,None],[None,None,None],[None,None,None]] , depth=int ):
def list3make_all_translations( m , depth ):
    vector_list = []
    for ia in range (-depth, depth + 1 ): 
        for ib in range (-depth, depth + 1 ): 
            for ic in range (-depth, depth + 1 ): 
                vector_list.append(list3vecmatrix([ia,ib,ic] , m))
    return sorted(vector_list, key = lambda x: x[0]**2 + x[1]**2 + x[2]**2 )

cpdef list3delparallelvecs(list set_vecs , double param):
    cdef int len_vecs
    len_vecs = len(set_vecs)
    cdef int i
    i= 0
    cdef int j
    j=0
    while i < len_vecs :
        j=i+1
        while j < len_vecs :
#            if abs(list3dot (set_vecs[i],set_vecs[j])) > param:
            #if abs(set_vecs[i][0] * set_vecs[j][0] + set_vecs[i][1] * set_vecs[j][1] + set_vecs[i][2] * set_vecs[j][2]) > param:
            if abs(l3dot(set_vecs[i][0],set_vecs[i][1],set_vecs[i][2],set_vecs[j][0],set_vecs[j][1],set_vecs[j][2])) > param:
                del set_vecs[j]
                len_vecs-=1
            else:
                j+=1
        i+=1
    return set_vecs

def list3edge_vecs(edges , op_mat , op_vec):
    vecs_edges = []
    for i in range ( 0 , len(edges) ) :
        for j in range ( 0 , i ) : #NOTE that for op= x,y,z we may have trouble when going back to above line as edges overlap
            vecs_edges.append(list3normalize( list3cross( list3subtract( edges[i][1],edges[i][0] ),list3subtract(list3apply_sym(edges[j][1],op_mat,op_vec),list3apply_sym(edges[j][0],op_mat,op_vec)))))
    return vecs_edges

def list3square_trace(listpoints):
    #Tr (listpoints^T . listpoints)
    cdef int i, len_pts
    cdef double x = 0.0
    len_pts = len(listpoints)
    for i in range(len_pts):
        x += listpoints[i][0] * listpoints[i][0]
        x += listpoints[i][1] * listpoints[i][1]
        x += listpoints[i][2] * listpoints[i][2]
    return x

def list3list_dot(listpoints1, listpoints2):
    #listpoints1 ^ T . listpoints2
    cdef int i, len_pts
    cdef double *Rmat  = <double *>malloc( 9 * sizeof(double))
    cdef double l10, l11, l12, l20, l21, l22
    len_pts = len(listpoints1)
    for i in range(9):
        Rmat[i] = 0.0
    for i in range(len_pts):
        l10 = listpoints1[i][0]
        l11 = listpoints1[i][1]
        l12 = listpoints1[i][2]
        l20 = listpoints2[i][0]
        l21 = listpoints2[i][1]
        l22 = listpoints2[i][2]
        Rmat[0] += l10 * l20
        Rmat[1] += l11 * l20
        Rmat[2] += l12 * l20
        Rmat[3] += l10 * l21
        Rmat[4] += l11 * l21
        Rmat[5] += l12 * l21
        Rmat[6] += l10 * l22
        Rmat[7] += l11 * l22
        Rmat[8] += l12 * l22
    try:
        return ((Rmat[0], Rmat[3], Rmat[6]), (Rmat[1], Rmat[4], Rmat[7]), (Rmat[2], Rmat[5], Rmat[8]))
    finally:
        free(Rmat)

def list3list_add(list_vecs, add_vec):
#  commented code doesn't seem to work
#    cdef int i, len_vecs
#    len_vecs = len(list_vecs)
#    cdef double *cadd_vec  = <double *>malloc( 3 * sizeof(double))
#    cdef double *cvec  = <double *>malloc( 3 * sizeof(double))
#    for i in range(3):
#        cadd_vec[i] = add_vec[i]
#    for i in range(len_vecs):
#        cvec[0] = list_vecs[i][0]
#        cvec[1] = list_vecs[i][1]
#        cvec[2] = list_vecs[i][2]
#        list_vecs[i] = (cvec[0] + cadd_vec[0], cvec[1] + cadd_vec[1], cvec[2] + cadd_vec[2] )
#    free(cvec)
#    free(cadd_vec)
#    return list_vecs
#    return [list3add(vec,add_vec)  for vec in list_vecs]
    a0, a1, a2 = add_vec[0],  add_vec[1],  add_vec[2]
    return [(vec[0] + a0, vec[1] + a1, vec[2] + a2 )  for vec in list_vecs]

cdef l3sym(double v0 ,double v1 ,double v2 ,double m0 ,double m1 ,double m2 ,double m3 ,double m4 ,double m5 ,double m6 ,double m7 ,double m8 ,double w0 ,double w1 ,double w2 ):
    return [m0*v0 + m3*v1 + m6*v2 + w0 , m1*v0 + m4*v1 + m7*v2 + w1 , m2*v0 + m5*v1 + m8*v2 + w2]

def list3apply_sym(vec, op_mat , op_vec):
    return l3sym(vec[0],vec[1],vec[2],op_mat[0][0],op_mat[0][1],op_mat[0][2],op_mat[1][0],op_mat[1][1],op_mat[1][2],op_mat[2][0],op_mat[2][1],op_mat[2][2],op_vec[0],op_vec[1],op_vec[2])

def list3apply_sym_list_norm(vec_list , op_mat , op_vec):
    new_vec_list = [list3normalize(l3sym(vec[0],vec[1],vec[2],op_mat[0][0],op_mat[0][1],op_mat[0][2],op_mat[1][0],op_mat[1][1],op_mat[1][2],op_mat[2][0],op_mat[2][1],op_mat[2][2],op_vec[0],op_vec[1],op_vec[2]))  for vec in vec_list]
    return new_vec_list

def list3apply_sym_list(vec_list , op_mat , op_vec):
    new_vec_list = [l3sym(vec[0],vec[1],vec[2],op_mat[0][0],op_mat[0][1],op_mat[0][2],op_mat[1][0],op_mat[1][1],op_mat[1][2],op_mat[2][0],op_mat[2][1],op_mat[2][2],op_vec[0],op_vec[1],op_vec[2])  for vec in vec_list]
    return new_vec_list

cdef l3rdv(double o, double v0, double v1, double v2, double a0, double a1, double a2):
    cdef double drdvt = (v0 * a0 + v1 * a1 + v2 * a2) / ( v0 * v0 + v1 * v1 + v2 * v2)**0.5 + 0.0000001
    return (a0 * o / drdvt, a1 * o / drdvt, a2 * o / drdvt)

def list3required_dvec(ov, vec, ax):
    return l3rdv(ov, vec[0], vec[1], vec[2], ax[0], ax[1], ax[2])

def list3find_overlap(EP1, EP2):
    return l3find_overlap(EP1[0], EP1[1], EP2[0], EP2[1])

cdef l3find_overlap(double EP10, double EP11, double EP20, double EP21):
    if( EP10 > EP20 ):
        if( EP11 < EP20):
            if( EP11 > EP21): 
                return abs(EP20 - EP11)
            else: 
                if abs(EP10 - EP21) <= abs(EP11 -EP20):
                    return abs(EP10 - EP21)
                else:
                    return abs(EP11 - EP20)
        else: 
            return 0.0
    else:
        if( EP10 > EP21 ):
            if( EP11 < EP21):
                return abs(EP10 - EP21)
            else: 
                if abs(EP10 - EP21) <= abs(EP11 -EP20):
                    return abs(EP10 - EP21)
                else:
                    return abs(EP11 - EP20)
        else:
             return 0.0



def list3perp_face_vecs(faces):
    perp_vectors = []
    for face_pt in faces:
        perp_vectors.append(list3normalize ( list3cross(list3subtract(face_pt[2],face_pt[1]),list3subtract(face_pt[1],face_pt[0]))))
    return perp_vectors

#from libcpp.vector cimport vector
def list3_proj_pts_axis ( axis , set_vertices , set_radii):
    cdef double proj , my_min , my_max #,axis0,axis1,axis2
    cdef int i, len_verts, len_radii
    cdef double *caxis  = <double *>malloc( 3 * sizeof(double))
    len_verts = len(set_vertices)
    len_radii = len(set_radii)
    cdef double *cverts = <double *>malloc( len_verts * 3 * sizeof(double))
    cdef double *cradii = <double *>malloc( len_radii * sizeof(double))
    for i in range(3):
        caxis [ i ] = axis[i]
    for i in range(len_verts):
        cverts [ i * 3 ]    = set_vertices[i][0]
        cverts [ i * 3 + 1] = set_vertices[i][1]
        cverts [ i * 3 + 2] = set_vertices[i][2]
    for i in range(len_radii):
        cradii [ i ] = set_radii[ i ]
    proj = caxis[ 0 ] * cverts [ 0 ] + caxis[ 1 ] * cverts [ 1 ] + caxis[ 2 ] * cverts [ 2 ] 
    my_min = proj - cradii[0]
    my_max = proj + cradii[0]
    for i in range ( 1, len_verts):
        proj = caxis[ 0 ] * cverts[ i * 3 ] + caxis[ 1 ] * cverts [ i * 3 + 1 ] + caxis[ 2 ] * cverts [ i * 3 + 2 ] 
        if proj - cradii [ i ] < my_min :
            my_min = proj - cradii [ i ] 
        if proj + cradii [ i ] > my_max :
            my_max = proj + cradii [ i ] 
    free(cradii)
    free(cverts)
    free(caxis)
    return my_max , my_min


cpdef list3cross_combinations( set_vecs1, set_vecs2):
    set_out = []
    cdef int i, j, len1, len2
    len1 = len(set_vecs1) 
    len2 = len(set_vecs2) 
    set_vecs1n = [l3normalize(x[0], x[1], x[2]) for x in set_vecs1]
    set_vecs2n = [l3normalize(x[0], x[1], x[2]) for x in set_vecs2]
    for i in range(len1):
#            for j in range ( 0 , i ): # should this be i ?? test this- do both ways and delete parallel- are these same?
        for j in range(len2):
            try:
                if abs(list3dot(set_vecs1n[i], set_vecs2n[j])) < 1.0:
                    set_out.append(list3normalize(list3cross( set_vecs1[i] , set_vecs2[j] )))
#                set_out.append(list3normalize(list3cross( list3subtract(set_vecs1[i][1],set_vecs1[i][0]) , list3subtract(set_vecs2[j][1],set_vecs2[j][0]) )))
            except:
                pass
    return set_out    

def list3cross_comb_tri(set_vecs1 , set_vecs2):
    cdef int i,j,len_vecs
    len_vecs = len(set_vecs1)
    cdef double *cvecs1 = <double *>malloc(len_vecs * 3 * sizeof(double))
    cdef double *cvecs2 = <double *>malloc(len_vecs * 3 * sizeof(double))
    for i in range(len_vecs):
        cvecs1 [ i * 3 ]    = set_vecs1[i][0]
        cvecs1 [ i * 3 + 1] = set_vecs1[i][1]
        cvecs1 [ i * 3 + 2] = set_vecs1[i][2]
        cvecs2 [ i * 3 ]    = set_vecs2[i][0]
        cvecs2 [ i * 3 + 1] = set_vecs2[i][1]
        cvecs2 [ i * 3 + 2] = set_vecs2[i][2]
    set_out = []
    for i in range( len_vecs ):
        for j in range( i ):
            if (cvecs1[i * 3] * cvecs2[j* 3] + cvecs1[i * 3 + 1] * cvecs2[j * 3 + 1] + cvecs1[i * 3 + 2] * cvecs2[j * 3 + 2]) ** 2.0 < \
                   (cvecs1[i * 3] ** 2.0 + cvecs1[i * 3 + 1] ** 2.0 + cvecs1[i * 3 + 2] ** 2.0) * \
                   (cvecs2[j * 3] ** 2.0 + cvecs2[j * 3 + 1] ** 2.0 + cvecs2[j * 3 + 2] ** 2.0)  :
                set_out.append(list3normalize(list3cross( set_vecs1[i] , set_vecs2[j] )))
    free(cvecs2)
    free(cvecs1)
    return set_out

def list3edge_vecs(edges,op_mat,op_vec):
    vecs_edges = []
    for i in range ( 0 , len(edges) ) :
        for j in range ( 0 , i ) : #NOTE that for op= x,y,z we may have trouble when going back to above line as edges overlap
            vecs_edges.append(list3normalize( list3cross( list3subtract( edges[i][1],edges[i][0] ),list3subtract(list3add(list3vecmatrix(edges[j][1],op_mat),op_vec),list3add(list3vecmatrix(edges[j][0],op_mat),op_vec)))))
    return vecs_edges

#def list3_proj_pts_axis( axis , pts , size_pts):
##    tv = l3ppa( axis , len(axis) , pts , size_pts , len(pts)  )
#    tv = l3ppa( axis ,  pts , size_pts  )
#    return tv[0] , tv[1]

#cdef l3ppa( axis , set_vertices , set_radii):
#        float proj = l3dot(axis[0],axis[1],axis[2] ,set_vertices[0],set_vertices[1],set_vertices[2] );
#        float my_min = proj - set_radii[0];
#        float my_max = proj + set_radii[0];
#        for (int i=0 , length=sizeof(set_vertices); i<length; ++i ) 
#        {
#            proj = l3dot(axis[0],axis[1],axis[2] , set_vertices[i][0],set_vertices[i][1],set_vertices[i][2] );
#            if proj - set_radii [ i ] < my_min :
#            {
#                my_min = proj - set_radii [ i ];
#            }
#            if proj + set_radii [ i ] > my_max :
#            {
#                my_max = proj + set_radii [ i ];
#            }
#        }
#        return [my_max , my_min]

cpdef flist3angle(a, b, c):
    cdef double d, mu, mv
    mu = l3n(a[0],a[1],a[2],b[0],b[1],b[2])
    mv = l3n(c[0],c[1],c[2],b[0],b[1],b[2])
    d = l3dot(a[0]-b[0],a[1]-b[1],a[2]-b[2],c[0]-b[0],c[1]-b[1],c[2]-b[2])
    return acosf(d/(mu*mv))

cpdef flist3dihedral(tuple a1, tuple a2, tuple a3, tuple a4):
    b1 = list3subtract(a2, a1)
    b2 = list3subtract(a3, a2)
    b3 = list3subtract(a4, a3)
    y = list3dot(list3cross(list3cross(b1, b2), list3cross(b2, b3)), b2) / list3mag(b2)
    x = list3dot(list3cross(b1, b2), list3cross(b2, b3))
    return math.atan2(y, x)

cpdef fdihedral_st_vectors_xyz(a1, a2, a3, a4):
    import math
    e12 = list3normalize(list3diff(a2, a1))
    e23 = list3normalize(list3diff(a3, a2))
    e43 = list3normalize(list3diff(a3, a4))
    e32 = list3normalize(list3diff(a2, a3))

    fa = 1.0
    r12 = list3norm(a1, a2)*fa
    r23 = list3norm(a2, a3)*fa

    r43 = list3norm(a4, a3)*fa
    r32 = list3norm(a3, a2)*fa

    #print a1, a2, a3
    ang123 = flist3angle(a1, a2, a3)
    sang123 = math.sin(ang123)
    cang123 = math.cos(ang123)
    ang234 = flist3angle(a2, a3, a4)
    sang234 = math.sin(ang234)
    cang234 = math.cos(ang234)

    st1 = list3multiply(list3cross(e12, e23),
                        -1.0/(r12*(sang123**2)))

    st2_1 = (r23-r12*cang123)/(r23*r12*sang123)
    st2_2 = list3multiply(list3cross(e12, e23), 1.0/sang123)
    st2_3 = cang234/(r23 * sang234)
    st2_4 = list3multiply(list3cross(e43, e32), 1.0/sang234)
    st2 = list3add(list3multiply(st2_2, st2_1),
                   list3multiply(st2_4, st2_3))

    st3_1 = (r32-r43*cang234)/(r32*r43*sang234)
    st3_2 = list3multiply(list3cross(e43, e32), 1.0/sang234)
    st3_3 = cang123/(r32*sang123)
    st3_4 = list3multiply(list3cross(e12, e23), 1.0/sang123)
    st3 = list3add(list3multiply(st3_2, st3_1),
                   list3multiply(st3_4, st3_3))

    st4 = list3multiply(list3cross(e43, e32),
                        -1.0/(r43*sang234*sang234))

    return (st1, st2, st3, st4)

cpdef improper_value(tuple a1, tuple  a2,tuple  a3,tuple  a4): 

    e41 = list3diffnormalize(a1, a4)
    e42 = list3diffnormalize(a2, a4)
    e43 = list3diffnormalize(a3, a4)
    ang243 = flist3angle(a2, a4, a3)

    e42xe43 = list3cross(e42,e43)

    return math.asin(list3dot(list3multiply(e42xe43, 1./math.sin(ang243)),e41))


cpdef set_impropertorsion_st_vectors_fast(tuple a1, tuple  a2,tuple  a3,tuple  a4, int n_a, int n_b, int n_c, int n_d,
                                   Bprim, q, int internal_index): 

    cdef double r12, r23, r43, r32
    cdef double ang123, cang123, sang123
    cdef double ang234, cang234, sang234
    cdef double st2_1, st2_3, st3_1, st3_3

    e41 = list3diffnormalize(a1, a4)
    e42 = list3diffnormalize(a2, a4)
    e43 = list3diffnormalize(a3, a4)
    ang243 = flist3angle(a2, a4, a3)
    r41 = list3norm(a4, a1)
    r42 = list3norm(a4, a2)
    r43 = list3norm(a4, a3)

    e42xe43 = list3cross(e42,e43)
    e43xe41 = list3cross(e43,e41)
    e41xe42 = list3cross(e41,e42)

    q[internal_index] = math.asin(list3dot(list3multiply(e42xe43, 1./math.sin(ang243)),e41))

    st1 = list3multiply(list3subtract(list3multiply(e42xe43, 1./(math.cos(q[internal_index])*math.sin(ang243)) )
                                      , list3multiply(e41, math.tan(q[internal_index])))
                        ,1./r41)

    st2 = list3multiply(list3subtract(list3multiply(e43xe41, 1./(math.cos(q[internal_index])*math.sin(ang243)) )
                                      , list3multiply(list3subtract(e42
                                                                    ,list3multiply(e43, math.cos(ang243)))
                                                      , math.tan(q[internal_index])/(math.sin(ang243)**2)))
                        ,1./r42)

    st3 = list3multiply(list3subtract(list3multiply(e41xe42, 1./(math.cos(q[internal_index])*math.sin(ang243)) )
                                      , list3multiply(list3subtract(e43
                                                                    ,list3multiply(e42, math.cos(ang243)))
                                                      , math.tan(q[internal_index])/(math.sin(ang243)**2)))
                        ,1./r43)

    st4 = list3multiply(st1,-1.)
    st4 = list3subtract(st4,st2)
    st4 = list3subtract(st4,st3)

    Bprim[internal_index, n_a*3+0] = st1[0]
    Bprim[internal_index, n_a*3+1] = st1[1]
    Bprim[internal_index, n_a*3+2] = st1[2]
    Bprim[internal_index, n_b*3+0] = st2[0]
    Bprim[internal_index, n_b*3+1] = st2[1]
    Bprim[internal_index, n_b*3+2] = st2[2]
    Bprim[internal_index, n_c*3+0] = st3[0]
    Bprim[internal_index, n_c*3+1] = st3[1]
    Bprim[internal_index, n_c*3+2] = st3[2]
    Bprim[internal_index, n_d*3+0] = st4[0]
    Bprim[internal_index, n_d*3+1] = st4[1]
    Bprim[internal_index, n_d*3+2] = st4[2]


cpdef set_dihedral_st_vectors_fast(tuple a1,tuple  a2,tuple  a3,tuple  a4, int n_a, int n_b, int n_c, int n_d,
                                   Bprim, q, int internal_index): 

    cdef double r12, r23, r43, r32
    cdef double ang123, cang123, sang123
    cdef double ang234, cang234, sang234
    cdef double st2_1, st2_3, st3_1, st3_3


    q[internal_index] = flist3dihedral(a1, a2, a3, a4)

    e12 = list3diffnormalize(a2, a1)
    e23 = list3diffnormalize(a3, a2)
    e43 = list3diffnormalize(a3, a4)
    e32 = list3diffnormalize(a2, a3)

    r12 = list3norm(a1, a2)
    r23 = list3norm(a2, a3)

    r43 = list3norm(a4, a3)
    r32 = list3norm(a3, a2)

    #print a1, a2, a3
    ang123 = flist3angle(a1, a2, a3)
    sang123 = sinf(ang123)
    cang123 = math.cos(ang123)
    ang234 = flist3angle(a2, a3, a4)
    sang234 = sinf(ang234)
    cang234 = math.cos(ang234)

    st1 = list3multiply(list3cross(e12, e23),
                        -1.0/(r12*(sang123**2)))

    st2_1 = (r23-r12*cang123)/(r23*r12*sang123)
    st2_2 = list3multiply(list3cross(e12, e23), 1.0/sang123)
    st2_3 = cang234/(r23 * sang234)
    st2_4 = list3multiply(list3cross(e43, e32), 1.0/sang234)
    st2 = list3add(list3multiply(st2_2, st2_1),
                   list3multiply(st2_4, st2_3))

    st3_1 = (r32-r43*cang234)/(r32*r43*sang234)
    st3_2 = list3multiply(list3cross(e43, e32), 1.0/sang234)
    st3_3 = cang123/(r32*sang123)
    st3_4 = list3multiply(list3cross(e12, e23), 1.0/sang123)
    st3 = list3add(list3multiply(st3_2, st3_1),
                   list3multiply(st3_4, st3_3))

    st4 = list3multiply(list3cross(e43, e32),
                        -1.0/(r43*sang234*sang234))

    Bprim[internal_index, n_a*3+0] = st1[0]
    Bprim[internal_index, n_a*3+1] = st1[1]
    Bprim[internal_index, n_a*3+2] = st1[2]
    Bprim[internal_index, n_b*3+0] = st2[0]
    Bprim[internal_index, n_b*3+1] = st2[1]
    Bprim[internal_index, n_b*3+2] = st2[2]
    Bprim[internal_index, n_c*3+0] = st3[0]
    Bprim[internal_index, n_c*3+1] = st3[1]
    Bprim[internal_index, n_c*3+2] = st3[2]
    Bprim[internal_index, n_d*3+0] = st4[0]
    Bprim[internal_index, n_d*3+1] = st4[1]
    Bprim[internal_index, n_d*3+2] = st4[2]

cpdef set_angle_st_vectors_fast(tuple a1, tuple a2, tuple a3, int n_a, int n_b, int n_c, Bprim, int internal_index):
    cdef double alpha, r31, r32
    #Atoms in bond number 1-3-2 in Wilson, Molecular Vibrations, 1955
    # this is NOT a bug, leave the order intact
    alpha = flist3angle(a1, a3, a2)

    e31 = list3normalize(list3diff(a1, a3))
    e32 = list3normalize(list3diff(a2, a3))

    r31 = list3norm(a3, a1)
    r32 = list3norm(a3, a2)

    st1 = list3multiply(list3diff(list3multiply(e31, math.cos(alpha)),
                                  e32), 1.0/(r31 * math.sin(alpha)))
    st2 = list3multiply(list3diff(list3multiply(e32, math.cos(alpha)),
                                  e31), 1.0/(r32 * math.sin(alpha)))
    st3 = list3diff(list3multiply(st1, -1.0), st2)

    Bprim[internal_index, n_a*3+0] = st1[0]
    Bprim[internal_index, n_a*3+1] = st1[1]
    Bprim[internal_index, n_a*3+2] = st1[2]
    Bprim[internal_index, n_b*3+0] = st3[0]
    Bprim[internal_index, n_b*3+1] = st3[1]
    Bprim[internal_index, n_b*3+2] = st3[2]
    Bprim[internal_index, n_c*3+0] = st2[0]
    Bprim[internal_index, n_c*3+1] = st2[1]
    Bprim[internal_index, n_c*3+2] = st2[2]

cpdef set_bond_st_vectors_fast(tuple a1, tuple a2, int n_a, int n_b, Bprim, int internal_index):
    cdef double alpha, r31, r32

    st1 = list3normalize(list3diff(a1, a2))

    Bprim[internal_index, n_a*3+0] = st1[0]
    Bprim[internal_index, n_a*3+1] = st1[1]
    Bprim[internal_index, n_a*3+2] = st1[2]
    Bprim[internal_index, n_b*3+0] = -st1[0]
    Bprim[internal_index, n_b*3+1] = -st1[1]
    Bprim[internal_index, n_b*3+2] = -st1[2]
