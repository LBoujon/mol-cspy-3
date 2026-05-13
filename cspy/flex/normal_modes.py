import logging
import math
from collections import namedtuple
from operator import itemgetter

import numpy
import scipy

from cspy.flex.listmath import list3subtract

logger = logging.getLogger('molecule_comparisons.normal_modes')

NormalModeDisplacement = namedtuple('NormalModeDisplacement',
                                    ['mapping', 'factors'])

FminDisplacementResult = namedtuple('FminDisplacementResult',
                                    ['final_U', 'final_s'])


def pirange(x):
    if x > math.pi:
        return x - math.pi * 2.0
    elif x <= -math.pi:
        return x + math.pi * 2.0
    else:
        return x


def generalised_inverse(M):
    return numpy.linalg.pinv(M)
    # return numpy.dot(numpy.linalg.inv(numpy.dot(M, M.transpose())), M)


def non_zero_eigenvalues_matrix(M, threshold, report=False):
    G = numpy.dot(M, M.transpose())
    w, v = numpy.linalg.eig(G)
    idx = w.argsort()[::1]
    w = w[idx]
    v = v[:, idx]
    n = 0
    K = []
    L = []
    for i in range(len(w)):
        if w[i] > threshold:
            if report:
                logger.info('Keeping', w[i])
            n += 1
            K.append(v[:, i].real.tolist())
            L.append(w[i])
        else:
            if report:
                logger.info('Removing', w[i])
                logger.info(v[:, i].real.tolist())
            pass
    K = numpy.array(K)
    L = numpy.array(L)
    if report:
        logger.info('Eigenvalues', w.shape)
        logger.info(w)
        return K, L
    else:
        return K


def localize_K_matrix(K_base, localization_keys, all_keys):
    logger.info('Localising Basis')
    lk = list(localization_keys)
    logger.info('Localization_keys', lk)
    local_space = K_base.copy()
    for il in lk:  # [0,1,2,3,4,5,6,7]:#range(6):
        drop_row = []
        C = numpy.zeros(K_base.shape[1])
        C[il] = 1.
        Cproj = numpy.zeros(K_base.shape[1])
        for Uk in local_space:
            Cproj += numpy.dot(numpy.dot(C, Uk), Uk)

        if numpy.linalg.norm(Cproj) > 0:
            logger.info('Schmidt Othogonalisation of ', il, all_keys[il],
                        numpy.linalg.norm(Cproj))
            Cproj = Cproj / numpy.linalg.norm(Cproj)
            logger.info(C)
            logger.info(Cproj)
            local_space = numpy.vstack([Cproj, local_space])
        else:
            logger.info('Eliminating ', il, all_keys[il],
                        'due to zero projection')
            drop_row.append([0])

        def proj(a, b):
            return numpy.dot(numpy.dot(a, b), b)

        # Modified Gram-Schmidt
        for i in range(local_space.shape[0]):
            # local_space[i] = local_space[i]/numpy.linalg.norm(local_space[i])
            for j in range(0, i):
                local_space[i] = local_space[i] - proj(local_space[i],
                                                       local_space[j])
            m = numpy.linalg.norm(local_space[i])
            logger.info(i, m)
            if m > 1e-3:
                local_space[i] = local_space[i] / m
                pass
            else:
                logger.info('Dropping row', i)
                drop_row.append(i)
        if len(drop_row) != 1:
            logger.info(local_space)
            logger.info('Bad drop_row', drop_row)
            exit()

        logger.info('Non-redundant Coordinate', i)
        logger.info(local_space[i])
        logger.info('Localised Space Before removal of row: ', drop_row)
        logger.info(numpy.round(local_space, 3))
        local_space = numpy.delete(local_space, drop_row, 0)
        logger.info('Primitive weighting in Delocalized Coorindates')
        for s in range(len(all_keys)):
            logger.info(s, all_keys[s],
                        numpy.linalg.norm(local_space[:, s]) ** 2)
    return local_space

    for i in lk:  # [0,1,2,3,4,5,6,7]:#range(6):
        for k in range(local_space.shape[0]):
            proj = numpy.zeros(local_space[k].shape)
            uk = local_space[k].copy()
            for l in range(0, k):
                proj += numpy.dot(numpy.dot(local_space[k], local_space[l]),
                                  local_space[l])
            if numpy.linalg.norm(proj) < 1e-6:
                pass
                # print 'Zero Projection',k
            else:
                # local_space[k] = uk.copy()
                local_space[k] = local_space[k] - proj
                m = numpy.linalg.norm(local_space[k])
                logger.info(k, m)
                if m > 1e-6:
                    local_space[k] = local_space[k] / m
                    pass
                else:
                    logger.info('Dropping row', k)
                    drop_row.append(k)
            # if m > 0:
            #    local_space[k] = local_space[k] / numpy.linalg.norm(local_space[k])
            #    if k!=0 and abs(local_space[k,i]) > 1e-9:
            #        drop_row.append(k)
            # else:
            #    print 'New coordinate',k,' has no magnitude'
        if not drop_row:
            logger.info('no drop_row', drop_row)
            exit()
        logger.info('Non-redundant Coordinate', k)
        logger.info(local_space[k])
        logger.info('Localised Space Before removal of row: ', drop_row)
        logger.info(numpy.round(local_space, 3))
        local_space = numpy.delete(local_space, drop_row, 0)
        logger.info('Primitive weighting in Delocalized Coorindates')
        for s in range(len(all_keys)):
            logger.info(s, all_keys[s],
                        numpy.linalg.norm(local_space[:, s]) ** 2)
    return local_space


def initialise_s(nm_mapping, nm_starting, length):
    if nm_mapping is None:
        nm_mapping = []
    s = [0 for _ in range(length)]
    for i, v in zip(nm_mapping, nm_starting):
        s[i] = v
    return s


def distance_in_non_redundant_internals(base, target):
    return fmin_displacements(base, target,
                              nm_mapping=[], nm_starting=[],
                              minimize_U=False)


# Make U=|q_target_nr-q_nr_s|
class NormalModeDisplacementDifferenceEngine(object):
    def __init__(self):
        self.normal_mode_mapping = []
        self.base_number_of_atoms = None
        self.base_cartesian_normal_modes = None
        self.base_cartesian_to_primitives = None
        self.base_primitives_to_non_redundants = None
        self.base_non_redundants_to_cartesian = None
        self.base_primitives_values = None
        self.target_primitives_values = None
        self.print_level = 0
        self.num_bonds = 0
        self.lstsq_solution = False

    def solve_inf_displacements(self, number_of_modes):
        if number_of_modes == 0:
            return numpy.array([])
        q_diff = numpy.array(map(pirange, [
            self.target_primitives_values[i] - self.base_primitives_values[i]
            for i in range(len(self.base_primitives_values))]))
        # q_diff = numpy.zeros(len(self.base_primitives_values))
        # q_diff[0] = 0.5

        q_nr_diff = numpy.dot(self.base_primitives_to_non_redundants, q_diff)
        # print self.base_non_redundants_to_cartesian.shape, q_nr_diff.shape
        dX = numpy.dot(self.base_non_redundants_to_cartesian.transpose(),
                       q_nr_diff)
        # dX = numpy.dot(generalised_inverse(self.base_cartesian_to_primitives), q_diff)
        # print dX
        if self.lstsq_solution:
            nm_displacements = numpy.linalg.lstsq(
                self.base_cartesian_normal_modes[:number_of_modes].transpose(),
                dX)[0]
        else:
            nm_displacements = numpy.dot(generalised_inverse(
                self.base_cartesian_normal_modes[
                :number_of_modes].transpose()), dX)
        # Check the solution for linear bends
        new_q_prim = numpy.dot(self.base_cartesian_to_primitives,
                               numpy.dot(nm_displacements,
                                         self.base_cartesian_normal_modes[
                                         :number_of_modes]))
        # print numpy.linalg.norm(new_q_prim), new_q_prim
        for i in range(self.num_bonds, self.num_bonds + self.num_angles):
            if abs(pirange(
                    self.base_primitives_values[i] + new_q_prim[i])) > 3.0:
                logger.info('internal became near linear', i, abs(
                    pirange(self.base_primitives_values[i] + new_q_prim[i])))
                raise CSPyException('internal will become near linear')

        return nm_displacements

    def U_difference(self, nm_displacements):
        s = initialise_s(self.normal_mode_mapping, nm_displacements,
                         3 * self.base_number_of_atoms - 6)
        logger.info(s)
        logger.info(self.base_cartesian_normal_modes)
        sC = numpy.dot(s, self.base_cartesian_normal_modes)
        # Make delta_q_nr = B_nr.sC
        delta_base_primitives_values = numpy.dot(
            self.base_cartesian_to_primitives, sC)
        # Make q_nr = K_base.q_primitive
        # q_nr = numpy.dot(K_base, q_primitive)
        # Make q_nr_s = q_nr + delta_q_nr
        displaced_base_primitives_values = self.base_primitives_values + delta_base_primitives_values

        difference_primitives_values = self.target_primitives_values - displaced_base_primitives_values
        difference_primitives_values = map(pirange,
                                           difference_primitives_values.tolist())

        difference_non_redundants_values = numpy.dot(
            self.base_primitives_to_non_redundants,
            difference_primitives_values)

        U = numpy.linalg.norm(difference_non_redundants_values)
        # for i,v in enumerate(diff_primitives):
        #    if abs(v) > 0.5:
        #       print i, all_keys[i], v
        #       print target.angle_value(all_keys[i][0], all_keys[i][1],all_keys[i][2])
        #       print target.angle_value(all_keys[i][1], all_keys[i][2],all_keys[i][3])
        #       #raise Exception
        # logger.debug('Displacement:'+str(nm_displacements)+' U:'+str(U))
        if self.print_level > 0:
            logger.info(
                'Displacement:' + str(nm_displacements) + ' U:' + str(U))
        return U


# Make U=|q_target_nr-q_nr_s|
def Us(nm_displacements):
    s = initialise_s(nm_mapping, nm_displacements, 3 * base.num_atoms() - 6)
    sC = numpy.dot(s, C)
    # Make delta_q_nr = B_nr.sC
    delta_q_nr = numpy.dot(B_nr, sC)
    # Make q_nr = K_base.q_primitive
    q_nr = numpy.dot(K_base, q_primitive)
    # Make q_nr_s = q_nr + delta_q_nr
    q_nr_s = q_nr + delta_q_nr

    U = numpy.linalg.norm(q_target_nr - q_nr_s)
    # for i,v in enumerate((q_target_nr-q_nr_s).tolist()):
    #    if abs(v) > 1:
    #       print i
    #       print len(K_base.tolist()[i])
    #       for xi, x in enumerate(K_base.tolist()[i]):
    #           if abs(x) > 1.:
    #              print xi
    # raise Exception
    logger.debug('Displacement:' + str(nm_displacements) + ' U:' + str(U))
    logger.info('Displacement:' + str(nm_displacements) + ' U:' + str(U))
    return U


def fmin_displacements(base, target,
                       nm_mapping, nm_starting,
                       minimize_U=True, partial_minimize=None,
                       basin_hopping=False, solve_inf_displacements=False,
                       K_base=None):
    numpy.set_printoptions(precision=6, suppress=True, linewidth=200)

    localize_active_internal_coordinates = False

    logger.debug('Starting fmin_displacements')

    # obtain all the primitive keys
    prim_base_keys = base.primitive_keys(linear_thr=175. * math.pi / 180.)
    logger.info('Length of base primitive bond keys ' + str(
        len(prim_base_keys['o_bond_keys'])))
    logger.info('Length of base primitive angle keys ' + str(
        len(prim_base_keys['o_angle_keys'])))
    logger.info('Length of base primitive dihedral keys ' + str(
        len(prim_base_keys['o_dihedral_keys'])))
    prim_base_values = base.primitive_values(prim_base_keys)
    logger.debug('Length of base primitive bond values ' + str(
        len(prim_base_values['o_bond_keys'])))
    logger.debug('Length of base primitive angle values ' + str(
        len(prim_base_values['o_angle_keys'])))
    logger.debug('Length of base primitive dihedral values ' + str(
        len(prim_base_values['o_dihedral_keys'])))
    # target values come from base keys
    prim_target_keys = target.primitive_keys(linear_thr=175. * math.pi / 180.)
    prim_target_values = target.primitive_values(prim_base_keys)

    # perform a check that all the bond keys are the same
    for v in prim_target_keys['o_bond_keys']:
        if v in prim_base_keys['o_bond_keys']:
            pass
        elif tuple(reversed(list(v))) in prim_base_keys['o_bond_keys']:
            pass
        else:
            logger.info(v)
    assert prim_target_keys == prim_base_keys

    all_keys = prim_base_keys['o_bond_keys'] + prim_base_keys['o_angle_keys'] + \
               prim_base_keys['o_dihedral_keys']
    # print all_keys.index((2, 0, 1, 9))
    # Could add improper torsions too with
    # all_keys += prim_base_keys['o_improper_keys']

    # Size and make B_primitive
    # Populate B_primitive with st_primitive
    B_primitive = base.B_primitive(prim_base_keys, prim_base_values)

    # Size and make q_primitive
    # Populate q_primitive
    q_primitive = base.q_primitive(prim_base_keys, prim_base_values)

    # Size and make q_target_primitive
    q_target_primitive = target.q_primitive(prim_base_keys, prim_target_values)

    # Make K_base
    # G = B_primitive.B_primitive^T
    # Eigenvectors/values of G
    # K_base = eigenvectors with non-zero eigenvalues
    if K_base is None:
        K_base = non_zero_eigenvalues_matrix(B_primitive, 1e-9)
        if localize_active_internal_coordinates:
            K_base = localize_K_matrix(K_base, reversed(range(len(all_keys))),
                                       all_keys)

    # Make B_nr = K.B_primitive
    try:
        B_nr = numpy.dot(K_base, B_primitive)
    except ValueError as e:
        logger.info(K_base.shape, B_primitive.shape)
        raise e
    # Make B_nr_inv = ((B_nr.B_nr^T)^-1).B_nr
    B_nr_inv = generalised_inverse(B_nr.transpose())
    B_prim_inv = generalised_inverse(B_primitive)

    # Make sC = s.C
    C = base.cart_dof_array()
    # Make q_target_nr = K_base.q_target_primitive
    q_target_nr = numpy.dot(K_base, q_target_primitive)

    # Make and configure the difference engine
    difference_engine = NormalModeDisplacementDifferenceEngine()
    difference_engine.normal_mode_mapping = nm_mapping
    difference_engine.base_number_of_atoms = base.num_atoms()
    difference_engine.base_cartesian_normal_modes = C
    difference_engine.base_cartesian_to_primitives = B_primitive
    difference_engine.base_non_redundants_to_cartesian = B_nr_inv
    difference_engine.base_primitives_to_non_redundants = K_base
    difference_engine.base_primitives_values = q_primitive
    difference_engine.target_primitives_values = q_target_primitive
    difference_engine.num_bonds = len(prim_base_keys['o_bond_keys'])
    difference_engine.num_angles = len(prim_base_keys['o_angle_keys'])
    if False:
        numpy.savetxt('C.txt', C)
        numpy.savetxt('B_prim.txt', B_primitive)
        numpy.savetxt('K_base.txt', K_base)
        numpy.savetxt('q_primitive.txt', q_primitive)
        numpy.savetxt('q_target_primitive.txt', q_target_primitive)
        exit()

    if basin_hopping:
        difference_engine.print_level = 1

    if solve_inf_displacements:
        s = difference_engine.solve_inf_displacements(len(nm_starting))
        U = difference_engine.U_difference(s)
    elif minimize_U:
        if partial_minimize is not None:
            fmin_result = partial_minimize(difference_engine.U_difference,
                                           nm_starting)
        elif basin_hopping:
            fmin_result = scipy.optimize.basinhopping(
                difference_engine.U_difference, nm_starting,
                minimizer_kwargs={'method': 'Nelder-Mead'},
                disp=True, stepsize=2.0, T=0.5)
        else:
            fmin_result = scipy.optimize.minimize(
                difference_engine.U_difference, nm_starting,
                method='Nelder-Mead', tol=1e-7)
        nm_final = fmin_result.x
        U = difference_engine.U_difference(nm_final)
        s = fmin_result.x
    else:
        U = difference_engine.U_difference(nm_starting)
        s = numpy.array(nm_starting)

    return FminDisplacementResult(U, s)


def report_delocalized_internals(base):
    c = base.centroid()
    for i in range(base.num_atoms()):
        base.atoms[i].xyz = list3subtract(base.atoms[i].xyz, c)
    # obtain all the primitive keys
    numpy.set_printoptions(precision=6, suppress=True, linewidth=200)
    prim_base_keys = base.primitive_keys(linear_thr=175. * math.pi / 180.)
    # prim_base_keys['o_angle_keys'] = []
    # prim_base_keys['o_dihedral_keys'] = []
    prim_base_keys['o_improper_keys'] = []
    logger.info('Length of base primitive bond keys ' + str(
        len(prim_base_keys['o_bond_keys'])))
    logger.info('Length of base primitive angle keys ' + str(
        len(prim_base_keys['o_angle_keys'])))
    logger.info('Length of base primitive dihedral keys ' + str(
        len(prim_base_keys['o_dihedral_keys'])))
    prim_base_keys['o_bond_keys'] = sorted(prim_base_keys['o_bond_keys'],
                                           key=itemgetter(0, 1))
    prim_base_keys['o_angle_keys'] = sorted(prim_base_keys['o_angle_keys'],
                                            key=itemgetter(2, 0))
    prim_base_keys['o_dihedral_keys'] = sorted(
        prim_base_keys['o_dihedral_keys'], key=itemgetter(3, 0))
    logger.info(prim_base_keys)
    prim_base_values = base.primitive_values(prim_base_keys)
    logger.debug('Length of base primitive bond values ' + str(
        len(prim_base_values['o_bond_keys'])))
    logger.debug('Length of base primitive angle values ' + str(
        len(prim_base_values['o_angle_keys'])))
    logger.debug('Length of base primitive dihedral values ' + str(
        len(prim_base_values['o_dihedral_keys'])))
    # perform a check that all the bond keys are the same
    all_keys = prim_base_keys['o_bond_keys'] + prim_base_keys['o_angle_keys'] + \
               prim_base_keys['o_dihedral_keys'] + prim_base_keys[
                   'o_improper_keys']

    B_primitive = base.B_primitive(prim_base_keys, prim_base_values)
    logger.info('B_primitive in Angstrom', B_primitive.shape)
    logger.info(B_primitive)
    for i in range(len(all_keys)):
        if len(all_keys[i]) == 2:
            B_primitive[i] = B_primitive[i]  # 1.889725989
            logger.info(i, all_keys[i], B_primitive[i])
        if len(all_keys[i]) == 3:
            B_primitive[i] = B_primitive[i]  # /1.889725989
            # f = (1./base.bond_value(*all_keys[i][0:2]))
            # B_primitive[i][all_keys[i][0]*3:all_keys[i][0]*3+3] = (1)*B_primitive[i][all_keys[i][0]*3:all_keys[i][0]*3+3]
            # f = (1./((base.bond_value(*all_keys[i][0:2])*base.bond_value(*all_keys[i][1:3]))**0.5)  )
            # B_primitive[i][all_keys[i][1]*3:all_keys[i][1]*3+3] = (1)*B_primitive[i][all_keys[i][1]*3:all_keys[i][1]*3+3]
            # f = (1./base.bond_value(*all_keys[i][1:3]))
            # B_primitive[i][all_keys[i][2]*3:all_keys[i][2]*3+3] = (1)*B_primitive[i][all_keys[i][2]*3:all_keys[i][2]*3+3]
            logger.info(i, all_keys[i], B_primitive[i])
        if len(all_keys[i]) == 4:
            B_primitive[i] = B_primitive[i]  # /1.889725989
            # B_primitive[i] =  (1./((base.bond_value(*all_keys[i][0:2])*base.bond_value(*all_keys[i][1:3])*base.bond_value(*all_keys[i][2:4]))**0.5))*B_primitive[i]
            logger.info(i, all_keys[i], B_primitive[i])
    logger.info('B_primitive in Bohr', B_primitive.shape)
    logger.info(B_primitive)

    q_primitive = numpy.array(
        base.q_primitive(prim_base_keys, prim_base_values))
    logger.info('q_primitive', q_primitive.shape)
    logger.info(q_primitive)
    # Size and make q_target_primitive
    # Make K_base
    # G = B_primitive.B_primitive^T
    # Eigenvectors/values of G
    # K_base = eigenvectors with non-zero eigenvalues
    K_base, eigenvalues = non_zero_eigenvalues_matrix(B_primitive, 1e-9,
                                                      report=True)
    logger.info('K_base', K_base.shape)
    logger.info(K_base)
    logger.info('Primitive weighting in Delocalized Coorindates')
    for i in range(len(all_keys)):
        logger.info(i, all_keys[i], numpy.linalg.norm(K_base[:, i]) ** 2)
    q_nr = numpy.dot(K_base, q_primitive)
    logger.info('q_nr', q_nr.shape)
    logger.info(q_nr)

    # Make B_nr = K.B_primitive
    B_nr = numpy.dot(K_base, B_primitive)
    logger.info('B_nr', B_nr.shape)
    logger.info(B_nr)

    logger.info('Removing fake symmetry')
    # K_base = numpy.delete(K_base, [5,6,11],0)
    # LK = localize_K_matrix(K_base, reversed(range(len(all_keys))), all_keys)
    # LK = localize_K_matrix(K_base, reversed([i for i,k in enumerate(all_keys)
    #                                           if all_keys[0][0] in k or all_keys[0][1] in k]), all_keys)
    # LK = localize_K_matrix(K_base, reversed([i for i,k in enumerate(all_keys)
    #                                           if len(k) == 2]), all_keys)
    LK = K_base
    for row in LK:
        logger.info(numpy.round(row, 3))

    # Make B_nr = K.B_primitive
    B_nr = numpy.dot(LK, B_primitive)
    logger.info('B_nr', B_nr.shape)
    logger.info(B_nr)

    dq_prim = numpy.zeros(len(all_keys))
    dq_prim[77] = 0.5
    logger.info('dX_prim',
                numpy.dot(generalised_inverse(B_primitive), dq_prim))
    logger.info('dX_nr',
                numpy.dot(generalised_inverse(B_nr), numpy.dot(LK, dq_prim)))
    base.cart_dof_matrix = [[0 for _ in range(base.num_atoms() * 3)] for _ in
                            range(base.num_atoms() * 3 - 6)]
    base.cart_dof_matrix[0] = numpy.dot(generalised_inverse(B_nr),
                                        numpy.dot(LK, dq_prim)).tolist()
    # base.cart_dof_matrix[0] = numpy.dot(generalised_inverse(B_primitive), dq_prim).tolist()
    logger.info(numpy.array(base.cart_dof_matrix[0]))
    logger.info(numpy.dot(generalised_inverse(LK),
                          numpy.dot(B_nr, base.cart_dof_matrix[0])))
    base.xyz_string_form_print()
    base.apply_mode_addition_in_redundants([1.0], dQnormThr=1e-6)
    base.xyz_string_form_print()
    # print base.connectivity_matrix()
    # print base.connected_keys(1)


def internals_diff(base, target, ndegrees=0.5):
    logger.debug('Starting fmin_displacements')

    # obtain all the primitive keys
    prim_base_keys = base.primitive_keys(linear_thr=175. * math.pi / 180.)
    logger.info(prim_base_keys)
    logger.info(('Length of base primitive bond keys ' + str(
        len(prim_base_keys['o_bond_keys']))))
    logger.info(('Length of base primitive angle keys ' + str(
        len(prim_base_keys['o_angle_keys']))))
    logger.info(('Length of base primitive dihedral keys ' + str(
        len(prim_base_keys['o_dihedral_keys']))))
    prim_base_values = base.primitive_values(prim_base_keys)
    logger.info(('Length of base primitive bond values ' + str(
        len(prim_base_values['o_bond_keys']))))
    logger.info(('Length of base primitive angle values ' + str(
        len(prim_base_values['o_angle_keys']))))
    logger.info(('Length of base primitive dihedral values ' + str(
        len(prim_base_values['o_dihedral_keys']))))
    # target values come from base keys
    prim_target_keys = target.primitive_keys(linear_thr=175. * math.pi / 180.)
    prim_target_values = target.primitive_values(prim_base_keys)

    from scipy.sparse.csgraph import csgraph_from_dense
    from scipy.sparse.csgraph import shortest_path

    conn = base.connectivity_matrix()

    Gmask = numpy.ma.masked_invalid(numpy.array(conn))
    G = csgraph_from_dense(Gmask, null_value=0)
    dist_matrix, predecessors = shortest_path(G, return_predecessors=True)

    identical_env = []

    for i in range(base.num_atoms() - 1):
        irow = dist_matrix[i].copy()
        irow = numpy.delete(irow, i)
        for j in range(i + 1, base.num_atoms()):
            jrow = dist_matrix[j].copy()
            jrow = numpy.delete(jrow, j)
            if numpy.array_equal(irow, jrow):
                logger.info("Same environment for", i, j)
                identical_env.append([i, j])
    import itertools
    for i, j in identical_env:
        q = list(
            itertools.chain(*base.primitive_values(prim_base_keys).values()))
        before = numpy.linalg.norm(numpy.array(q))
        temp = base.atoms[i].xyz
        base.atoms[i].xyz = base.atoms[j].xyz
        base.atoms[j].xyz = base.atoms[i].xyz
        q = list(
            itertools.chain(*base.primitive_values(prim_base_keys).values()))
        after = numpy.linalg.norm(numpy.array(q))
        logger.info(i, j, before, after)
        if before < after:
            temp = base.atoms[i].xyz
            base.atoms[i].xyz = base.atoms[j].xyz
            base.atoms[j].xyz = base.atoms[i].xyz

    logger.info(prim_target_keys)

    for typ in prim_base_keys.keys():
        keys = prim_base_keys[typ]
        b_values = prim_base_values[typ]
        t_values = prim_target_values[typ]

        for i, k in enumerate(keys):
            diff = b_values[i] - t_values[i]
            if len(k) == 2 and diff > 0.005:
                logger.info(k, diff, 'Angstrom')
            elif abs(pirange(diff)) > 0.0175 * ndegrees:
                logger.info(k, pirange(diff), 'Radians ',
                            pirange(diff) / 0.0175, 'Deg')
