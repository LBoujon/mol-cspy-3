
from .inputs import *
from .pgamma import *
from functools import reduce

def pi_range(x):
    """
    :param x: angular value in radians
    :return: angular value in radians in (-pi, pi] range
    """
    x = float(x)
    if mt.sin(x) > 0.0:
        return mt.acos(mt.cos(x))
    else:
        return -mt.acos(mt.cos(x))


def degree_to_radian(data):
    """
    :param data: numpy array
    :return: numpy array sample convertedfrom degrees to radians
    """
    vecfunc = np.vectorize(pi_range)
    return vecfunc(np.radians(data))


def radian_to_degree(data):
    """
    :param data: numpy array
    :return: numpy array sample convertedfrom radians to degrees
    """
    return np.degrees(data)


def radian_to_linear(data):
    """
    :param data: numpy array
    :return: sample conversion to vector description via sine and cosine (data in radians)
    """
    return np.array([[np.sin(x), np.cos(x)] for x in data])


def p_trig_a(data, p):
    """
    :param data: numpy array (1 D)
    :param p: natural number (>0)
    :return: 'a' coefficient in the p-th trigonometric moment a+ib
        about the zero direction
    """
    return np.sum(np.cos(p * data)) / data.shape[0]


def p_trig_b(data, p):
    """
    :param data: numpy array (1 D)
    :param p: natural number (>0)
    :return: 'b' coefficient in the p-th trigonometric moment a+ib
        about the zero direction
    """
    return np.sum(np.sin(p * data)) / data.shape[0]


def mean_resultant_length_p(data, p):
    """a helper function"""
    return mt.sqrt(p_trig_a(data, p) * p_trig_a(data, p) + p_trig_b(data, p) * p_trig_b(data, p))


def mean_direction_p(data, p):
    """ a helper function """
    return mt.atan2(p_trig_b(data, p), p_trig_a(data, p))


def mean_resultant_length(data):
    """
    :param data: 1D numpy array
    :return: mean resultant length
    """
    return mean_resultant_length_p(data, 1)


def mean_direction(data):
    """
    :param data: 1D numpy array
    :return: mean direction
    """
    return mean_direction_p(data, 1)


def p_trig_mean_a(data, p):
    """ 'a' coefficient in the p-th sample trigonometric moment a+ib
        about the mean direction """
    return np.sum(np.cos(p * (np.array(data) - mean_direction(data)))) / data.shape[0]


def p_trig_mean_b(data, p):
    """ 'b' coefficient in the p-th sample trigonometric moment a+ib
        about the mean direction """
    return np.sum(np.sin(p * (np.array(data) - mean_direction(data)))) / data.shape[0]


def median_direction(data):
    """
    :param data: 1 D numpy array
    :return: median direction
    """
    value = None
    min_val = 1000
    if data.shape[0] % 2 == 1:
        for el in data:
            d = np.sum(np.pi - np.abs(np.pi - np.abs(np.array(data) - el))) / data.shape[0]
            if d < min_val:
                min_val = d
                value = el
        return value
    else:
        list1 = np.copy(data)
        list1.sort()
        for i in range(list1.shape[0]):
            if i % 2 == 0:
                psi = (list1[i] + list1[i + 1]) / 2.0
                d = np.sum(np.pi - np.abs(np.pi - np.abs(list1 - psi))) / list1.shape[0]
                if d < min_val:
                    min_val = d
                    value = psi
        psi = (list1[0] + list1[-1]) / 2.0
        d = np.sum(np.pi - np.abs(np.pi - np.abs(list1 - psi))) / list1.shape[0]
        if d < min_val:
            min_val = d
            value = psi
        return value


def circular_variance(data):
    """
    :param data: 1D numpy array
    :return: circular variance
    """
    return 1 - mean_resultant_length(data)


def circ_stddev(data):
    """
    :param data: 1D numpy array
    :return: circular standard deviation
    """
    return np.sqrt((-2 * np.log(mean_resultant_length(data))))


def circular_dispersion(data):
    """
    :param data: 1D numpy array
    :return: circular dispersion
    """
    return (1 - mean_resultant_length_p(data, 2)) / (2 * mean_resultant_length(data) * mean_resultant_length(data))


def circular_range(data):
    """
    :param data: 1D numpy array
    :return: circular range
    """
    vecfunc = np.vectorize(pi_range)
    print(data)
    list1 = vecfunc(data)
    list1.sort()
    print(list1)
    return list1[-1] - list1[0]


def circular_correlation(list1, list2):
    """
    :param list1: 1D numpy array
    :param list2: 1D numpy array
    :return: circular correlation
    """
    t1 = mean_direction(list1)
    t2 = mean_direction(list2)
    return np.sum(np.cos(np.array(list1) - np.array(list2) + t2-t1) - np.cos(list1+list2-t1-t2)) / (2*mt.sqrt(
        np.sum(np.sin(np.array(list1) - t1) ** 2) * np.sum(np.sin(np.array(list2) - t2) ** 2)))


def skewness(data):
    """
    :param data: 1D numpy array
    :return: circular skewness
    """
    return p_trig_mean_b(data, 2)


def kurtosis(data):
    """
    :param data: 1D numpy array
    :return: circular kurtosis
    """
    return p_trig_mean_a(data, 2)


def skew_mardia(data):
    """
    :param data: 1D numpy array
    :return: circular skewness as as proposed by Mardia
    """
    return skewness(data) / np.power((np.sqrt(1 - mean_resultant_length(data))), 3)


def kurtosis_mardia(data):
    """
    :param data: 1D numpy array
    :return: circular kurtosis as proposed by Mardia
    """
    return (kurtosis(data) - np.power(mean_resultant_length(data), 4)) / np.power(circular_variance(data), 2)


def r_uniform(data):
    """
    :param data: 1D numpy array
    :return: Raileigh test p-value for uniformity of circular data.
        Important assumptions of the test:
        Data is not diametrically bidirectional
        Distribution is unimodal.
        The Null hypothesis: distribution is uniform.
    """
    n = len(data)
    z = n * pow(mean_resultant_length(data), 2)
    if n < 50:
        p = np.exp(-z) * (1+(2 * z - z * z) / (4 * n) - (24 * z - 132 * z * z + 76 * pow(z, 3) - 9 * pow(z, 4)) / (288*n*n))
    else:
        p = np.exp(-z)
    if p < 0.05:
        print("The p-value is %f, we can reject uniformity")
    else:
        print("The p-value is %f, we cant reject uniformity" % p)
    return p


def r_uniform_mean(data, mu):
    """
    :param data: 1D numpy array
    :param mu: mean direction in radians
    :return: Raileigh test for uniformity of circular data
        with mean direction specified. Important assumptions of
        the test: Data is not diametrically bidirectional
        Distribution is unimodal
        The Null hypothesis: distribution is uniform.
    """
    n = len(data)
    r0 = sum(np.cos(data - mu)) / n
    z0 = r0 * mt.sqrt(2 * n)
    pz = 0.5 + 0.5 * mt.erf(z0 / np.sqrt(2))
    p = 1 - pz + mt.exp(-z0 * z0 / 2) * ((3 * z0 - pow(z0, 3)) / (16 * n) + (15 * z0 + 305 * pow(z0, 3) - 125 * pow(z0, 5) + 9 * pow(z0, 7)) / (4608 * n * n)) / np.sqrt(2 * np.pi)
    if p < 0.05:
        print("The p-value is %f, we can reject uniformity")
    else:
        print("The p-value is %f, we cant reject uniformity" % p)
    return p

def watson_common_direction_test(*args):
    """ Watson large sample test for common direction (samples > 25 in size)
        no assumptions on common dispersion or shape. The Null hypothesis: no difference between mean directions.
    :param args: samples to test
    :return: p value for the test
    """
    disp = np.array([circular_dispersion(np.array(el)) for el in args])
    print(disp)
    proc = np.max(disp)/np.min(disp)
    average = np.array([mean_direction(np.array(el)) for el in args])
    if proc <= 4:
        cp = sum([len(args[i])*np.cos(average[i]) for i in range(len(args))])
        sp = sum([len(args[i])*np.sin(average[i]) for i in range(len(args))])
        n = sum([len(args[i]) for i in range(len(args))])
        delta0 = sum([len(args[i])*disp[i] for i in range(len(args))])/n
        rp = np.sqrt(cp*cp+sp*sp)
        yg = 2*(n-rp)/delta0
    else:
        cm = sum([len(args[i])*np.cos(average[i])/disp[i] for i in range(len(args))])
        sm = sum([len(args[i])*np.sin(average[i])/disp[i] for i in range(len(args))])
        rm = np.sqrt(cm*cm+sm*sm)
        yg = 2*(sum([len(args[i])/disp[i] for i in range(len(args))])-rm)

    p_val = pchisq(yg, len(args)-1, lower_tail=False, log_p=0)
    if p_val < 0.05:
        print("""The p-values is %f, indicating difference between the mean directions""" % p_val)
    else:
        print(""""The p-value is %f, indicating no significant difference
        between the mean directions.""" % p_val)
    return p_val


def common_median_test(*args):
    """
    :param samples: samples (all in a rangefrom (-pi,pi]
    :return: p-value for common median direction
    """
    samples = [[pi_range(el) for el in args[i]] for i in range(len(args))]
    median = median_direction(reduce(lambda u, y: np.hstack((u, y)), np.array(samples)))
    x = [[el for el in np.array(samples[i])-median] for i in range(len(samples))]
    number_of_negatives = [len([u for u in x[i] if u < 0.0]) for i in range(len(samples))]
    m = sum(number_of_negatives)
    n = sum([len(samples[i]) for i in range(len(samples))])
    pg = (n*n/(m*(n-m)))*(sum([number_of_negatives[i]*number_of_negatives[i]/len(samples[i]) for i in range(len(samples))]))-n*m/(n-m)
    p_val = pchisq(pg, len(samples)-1, lower_tail=False, log_p=0)
    if p_val < 0.05:
        print("""The p-values is %f, indicating difference between the median directions""" % p_val)
    else:
        print(""""The p-value is %f, indicating no significant difference
        between the median directions.""" % p_val)
    return p_val


def rank(data):
    """
    :param data: a list of numbers
    :return: ranking with ties broken randomly
    """

    data = np.array(data)
    ranking_ind = data.argsort()
    ranking = np.zeros(len(data))
    for i in range(len(data)):
        ranking[ranking_ind[i]] = i+1
    return ranking


def large_mww_test(*args):
    """
    :param args: samples
    :return: p-value for large sample Mardia-Watson-Wheeler test
    """
    samples = [[pi_range(el) for el in args[i]] for i in range(len(args))]
    all_together = reduce(lambda x, y: np.hstack((x, y)), np.array(samples))
    ranking = rank(all_together)
    ranks = [ranking[len(args[i-1]):len(args[i])] for i in range(1, len(args))]
    c, s, wg = [], [], 0
    n = sum([len(samples[i]) for i in range(len(samples))])
    for k in range(0, len(args)):
        c[k] = np.sum(np.cos(2*np.pi*np.array(ranks[k])/n))
        s[k] = np.sum(np.sin(2*np.pi*np.array(ranks[k])/n))
        wg += (c[k]*c[k] + s[k]*s[k])/len(samples[k])
    wg *= 2
    p_val = pchisq(wg, 2*(len(samples)-1), lower_tail=False, log_p=0)
    if p_val < 0.05:
        print("""The p-values is %f, indicating difference between the distributionsfrom which the samples are drawn""" % p_val)
    else:
        print(""""The p-value is %f, indicating no significant difference
        between the distributionsfrom which the samples are drawn.""" % p_val)
    return p_val


def paired_moore_test(list1, list2):
    """
    :param list1: first sample
    :param list1: second sample
    :return: p-value for Moore's test of paired data
    """
    assert len(list1) == len(list2), "sizes of two lists of numbers are not equal"
    x = np.cos(np.array(list1))-np.cos(np.array(list2))
    y = np.sin(np.array(list1))-np.sin(np.array(list2))
    r = list(map(lambda u, v: mt.sqrt(u*u+v*v), x, y))
    cosphi = np.array([x[i]/r[i] for i in range(x.shape[0])])
    sinphi = np.array([y[i]/r[i] for i in range(x.shape[0])])
    rc = np.inner(r, cosphi)/len(list1)
    rs = np.inner(r, sinphi)/len(list1)
    rval = mt.sqrt((rs*rs+rc*rc)/len(list1))
    return rval


def application(funct):
    def applied(a, col):
        if a.units == "degrees":
            return funct(degree_to_radian(a.dataset[:, col-1]))*180/np.pi
        else:
            print(a.dataset[:, col-1])
            return funct(a.dataset[:, col-1])
    return applied

mean = application(mean_direction)
median = application(median_direction)
variance = application(circular_variance)
stddev = application(circ_stddev)
dispersion = application(circular_dispersion)
mean_rlen = application(mean_resultant_length)
crange = application(circular_range)

def correlation(a, col1, b, col2):
    """

    :param a: an Angle object
    :param col1: column number for the 'a' object
    :param b: an Angle object
    :param col2: column number for the 'b' object
    :return: circular correlation
    """
    if a.units == "degrees":
        x = degree_to_radian(a.dataset[:, col1-1])*180/np.pi
    else:
        x = a.dataset[:, col1-1]
    if b.units == "degrees":
        y = degree_to_radian(b.dataset[:, col2-1])*180/np.pi
    else:
        y = b.dataset[:, col2-1]
    return circular_correlation(x, y)


def help(name):
    """
    :param name: command name
    :return: what it is expected to be done
    """
    if name == "mean":
        print("""Calculates mean direction. First argument is an Angles object,
        second argument is a column numberfrom dataset containing in an Angle
        object""")
    if name == "median":
        print("""Calculates median direction. First argument is an Angles object,
        second argument is a column numberfrom dataset containing in an Angle
        object""")
    if name == "variance":
        print("""Calculates circular variance (one minus mean resultant length). First argument is an Angles object,
        second argument is a column numberfrom dataset containing in an Angle
        object. A good choice to describe spread of unimodal sample. Takes valuesfrom [0, 1].""")
    if name == "stddev":
        print("""Calculates circular standard deviation. First argument is an Angles object,
        second argument is a column numberfrom dataset containing in an Angle
        object""")
    if name == "dispersion":
        print("""Calculates circular dispersion. First argument is an Angles object,
        second argument is a column numberfrom dataset containing in an Angle
        object""")
    if name == "mean_rlen":
        print("""Calculates mean resultant length. First argument is an Angles object,
        second argument is a column numberfrom dataset containing in an Angle
        object. A good choice to describe concentration of unimodal sample. Takes values in interval [0, 1].
        It is equal 1 in case all the data are at the same point.""")
    if name == "crange":
        print("""Calculates circular range. First argument is an Angles object,
        second argument is a column numberfrom dataset containing in an Angle
        object. A good choice to describe spread of multimodal sample.""")
    if name == "large_mww_test":
        print(""" Large-sample Mardia-Watson-Wheeler could be applied to several independent samples
                  to establish whether these samples are drawnfrom a common distribution.
                  It requires sample length to be greater than 10 elements.
                  The null hypothesis is: The distribution is common.
              """)
    if name == "r_uniform":
        print(""" Raileigh test p-value for uniformity of circular data.
        Important assumptions of the test:
        Data is not diametrically bidirectional
        Distribution is unimodal.
        The Null hypothesis: distribution is uniform.""")

    if name == "r_uniform_mean":
        print("""Raileigh test for uniformity of circular data
        with mean direction specified. Important assumptions of
        the test: Data is not diametrically bidirectional
        Distribution is unimodal
        The Null hypothesis: distribution is uniform.""")

    if name == "watson_common_direction_test":
        print(""" Watson large sample test for common direction (samples > 25 in size)
        no assumptions on common dispersion or shape. The Null hypothesis: no difference between mean directions""")

    if name == "common_median_test":
        print("""Test for common direction.
              The Null hypothesis: no difference between mean directions.""")

    if name == "paired_moore_test":
        print(""" If two samples of are paired, we can test if two
                samples are drawnfrom the same distribution.
                The Null hypothesis: samples are drawnfrom the same distribution""")
