import math as mt
import numpy as np

DBL_EPSILON = 2.2204460492503131e-16
DBL_MIN = 2.2250738585072014e-308
M_LN2 = 0.69314718055994530942 
scalefactor = mt.pow(mt.pow(mt.pow(4294967296.0, 2), 2), 2)
M_cutoff = 3.196577e18
M_LN_SQRT_2PI = 0.918938533204672741780329736406

def R_Log1_Exp(x):
    if x > -M_LN2:
        return mt.log(-mt.expm1(x))
    else:
        return mt.log1p(-mt.exp(x))

def logcf(x, i, d, eps):
    c1 = 2 * d
    c2 = i + d
    c4 = c2 + d
    a1 = c2
    b1 = i * (c2 - i * x)
    b2 = d * d * x
    a2 = c4 * c2 - b2
    assert (i > 0)
    assert (d >= 0)
    b2 = c4 * b1 - i * b2

    while (mt.fabs(a2 * b1 - a1 * b2) > mt.fabs(eps * b1 * b2)):
        c3 = c2*c2*x
        c2 += d
        c4 += d
        a1 = c4 * a2 - c3 * a1
        b1 = c4 * b2 - c3 * b1

        c3 = c1 * c1 * x
        c1 += d
        c4 += d
        a2 = c4 * a1 - c3 * a2
        b2 = c4 * b1 - c3 * b2

        if (mt.fabs(b2) > scalefactor):
            a1 /= scalefactor
            b1 /= scalefactor
            a2 /= scalefactor
            b2 /= scalefactor
        elif (mt.fabs(b2) < 1./scalefactor):
            a1 *= scalefactor
            b1 *= scalefactor
            a2 *= scalefactor
            b2 *= scalefactor
    return a2 / b2

def log1pmx(x):

    minLog1Value = -0.79149064

    if (x > 1 or x < minLog1Value):
        return mt.log1p(x) - x
    else:
        r = x / (2 + x)
        y = r * r
        if (mt.fabs(x) < 1e-2):
            two = 2
            return r * ((((two / 9 * y + two / 7) * y + two / 5) * y + two / 3) * y - x)
        else:
            tol_logcf = 1e-14
            return r * (2 * y * logcf(y, 3, 2, tol_logcf) - x)


def lgamma1p(a):
    eulers_const = 0.5772156649015328606065120900824024

    N = 40
    coeffs = [0.3224670334241132182362075833230126e-0,
    0.6735230105319809513324605383715000e-1,
    0.2058080842778454787900092413529198e-1,
    0.7385551028673985266273097291406834e-2,
    0.2890510330741523285752988298486755e-2,
    0.1192753911703260977113935692828109e-2,
    0.5096695247430424223356548135815582e-3,
    0.2231547584535793797614188036013401e-3,
    0.9945751278180853371459589003190170e-4,
    0.4492623673813314170020750240635786e-4,
    0.2050721277567069155316650397830591e-4,
    0.9439488275268395903987425104415055e-5,
    0.4374866789907487804181793223952411e-5,
    0.2039215753801366236781900709670839e-5,
    0.9551412130407419832857179772951265e-6,
    0.4492469198764566043294290331193655e-6,
    0.2120718480555466586923135901077628e-6,
    0.1004322482396809960872083050053344e-6,
    0.4769810169363980565760193417246730e-7,
    0.2271109460894316491031998116062124e-7,
    0.1083865921489695409107491757968159e-7,
    0.5183475041970046655121248647057669e-8,
    0.2483674543802478317185008663991718e-8,
    0.1192140140586091207442548202774640e-8,
    0.5731367241678862013330194857961011e-9,
    0.2759522885124233145178149692816341e-9,
    0.1330476437424448948149715720858008e-9,
    0.6422964563838100022082448087644648e-10,
    0.3104424774732227276239215783404066e-10,
    0.1502138408075414217093301048780668e-10,
    0.7275974480239079662504549924814047e-11,
    0.3527742476575915083615072228655483e-11,
    0.1711991790559617908601084114443031e-11,
    0.8315385841420284819798357793954418e-12,
    0.4042200525289440065536008957032895e-12,
    0.1966475631096616490411045679010286e-12,
    0.9573630387838555763782200936508615e-13,
    0.4664076026428374224576492565974577e-13,
    0.2273736960065972320633279596737272e-13,
    0.1109139947083452201658320007192334e-13]
    
    c = 0.2273736845824652515226821577978691e-12
    tol_logcf = 1e-14

    if (mt.fabs(a) >= 0.5):
        return  mt.lgamma(a + 1)
    lgam = c * logcf(-a / 2, N + 2, 1, tol_logcf)
    i = N-1
    while (i >= 0):
        lgam = coeffs[i] - a * lgam
        i-=1
    return (a * lgam - eulers_const) * a - log1pmx (a)

def pgamma_smallx (x, alph, lower_tail, log_p):

    sum = 0
    c = alph
    n = 0

    while True:
        n+=1
        c *= -x / n
        term = c / (alph + n)
        sum += term
        if not (mt.fabs(term) > DBL_EPSILON * mt.fabs(sum)):
            break
        
    if lower_tail:
        if log_p:
            f1 = mt.log1p(sum)
        else:
            f1 = 1 + sum

        if alph > 1:
            f2 = dpois_raw(alph, x, log_p)
            if log_p:
                f2 += x
            else:
                f2 *= mt.exp(x)
        elif log_p:
            f2 = alph * mt.log(x) - lgamma1p (alph)
        else:
            f2 = mt.pow(x, alph) / mt.exp(lgamma1p (alph))
        if log_p:
            return f1 + f2
        else:
            return f1 * f2
    else:
        lf2 = alph * mt.log(x) - lgamma1p (alph)
        if (log_p):
            return R_Log1_Exp(mt.log1p(sum) + lf2)
        else:
            f1m1 = sum
            f2m1 = mt.expm1(lf2)
            return -(f1m1 + f2m1 + f1m1 * f2m1)

def pd_upper_series(x, y, log_p):

    term = x / y
    sum = term

    while True:
        y+=1
        term *= x / y
        sum += term
        if not (term > sum * DBL_EPSILON):
            break
    if log_p:
        return mt.log (sum)
    else:
        return sum

def R_D_exp(x, log_p):
    if log_p:
        return x
    else:
        return mt.exp(x)

def R_D_fexp(f, x, log_p):
    if log_p:
        return -0.5*mt.log(f)+x
    else:
        return mt.exp(x)/mt.sqrt(f)

def stirlerr(n):

    S0=0.083333333333333333333
    S1=0.00277777777777777777778    
    S2=0.00079365079365079365079365 
    S3=0.000595238095238095238095238 
    S4=0.0008417508417508417508417508

    sferr_halves = [0.1534264097200273452913848, 
    0.0810614667953272582196702, 
    0.0548141210519176538961390,  
    0.0413406959554092940938221, 
    0.03316287351993628748511048, 
    0.02767792568499833914878929,
    0.02374616365629749597132920, 
    0.02079067210376509311152277,
    0.01848845053267318523077934,
    0.01664469118982119216319487,
    0.01513497322191737887351255, 
    0.01387612882307074799874573, 
    0.01281046524292022692424986,
    0.01189670994589177009505572, 
    0.01110455975820691732662991, 
    0.010411265261972096497478567, 
    0.009799416126158803298389475, 
    0.009255462182712732917728637, 
    0.008768700134139385462952823,
    0.008330563433362871256469318, 
    0.007934114564314020547248100, 
    0.007573675487951840794972024, 
    0.007244554301320383179543912, 
    0.006942840107209529865664152, 
    0.006665247032707682442354394,
    0.006408994188004207068439631, 
    0.006171712263039457647532867, 
    0.005951370112758847735624416, 
    0.005746216513010115682023589, 
    0.005554733551962801371038690]

    if (n <= 15.0):
        nn = n + n
        if nn == int(nn):
            return sferr_halves[int(nn)]
        return mt.lgamma(n + 1.) - (n + 0.5)*mt.log(n) + n - M_LN_SQRT_2PI

    nn = n*n
    if n>500:
        return (S0-S1/nn)/n
    if n> 80:
        return (S0-(S1-S2/nn)/nn)/n
    if n> 35:
        return (S0-(S1-(S2-S3/nn)/nn)/nn)/n
    return (S0-(S1-(S2-(S3-S4/nn)/nn)/nn)/nn)/n


def bd0(x, np):

    if (mt.isinf(x) or mt.isinf(np) or np == 0.0):
        return np.NAN

    if (mt.fabs(x-np) < 0.1*(x+np)):
        v = (x-np)/(x+np)
        s = (x-np)*v
        if(mt.fabs(s) < DBL_MIN):
           return s
        ej = 2*x*v
        v = v*v
        for j in range (1, 1000):
            ej *= v
            s1 = s+ej/((j<<1)+1)
            if (s1 == s):
                return s1
            s = s1
    return (x*mt.log(x/np)+np-x)

def dpois_raw(x, lambda1, log_p):
    if lambda1 == 0:
        if x == 0:
            if log_p:
                return 0.0
            else:
                return 1.0
        else:
            if log_p:
                return float('-inf')
            else:
                return 0.0
    if mt.isinf(lambda1):
        if log_p:
            return float('-inf')
        else:
            return 0.0
    if x < 0:
        if log_p:
            return float('-inf')
        else:
            return 0.0
    if x <= lambda1 * DBL_MIN:
        return R_D_exp(-lambda1, log_p)
    if lambda1 < x * DBL_MIN:
        return R_D_exp(-lambda1 + x*mt.log(lambda1)-mt.lgamma(x+1), log_p)
    return R_D_fexp(2*np.pi*x, -stirlerr(x)-bd0(x, lambda1), log_p)


def dpois_wrap(x_plus_1, lambda1, log_p):

    if mt.isinf(lambda1):
        if log_p:
            return float('-inf')
        else:
            return 0.0
    if x_plus_1 > 1:
        return dpois_raw(x_plus_1 - 1, lambda1, log_p)
    if lambda1 > mt.fabs(x_plus_1 - 1) * M_cutoff:
        return R_D_exp(-lambda1 - mt.lgamma(x_plus_1), log_p)
    else:
        d = dpois_raw(x_plus_1, lambda1, log_p)
        if log_p:
            return d + mt.log(x_plus_1 / lambda1)
        else:
            return d * (x_plus_1 / lambda1)

def pd_lower_cf (y, d):

    f = 0.0

    max_it = 200000

    if (y == 0):
        return 0

    f0 = y/d
    
    if(mt.fabs(y - 1) < mt.fabs(d) * DBL_EPSILON):
        return (f0)

    if(f0 > 1.):
        f0 = 1.
    c2 = y
    c4 = d

    a1 = 0
    b1 = 1
    a2 = y
    b2 = d

    while (b2 > scalefactor):
        a1 /= scalefactor
        b1 /= scalefactor
        a2 /= scalefactor
        b2 /= scalefactor

    i = 0
    of = -1.
    while (i < max_it):
        i+=1
        c2-=1
        c3 = i * c2
        c4 += 2
        a1 = c4 * a2 + c3 * a1
        b1 = c4 * b2 + c3 * b1
        i+=1
        c2-=1
        c3 = i * c2
        c4 += 2
        a2 = c4 * a1 + c3 * a2
        b2 = c4 * b1 + c3 * b2

        if (b2 > scalefactor):
            a1 /= scalefactor
            b1 /= scalefactor
            a2 /= scalefactor
            b2 /= scalefactor

        if (b2 != 0):
            f = a2 / b2
            if (mt.fabs (f - of) <= DBL_EPSILON * max(f0, mt.fabs(f))):
                return f
            of = f
    return f

def pd_lower_series (lambda1, y):

    term = 1.0
    sum = 0.0

    while (y >= 1 and term > sum * DBL_EPSILON):
        term *= y / lambda1
        sum += term
        y-=1

    if (y != mt.floor(y)):
        f = pd_lower_cf (y, lambda1 + 1 - y)
        sum += term * f

    return sum

def pnorm_both(x, cum, i_tail, log_p):

    a = [2.2352520354606839287,
    161.02823106855587881,
    1067.6894854603709582,
    18154.981253343561249,
    0.065682337918207449113]
    
    b =[47.20258190468824187,
    976.09855173777669322,
    10260.932208618978205,
    45507.789335026729956]

    c = [0.39894151208813466764,
    8.8831497943883759412,
    93.506656132177855979,
    597.27027639480026226,
    2494.5375852903726711,
    6848.1904505362823326,
    11602.651437647350124,
    9842.7148383839780218,
    1.0765576773720192317e-8]

    d = [22.266688044328115691,
    235.38790178262499861,
    1519.377599407554805,
    6485.558298266760755,
    18615.571640885098091,
    34900.952721145977266,
    38912.003286093271411,
    19685.429676859990727]
    
    p = [0.21589853405795699,
    0.1274011611602473639,
    0.022235277870649807,
    0.001421619193227893466,
    2.9112874951168792e-5,
    0.02307344176494017303]
    
    q = [1.28426009614491121,
    0.468238212480865118,
    0.0659881378689285515,
    0.00378239633202758244,
    7.29751555083966205e-5]


    eps = DBL_EPSILON * 0.5
    ccum = 0
    lower = (i_tail != 1) #fix
    upper = (i_tail != 0)

    y = mt.fabs(x)
    if y <= 0.67448975:
        if y > eps:
            xsq = x * x
            xnum = a[4] * xsq
            xden = xsq
            for i in range(0,3):
                xnum = (xnum + a[i]) * xsq
                xden = (xden + b[i]) * xsq
    
        else:
            xnum = 0.0
            xden = 0.0

        temp = x * (xnum + a[3]) / (xden + b[3])
        if lower:
            cum = 0.5 + temp
        if upper:
            ccum = 0.5 - temp
        if log_p:
            if lower:
                cum = mt.log(cum)
            if upper:
                ccum = mt.log(ccum)
                
    elif y <= 5.656854249492380195206754896838:
        xnum = c[8] * y
        xden = y
        for i in range(0, 7):
            xnum = (xnum + c[i]) * y
            xden = (xden + d[i]) * y

        temp = (xnum + c[7]) / (xden + d[7])
        xsq = mt.trunc(y * 16) / 16
        deli = (y - xsq) * (y + xsq)
        if log_p:
            cum = (-xsq * xsq * 0.5) + (-deli * 0.5) + mt.log(temp)
            if (lower and x > 0.) or (upper and x <= 0.):
                ccum = mt.log1p(-mt.exp(-xsq * xsq * 0.5) *mt.exp(-deli * 0.5) * temp)
        else:
            cum = mt.exp(-xsq * xsq * 0.5) * mt.exp(-deli * 0.5) * temp
            ccum = 1.0 - cum
        if x > 0.:
            temp = cum
            if lower:
                cum = ccum
                ccum = temp
                   
    elif (log_p and y < 1e170) or (lower and -37.5193 < x  and  x < 8.2924) or (upper and -8.2924  < x  and  x < 37.5193):
        
        xsq = 1.0 / (x * x)
        xnum = p[5] * xsq
        xden = xsq
        for i in range(0,4):
            xnum = (xnum + p[i]) * xsq
            xden = (xden + q[i]) * xsq

        temp = xsq * (xnum + p[4]) / (xden + q[4])
        temp = (0.398942280401432677939946059934 - temp) / y

        # del(x)
        xsq = mt.trunc(x * 16) / 16
        deli = (x - xsq) * (x + xsq)
        if(log_p):
            cum = (-xsq * xsq * 0.5) + (-deli * 0.5) + mt.log(temp)
            if((lower and x > 0.) or (upper and x <= 0.)):
                ccum = mt.log1p(-mt.exp(-xsq * xsq * 0.5) *mt.exp(-deli * 0.5) * temp)
        else:
            cum = mt.exp(-xsq * xsq * 0.5) * mt.exp(-deli * 0.5) * temp
            ccum = 1.0 - cum
        #----
        
        if x > 0.:
            temp = cum
            if(lower):
                cum = ccum
                ccum = temp

    else: 
        if x > 0:
            if log_p:
                cum = 0.0
                ccum = float('-inf')
            else:
                cum = 1.0
                ccum = 0.0 
        else:
            if log_p:
                cum = float('-inf')
                ccum = 0.0
            else:
                cum = 0.0
                ccum = 1.0                 
    return cum, ccum

def pnorm(x, mu, sigma, lower_tail, log_p):

    if mt.isinf(x) and mu == x:
        return np.NAN
    if sigma <= 0:
        if sigma < 0:
            return np.NAN
        if x < mu:
            if log_p:
                return float('-inf')
            else:
                return 0.0
        else:
            if log_p:
                return 0.0
            else:
                return 1.0             
    p = (x - mu) / sigma
    if (mt.isinf(p)):
        if (x < mu):
            if log_p:
               return float('-inf')
            else:
                return 0.0
        else:
            if log_p:
                return 0.0
            else:
                return 1.0   
    x = p

    if lower_tail:
        p, cp = pnorm_both(x, p, 0, log_p)
        return p
    else:     
        p, cp = pnorm_both(x, p, 1, log_p)
        return cp


def dpnorm(x, lower_tail, lp):

    if (x < 0):
        x = -x
        lower_tail = not lower_tail
    
    if (x > 10 and not lower_tail):
        term = 1 / x
        sum = term
        x2 = x * x
        i = 1

        while True:
            term *= -i / x2
            sum += term
            i += 2
            if not (mt.fabs (term) > DBL_EPSILON * sum):
                break

        return 1 / sum
    else:
        d = dnorm(x, 0., 1., False)
        return d / mt.exp(lp)

def dnorm(x, mu, sigma, log_p):

    if(mt.isinf(sigma)):
        if log_p:
            return float('-inf')
        else:
            return 0.0
    if(mt.isinf(x) and mu == x):
        return np.NAN
    if (sigma <= 0):
        if (sigma < 0):
           return np.NAN

        if(x == mu):
            return 1.7976931348623157E+308
        else:
            if log_p:
                return float('-inf')
            else:
                return 0.0
    x = (x - mu) / sigma

    if(mt.isinf(x)):
        if log_p:
            return float('-inf')
        else:
            return 0.0

    x = mt.fabs (x)
    if (x >= 2 * mt.sqrt(1.7976931348623157E+308)):
        if log_p:
            return float('-inf')
        else:
            return 0.0
    if log_p:
        return -(M_LN_SQRT_2PI + 0.5 * x * x + mt.log(sigma))

    return 0.398942280401432677939946059934 * mt.exp(-0.5 * x * x) / sigma

def ppois_asymp (x, lambda1, lower_tail, log_p):

    coefs_a = [-1e99,
    2/3.,
    -4/135.,
    8/2835.,
    16/8505.,
    -8992/12629925.,
    -334144/492567075.,
    698752/1477701225.]

    coefs_b = [-1e99,
    1/12.,
    1/288.,
    -139/51840.,
    -571/2488320.,
    163879/209018880.,
    5246819/75246796800.,
    -534703531/902961561600.]

    dfm = lambda1 - x
    pt_ = - log1pmx (dfm / x) 
    s2pt = mt.sqrt (2 * x * pt_)
    if (dfm < 0):
        s2pt = -s2pt

    res12 = 0
    res1_ig = mt.sqrt(x)
    res1_term = mt.sqrt(x)
    res2_ig = s2pt
    res2_term = s2pt
    for i in range(1,8):
        res12 += res1_ig * coefs_a[i]
        res12 += res2_ig * coefs_b[i]
        res1_term *= pt_ / i
        res2_term *= 2 * pt_ / (2 * i + 1)
        res1_ig = res1_ig / x + res1_term
        res2_ig = res2_ig / x + res2_term

    elfb = x
    elfb_term = 1
    for i in range(1, 8):
        elfb += elfb_term * coefs_b[i]
        elfb_term /= x
        
    if not lower_tail:
        elfb = -elfb

    f = res12 / elfb
    np = pnorm (s2pt, 0.0, 1.0, not lower_tail, log_p) #********************

    if (log_p):
        n_d_over_p = dpnorm(s2pt, not lower_tail, np) #**************
        return np + mt.log1p(f * n_d_over_p)
    else:
        nd = dnorm(s2pt, 0., 1., log_p) #***********************
        return np + f * nd


def pgamma_raw(x, alph, lower_tail, log_p):

    if(x <= 0.0):
        if lower_tail:
            if log_p:
                return float('-inf')
            else:
                return 0.0
        else:
            if log_p:
                return 0.0
            else:
                return 1.0
                
    if(x >= 1.7976931348623157E+308):
        if lower_tail:
            if log_p:
                return 0.0
            else:
                return 1.0                 
        else:    
            if log_p:
                return float('-inf')
            else:
                return 0.0
    
    if (x < 1):
        res = pgamma_smallx (x, alph, lower_tail, log_p)
    elif (x <= alph - 1 and x < 0.8 * (alph + 50)):
        sum = pd_upper_series(x, alph, log_p)
        d = dpois_wrap(alph, x, log_p)
        if (not lower_tail):
            if log_p:
                res = R_Log1_Exp(d + sum)
            else:
                res = 1 - d*sum
        else:
            if log_p:
                res = sum + d
            else:
                res = sum * d
    elif (alph - 1 < x and alph < 0.8 * (x + 50)):
        d = dpois_wrap(alph, x, log_p)
        if alph < 1:
            if (x * DBL_EPSILON > 1 - alph):
                if log_p:
                    sum = 0.0
                else:
                    sum = 1.0
            else:
                f = pd_lower_cf(alph, x - (alph - 1)) * x / alph
                if log_p:
                    sum = mt.log(f)
                else:
                    sum = f
        else:
            sum = pd_lower_series(x, alph - 1)
            if log_p:
                sum = mt.log1p(sum)
            else:
                sum +=1

        if not lower_tail:
            if log_p:
                res = sum + d
            else:
                res = sum * d
        else:
            if log_p:
                res =  R_Log1_Exp (d + sum)
            else:
                res = 1 - d * sum
    else:
        res = ppois_asymp (alph - 1, x, not lower_tail, log_p)

    if (not log_p and res < DBL_MIN / DBL_EPSILON):
        return mt.exp(pgamma_raw (x, alph, lower_tail, 1))
    else:
        return res

def pgamma(x, alph, scale, lower_tail, log_p):
    if(alph < 0. or scale <= 0.):
        raise ValueError('A very specific bad thing happened')
    x /= scale
    if (alph == 0.):
        if (x<=0):
            if lower_tail:
                if log_p:
                    return float('-inf')
                else:
                    return 0.0
            else:
                if log_p:
                   return 0.0
                else:
                   return 1.0
        else:
            if lower_tail:
                if log_p:
                    return 0.0
                else:
                    return 1.0
            else:    
                if log_p:
                    return float('-inf')
                else:
                    return 0.0
    return pgamma_raw(x, alph, lower_tail, log_p)

def pchisq(x, df, lower_tail, log_p):
    return pgamma(x, df/2., 2., lower_tail, log_p)
