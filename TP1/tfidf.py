import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse.linalg import norm as sparse_norm
from scipy.sparse import csr_matrix

def tfidf(data, sublinear_tf = False, norm = 'l2', use_idf = True, smooth_idf = True, **kwargs):

    # Contamos las repeticiones
    tf = CountVectorizer(**kwargs).fit_transform(data)

    # Si estamos en sublinear tf, aplicamos logaritmo.
    if sublinear_tf: tf[tf != 0] = 1 + np.log(tf[tf != 0])

    # Dividimos por la suma en cada documento para sacar la frecuencia tf.
    tf = csr_matrix(tf / tf.sum(axis = 1))

    # D: Cantidad total de documentos; d: Cantidad de documentos en los que aparece cada palabra.
    D = tf.shape[0] + int(smooth_idf)
    d = tf.getnnz(axis = 0) + int(smooth_idf)

    # Aplicamos fórmula para idf.
    idf = 1 + np.log(D / d)

    # Multiplicamos tf * idf cuando corresponda.
    res = (tf.multiply(idf) if use_idf else tf)

    # Normalizamos a la norma que corresponda.
    return csr_matrix(res / sparse_norm(res, int(norm[1:]), axis = 1).reshape(-1, 1))

# Parámetros
sublinear_tf = False
norm = 'l2'
use_idf = True
smooth_idf = True

# Datos
string = 'bueno chau hola malo hola'
data = [string, string[6:], string[16:]]

_tfidf = TfidfVectorizer(sublinear_tf=sublinear_tf, use_idf=use_idf, smooth_idf=smooth_idf, norm = norm)

# Orishinal
print('Sklearn:')
print(_tfidf.fit_transform(data).toarray())

# Implementación
print('\nImplementación:')
print(tfidf(data, sublinear_tf=sublinear_tf, use_idf=use_idf, smooth_idf=smooth_idf, norm = norm).toarray())

