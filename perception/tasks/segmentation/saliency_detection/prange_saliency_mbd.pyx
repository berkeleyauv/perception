import sys
import numpy as np
cimport numpy as np
import scipy.spatial.distance
import skimage
import skimage.io
from skimage.util import img_as_float
from scipy.optimize import minimize
cimport cython
from libc.math cimport exp, floor, sqrt
from cython.parallel cimport prange

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef raster_scan(np.ndarray[double, ndim=2] img, np.ndarray[double, ndim=2] L, np.ndarray[double, ndim=2] U, np.ndarray[double, ndim=2] D):
	# cdef np.ndarray[double, ndim=2] img = np.ascontiguousarray(img1, dtype = double)
	cdef int n_rows = img.shape[0]
	cdef int n_cols = img.shape[1]
	
	
	cdef Py_ssize_t x, y
	cdef double ix
	cdef double d
	cdef double u1
	cdef double l1
	cdef double u2
	cdef double l2
	cdef double b1
	cdef double b2

	for x in xrange(1,n_rows - 1):
		for y in xrange(1,n_cols - 1):
			ix = img[x, y]
			d = D[x,y]

			u1 = U[x-1,y]
			l1 = L[x-1,y]

			u2 = U[x,y-1]
			l2 = L[x,y-1]

			b1 = max(u1,ix) - min(l1,ix)
			b2 = max(u2,ix) - min(l2,ix)

			if d <= b1 and d <= b2:
				continue
			elif b1 < d and b1 <= b2:
				D[x,y] = b1
				U[x,y] = max(u1,ix)
				L[x,y] = min(l1,ix)
			else:
				D[x,y] = b2
				U[x,y] = max(u2,ix)
				L[x,y] = min(l2,ix)

	return True

@cython.boundscheck(False)
@cython.wraparound(False)
cdef raster_scan_inv(np.ndarray[double, ndim=2] img, np.ndarray[double, ndim=2] L, np.ndarray[double, ndim=2] U, np.ndarray[double, ndim=2] D):
	cdef int n_rows
	cdef int n_cols

	n_rows = img.shape[0]
	n_cols = img.shape[1]

	cdef Py_ssize_t x, y
	cdef double ix
	cdef double d
	cdef double u1
	cdef double l1
	cdef double u2
	cdef double l2
	cdef double b1
	cdef double b2

	for x in xrange(n_rows - 2,1,-1):
		for y in xrange(n_cols - 2,1,-1):
			
			ix = img[x,y]
			d = D[x,y]

			u1 = U[x+1,y]
			l1 = L[x+1,y]

			u2 = U[x,y+1]
			l2 = L[x,y+1]

			b1 = max(u1,ix) - min(l1,ix)
			b2 = max(u2,ix) - min(l2,ix)

			if d <= b1 and d <= b2:
				continue
			elif b1 < d and b1 <= b2:
				D[x,y] = b1
				U[x,y] = max(u1,ix)
				L[x,y] = min(l1,ix)
			else:
				D[x,y] = b2
				U[x,y] = max(u2,ix)
				L[x,y] = min(l2,ix)

	return True

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mbd(np.ndarray[double, ndim=2] img, int num_iters):

	cdef np.ndarray[double, ndim=2] L = img[...]
	cdef np.ndarray[double, ndim=2] U = img[...]
	cdef np.ndarray[double, ndim=2] D = np.full_like(img, 10000000)
	D[0,:] = 0
	D[-1,:] = 0
	D[:,0] = 0
	D[:,-1] = 0

	cdef int x
	for x in xrange(0,num_iters):
		if x%2 == 1:
			raster_scan(img,L,U,D)
		else:
			raster_scan_inv(img,L,U,D)

	return D

cdef mbd_1(val, n_cols, border_thickness):
	x, v2 = val
	px, cov, px_mean = v2
	px_mean[x] = np.mean(px[x],axis=(0,1))
	px[x] = px[x].reshape((n_cols*border_thickness,3))
	cov[x] =  np.linalg.inv(np.cov(px[x].T))

cdef f(x):
	b = 10.0
	return 1.0 / (1.0 + exp(-b*(x - 0.5)))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_saliency_mbd(np.ndarray img,method='b'):
	# Saliency map calculation based on: Minimum Barrier Salient Object Detection at 80 FPS
	img_mean = np.mean(img,axis=(2))
	sal = mbd(img_mean,2)

	cdef int n_rows = img.shape[0]
	cdef int n_cols = img.shape[1]
	cdef int n_channels = img.shape[2]
	cdef double img_size = sqrt(n_rows * n_cols)
	cdef border_thickness = int(floor(0.1 * img_size))

	cdef np.ndarray img_lab = img_as_float(skimage.color.rgb2lab(img))
	
	## Process Left right
	cdef np.ndarray px_1 = np.zeros((2,border_thickness,img_lab.shape[1], img_lab.shape[2]))
	px_1[0] = img_lab[0:border_thickness,:,:]
	px_1[1] = img_lab[n_rows - border_thickness-1:-1,:,:]
	
	## Process Top bottom
	cdef np.ndarray px_2 = np.zeros((2,img_lab.shape[0],border_thickness,img_lab.shape[2]))
	px_2[0] = img_lab[:,0:border_thickness,:]
	px_2[1] = img_lab[:,n_cols - border_thickness-1:-1,:]
	
	## Combine into one array for easy access
	cdef px = [px_1,px_2]
	
	cdef  np.ndarray px_mean = np.zeros((4,3))
	cdef  np.ndarray cov = np.zeros((4,3,3))
	cdef px_new = [np.zeros((2, n_cols*border_thickness,3)), np.zeros((2, n_rows*border_thickness,3))]
	
	# with multiprocessing.Pool(processes=len(px)) as pool:
	# 	pool.starmap(mbd_1, (enumerate(zip(px, cov, px_mean)), n_cols, border_thickness))
	#TODO: work on prange implementation to not use numpy (to circumvent GIL)
	for x in xrange(4):
		px_mean[x] = np.mean(px[x//2][x%2],axis=(0,1))
		px_new[x//2][x%2] = px[x//2][x%2].reshape(((n_cols if x//2 == 0 else n_rows)*border_thickness,3))
		cov[x] =  np.linalg.inv(np.cov(px_new[x//2][x%2].T))

	u = np.zeros((4,img_lab.shape[0],img_lab.shape[1]), dtype=np.float)

	img_lab_unrolled = img_lab.reshape(img_lab.shape[0]*img_lab.shape[1],3)
	px_mean_2 = np.zeros((4,1,3), dtype=np.float)
	for x in xrange(4):
		px_mean_2[x,0] = px_mean[x]
		u[x] = scipy.spatial.distance.cdist(img_lab_unrolled,px_mean_2[x],'mahalanobis', VI=cov[x]).reshape((img_lab.shape[0],img_lab.shape[1]))
		u[x] = u[x]/np.max(u[x])
		

	u_max = np.maximum(np.maximum(u[0],u[1]),np.maximum(u[2],u[3]))

	u_final = np.sum (u,axis = 0) - u_max

	u_max_final = np.max(u_final)
	sal_max = np.max(sal)
	sal = sal / sal_max + u_final / u_max_final

	#postprocessing

	# apply centredness map
	sal = sal / np.max(sal)
	
	s = np.mean(sal)
	cdef double alpha = 50.0
	cdef double delta = alpha * sqrt(s)

	xv,yv = np.meshgrid(np.arange(sal.shape[1]),np.arange(sal.shape[0]))
	cdef int w = sal.shape[0]
	cdef int h = sal.shape[1]
	cdef double w2 = w/2.0
	cdef double h2 = h/2.0

	C = 1 - np.sqrt(np.power(xv - h2,2) + np.power(yv - w2,2)) / sqrt(np.power(w2,2) + np.power(h2,2))

	sal = sal * C

	fv = np.vectorize(f)

	sal = sal / np.max(sal)

	sal = fv(sal)

	return sal* 255.0