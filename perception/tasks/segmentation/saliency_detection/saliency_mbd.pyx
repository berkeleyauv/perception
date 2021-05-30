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

	cdef np.ndarray[double, ndim=2] L = np.copy(img)
	cdef np.ndarray[double, ndim=2] U = np.copy(img)
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
	border_thickness = int(floor(0.1 * img_size))

	cdef np.ndarray img_lab = img_as_float(skimage.color.rgb2lab(img))
		
	px_left = img_lab[0:border_thickness,:,:]
	px_right = img_lab[n_rows - border_thickness-1:-1,:,:]

	px_top = img_lab[:,0:border_thickness,:]
	px_bottom = img_lab[:,n_cols - border_thickness-1:-1,:]
		
	px_mean_left = np.mean(px_left,axis=(0,1))
	px_mean_right = np.mean(px_right,axis=(0,1))
	px_mean_top = np.mean(px_top,axis=(0,1))
	px_mean_bottom = np.mean(px_bottom,axis=(0,1))


	px_left = px_left.reshape((n_cols*border_thickness,3))
	px_right = px_right.reshape((n_cols*border_thickness,3))

	px_top = px_top.reshape((n_rows*border_thickness,3))
	px_bottom = px_bottom.reshape((n_rows*border_thickness,3))

	cov_left = np.cov(px_left.T)
	cov_right = np.cov(px_right.T)

	cov_top = np.cov(px_top.T)
	cov_bottom = np.cov(px_bottom.T)

	cov_left = np.linalg.inv(cov_left)
	cov_right = np.linalg.inv(cov_right)
		
	cov_top = np.linalg.inv(cov_top)
	cov_bottom = np.linalg.inv(cov_bottom)


	u_left = np.zeros(sal.shape)
	u_right = np.zeros(sal.shape)
	u_top = np.zeros(sal.shape)
	u_bottom = np.zeros(sal.shape)

	u_final = np.zeros(sal.shape)
	img_lab_unrolled = img_lab.reshape(img_lab.shape[0]*img_lab.shape[1],3)

	px_mean_left_2 = np.zeros((1,3))
	px_mean_left_2[0,:] = px_mean_left

	u_left = scipy.spatial.distance.cdist(img_lab_unrolled,px_mean_left_2,'mahalanobis', VI=cov_left)
	u_left = u_left.reshape((img_lab.shape[0],img_lab.shape[1]))

	px_mean_right_2 = np.zeros((1,3))
	px_mean_right_2[0,:] = px_mean_right

	u_right = scipy.spatial.distance.cdist(img_lab_unrolled,px_mean_right_2,'mahalanobis', VI=cov_right)
	u_right = u_right.reshape((img_lab.shape[0],img_lab.shape[1]))

	px_mean_top_2 = np.zeros((1,3))
	px_mean_top_2[0,:] = px_mean_top

	u_top = scipy.spatial.distance.cdist(img_lab_unrolled,px_mean_top_2,'mahalanobis', VI=cov_top)
	u_top = u_top.reshape((img_lab.shape[0],img_lab.shape[1]))

	px_mean_bottom_2 = np.zeros((1,3))
	px_mean_bottom_2[0,:] = px_mean_bottom

	u_bottom = scipy.spatial.distance.cdist(img_lab_unrolled,px_mean_bottom_2,'mahalanobis', VI=cov_bottom)
	u_bottom = u_bottom.reshape((img_lab.shape[0],img_lab.shape[1]))

	max_u_left = np.max(u_left)
	max_u_right = np.max(u_right)
	max_u_top = np.max(u_top)
	max_u_bottom = np.max(u_bottom)

	u_left = u_left / max_u_left
	u_right = u_right / max_u_right
	u_top = u_top / max_u_top
	u_bottom = u_bottom / max_u_bottom

	u_max = np.maximum(np.maximum(np.maximum(u_left,u_right),u_top),u_bottom)

	u_final = (u_left + u_right + u_top + u_bottom) - u_max

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