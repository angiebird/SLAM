import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import colorsys
import math

def get_Ix(im):
  kernel = [0.5, 0, -0.5]
  Ix = []
  for row in im:
    Ix.append(np.convolve(kernel, row, 'same'))
  return np.array(Ix)

def get_Iy(im):
  kernel = [0.5, 0, -0.5]
  Iy = []
  for row in im.T:
    Iy.append(np.convolve(kernel, row, 'same'))
  return np.array(Iy).T

def get_It(im_ls):
  It = -0.5 * im_ls[0] + 0.5 * im_ls[2]
  return It

def convolve_2d(kx, ky, im):
  tmp = []
  for row in im:
    tmp.append(np.convolve(kx, row, 'same'))
  tmp = np.array(tmp)
  result = []
  for row in tmp.T:
    result.append(np.convolve(ky, row, 'same'))
  return np.array(result).T

# M = [sum(g Ix^2),  sum(g Ix Iy);
#      sum(g Ix Iy), sum(g Iy^2) ;]
def get_M(Ix, Iy, g):
  M0 = convolve_2d(g, g, Ix * Ix)
  M1 = convolve_2d(g, g, Ix * Iy)
  M2 = convolve_2d(g, g, Ix * Iy)
  M3 = convolve_2d(g, g, Iy * Iy)
  return np.array([M0, M1, M2, M3])

# b = [sum(g Ix It), sum(g Iy It)]^T
def get_b(Ix, Iy, It, g):
  bx = - convolve_2d(g, g, Ix * It)
  by = - convolve_2d(g, g, Iy * It)
  return np.array([bx, by])

# motion vector u = M^-1 b
def get_u(M, b):
  rows = M.shape[1]
  cols = M.shape[2]
  u0 = np.zeros((rows, cols))
  u1 = np.zeros((rows, cols))
  for r in range(rows):
    for c in range(cols):
      m = M[:, r, c]
      bb = b[:, r, c]
      det = m[0] * m[3] - m[1] * m[2]
      if abs(det) > 0.001:
        u0[r, c] = (m[3] * bb[0] - m[1] * bb[1]) / det
        u1[r, c] = (m[0] * bb[1] - m[2] * bb[0]) / det
      else:
        # not enough info to determine motion
        u0[r, c] = 0
        u1[r, c] = 0
  return np.array([u0, u1])

def rgb_to_gray_img(im):
  rows = im.shape[0]
  cols = im.shape[1]
  gray_im = np.zeros((rows, cols))
  for r in range(rows):
    for c in range(cols):
      p = im[r, c]
      gray_im[r, c] = colorsys.rgb_to_yiq(p[0], p[1], p[2])[0]
  return gray_im

def draw_line(im, r, c, ur, uc):
  if ur == 0 and uc == 0:
    pass
  elif abs(ur) >= abs(uc):
    sgn = ur / abs(ur)
    for i in range(0, ur + sgn, sgn):
      j = round(i * 1. / ur * uc)
      im[r + i, c + j] = 0
  else:
    sgn = uc / abs(uc)
    for j in range(0, uc + sgn, sgn):
      i = round(j * 1. / uc * ur)
      im[r + i, c + j] = 0

def draw_motion(im, u):
  rows = im.shape[0]
  cols = im.shape[1]
  for r in range(3, rows, 8):
    for c in range(3, rows, 8):
      draw_line(im, r, c, int(round(u[1, r, c])), int(round(u[0, r, c])))
  plt.matshow(im)
  plt.show()
  
if __name__ == '__main__':
  #im0 = rgb_to_gray_img(mpimg.imread('0.png')[300:350, 150:200])
  #im1 = rgb_to_gray_img(mpimg.imread('1.png')[300:350, 150:200])
  #im2 = rgb_to_gray_img(mpimg.imread('2.png')[300:350, 150:200])
  im0 = np.zeros((50, 50))
  im1 = np.zeros((50, 50))
  im2 = np.zeros((50, 50))
  for r in range(50):
    for c in range(50):
      im0[r, c] = math.sin(r * 2 * math.pi/25) + math.sin(c * 2 * math.pi/33)
      im1[r, c] = math.sin((r-1) * 2 * math.pi/25) + math.sin((c+2) * 2 * math.pi/33)
      im2[r, c] = math.sin((r-2) * 2 * math.pi/25) + math.sin((c+4) * 2 * math.pi/33)
  im_ls = [im0, im1, im2]
  im = im1
  Ix = get_Ix(im)
  Iy = get_Iy(im)
  It = get_It(im_ls)
  g = [0.25, 0.75, 1, 0.75, 0.25]
  M = get_M(Ix, Iy, g)
  b = get_b(Ix, Iy, It, g)
  u = get_u(M, b)
  r = 20
  c = 20
  print 'im', im[r-1:r+2, c-1:c+2]
  print 'Ix', Ix[r-1:r+2, c-1:c+2]
  print 'Iy', Iy[r-1:r+2, c-1:c+2]
  print 'It', It[r-1:r+2, c-1:c+2]
  print 'M', M[:, r, c]
  print 'b', b[:, r, c]
  print 'u', u[:, r, c]
  draw_motion(im, u)
  plt.matshow(im2)
  plt.show()

