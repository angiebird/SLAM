import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import numpy as np

class RGBDImage:
  def __init__(self, rgbImgPath, depthImgPath):
    self.rgbImg = mpimg.imread(rgbImgPath)
    self.depthImg = mpimg.imread(depthImgPath)
    self.pointCloud = []
    self.textureMap = []
    self.textureList = []
    fx = 525.0
    fy = 525.0
    cx = 319.5
    cy = 239.5

    factor = 5000
    for v in range(self.depthImg.shape[0]):
      for u in range(self.depthImg.shape[1]):
        z = self.depthImg[v, u] / factor
        t = self.rgbImg[v, u]
        if z != 0:
          x = (u - cx) * z / fx
          y = (v - cy) * z / fy
          self.pointCloud.append([x, y, z])
          self.textureMap.append([u, v])
          self.textureList.append(t)
    self.pointCloud = np.array(self.pointCloud)


  def show(self):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #X = [1, 2, 3, 4]
    #Y = [3, 4, 1, 5]
    #Z = [0, 3, 4, 3] 
    ax.scatter(self.pointCloud[:, 0], self.pointCloud[:, 1], self.pointCloud[:, 2], c='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plt.show()

  def show_obj(self):
    for p in self.pointCloud:
      print 'v '+' '.join([str(v) for v in p])
      print 

    for p in self.textureMap:
      print 'vt '+' '.join([str(v) for v in p])

  def show_js(self):
    print 'var pLs = ['
    for p in self.pointCloud:
      print '[' + ', '.join([str(v) for v in p]) + '],'
    print ']'
    print
    print 'var tLs = ['
    for t in self.textureList:
      print '[' + ', '.join([str(v) for v in t]) + '],'
    print ']'

    #for p in self.textureMap:
    #  print 'vt '+' '.join([str(v) for v in p])

if __name__ == "__main__":
  path = '/Users/angiebird/code/SLAM/rgbd_dataset_freiburg1_xyz/'
  rgbFile = '1305031102.175304.png'
  depthFile = '1305031102.160407.png'
  rgbd = RGBDImage(path + '/rgb/' + rgbFile, path + '/depth/' + depthFile)
  rgbd.show_js()

