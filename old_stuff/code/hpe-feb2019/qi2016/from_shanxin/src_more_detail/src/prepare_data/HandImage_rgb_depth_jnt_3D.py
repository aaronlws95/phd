__author__ = 'QiYE'
"""
this file is to show the hand of NYU's dataset in 3D
"""""
import numpy
from PIL import Image
import pylab
import matplotlib.colors as colors
import scipy.io
from mpl_toolkits.mplot3d import Axes3D

def convert_uvd_to_xyz(uvd):
    xRes = 640
    yRes = 480
    xzfactor = 1.08836701
    yzfactor = 0.817612648
    normalizedX = numpy.asarray(uvd[:, :, 0], dtype='float32') / xRes - 0.5
    normalizedY = 0.5 - numpy.asarray(uvd[:, :, 1], dtype='float32') / yRes
    xyz = numpy.zeros(uvd.shape)
    xyz[:, :, 2] = numpy.asarray(uvd[:, :, 2], dtype='float32')
    xyz[:, :, 0] = normalizedX * xyz[:, :, 2] * xzfactor
    xyz[:, :, 1] = normalizedY * xyz[:, :, 2] * yzfactor
    return xyz

def convert_depth_to_uvd(depth):
    v, u = pylab.meshgrid(range(1, depth.shape[0] + 1, 1), range(1, depth.shape[1] + 1, 1), indexing= 'ij')
    v = numpy.asarray(v, 'uint16')[:, :, numpy.newaxis]
    u = numpy.asarray(u, 'uint16')[:, :, numpy.newaxis]
    depth = depth[:, :, numpy.newaxis]
    uvd = numpy.concatenate((u, v, depth), axis=2)
    print v.shape,u.shape,uvd.shape
    return uvd
def convert_xyz_to_uvd(xyz):
    halfResX = 640/2
    halfResY = 480/2
    coeffX = 588.036865
    coeffY = 587.075073
    uvd = numpy.zeros(xyz.shape)
    uvd[:, :, 0] = numpy.asarray(coeffX * xyz[:, :, 0] / xyz[:, :, 2] + halfResX, dtype='uint16')
    uvd[:, :, 1] = numpy.asarray(halfResY*1 - coeffY * xyz[:, :, 1] / xyz[:, :, 2], dtype='uint16')
    uvd[:, :, 2] = xyz[:, :, 2]
    return uvd

dataset_dir =  'D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\NYU_dataset\\NYU_dataset\\test\\'
image_index = 2
kinect_index = 1
filename_prefix = "%d_%07d" % (kinect_index, image_index)

#load and display an RGB example
rgb = numpy.array(Image.open('%srgb_%s.png' % (dataset_dir, filename_prefix)))
# pylab.imshow(rgb)
# pylab.show()

#load and display a depth example
#The top 8 bits of depth are packed into green and the lower 8 bits into blue.
depth = Image.open('%sdepth_%s.png' % (dataset_dir, filename_prefix))
depth = numpy.asarray(depth, dtype='uint16')
depth = depth[:, :, 2]+numpy.left_shift(depth[:, :, 1], 8)
xyz_feedback= numpy.loadtxt('D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\Exp_Result_Prior_Work\\ICCV15_Feedback\\CVWW15_NYU_Prior-Refinement.txt')/1000
xyz_feedback.shape=(xyz_feedback.shape[0],14,3)
uvd_fed = convert_xyz_to_uvd(xyz_feedback)
# pylab.imshow(rgb)
# pylab.show()
pylab.imshow(depth, cmap='gray', norm=colors.Normalize(vmin=0, vmax=numpy.max(depth)))
print uvd_fed[image_index-1,:,:]
pylab.scatter(uvd_fed[image_index-1,0:8,0],uvd_fed[image_index-1,0:8,1],s=20,c='r')
pylab.show()
#load and display a synthetic depth example
#The top 8 bits of depth are packed into green and the lower 8 bits into blue.
synthdepth0 = Image.open('%ssynthdepth_%s.png' % (dataset_dir, filename_prefix))
synthdepth0= numpy.asarray(synthdepth0, dtype='uint16')
synthdepth = synthdepth0[:, :, 2]+numpy.left_shift(synthdepth0[:, :, 1], 8)
loc = numpy.where(synthdepth > 0)
depthImg = pylab.imshow(synthdepth, cmap='gray',
                        norm=colors.Normalize(vmin=numpy.min(synthdepth[loc]) - 10,vmax = numpy.max(synthdepth[loc]) + 10, clip = None))

#Load the UVD Coefficients and display them on the depth image
keypoints = scipy.io.loadmat('%sjoint_data.mat' % dataset_dir)
joint_uvd = keypoints['joint_uvd']
jnt_uvd = numpy.squeeze(joint_uvd[kinect_index-1, image_index-1, :, :])
jnt_colors = numpy.random.rand(jnt_uvd.shape[0], 3)
# pylab.scatter(jnt_uvd[:, 0], jnt_uvd[:, 1], s=20, c=jnt_colors)
# pylab.show()

#Visualize the hand and the joints in 3D
uvd = convert_depth_to_uvd(depth)
xyz = convert_uvd_to_xyz(uvd)
points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)
colors = rgb.reshape(rgb.shape[0]*rgb.shape[1], 3)
jnt_uvd.shape = (1, jnt_uvd.shape[0], 3)
hand_points = numpy.squeeze(convert_uvd_to_xyz(jnt_uvd))
#% Collect the points within the AABBOX of the hand
axis_bounds = numpy.array([numpy.min(hand_points[:, 0]), numpy.max(hand_points[:, 0]),
                           numpy.min(hand_points[:, 1]), numpy.max(hand_points[:, 1]),
                           numpy.min(hand_points[:, 2]), numpy.max(hand_points[:, 2])])
mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
axis_bounds[mask] -= 20
axis_bounds[~mask] += 20
mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])

index1 = numpy.where(mask1 == True)
index2 = numpy.where(mask2 == True)
index3 = numpy.where(mask3 == True)
inputs = mask1 & mask2 & mask3
points0 = points[inputs]
colors = numpy.asarray(colors[inputs], dtype='float32') / 255

fig = pylab.figure()
ax = fig.add_subplot(111, projection='3d')
print numpy.max(points0[:, 2]), numpy.min(points0[:, 2])
print numpy.max(points0[:, 1]), numpy.min(points0[:, 1])
print numpy.max(points0[:, 0]), numpy.min(points0[:, 0])

#ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5, c=colors, marker='.')
#ax.plot_surface(points0[:, 2],points0[:, 0], points0[:, 1],rstride=1, cstride=1,color='b')
ax.set_xlim(axis_bounds[5], axis_bounds[4])
ax.set_ylim(axis_bounds[0], axis_bounds[1])
ax.set_zlim(axis_bounds[2], axis_bounds[3])
ax.azim = -30
ax.elev = 10
#ax.set_autoscale_on(True)
#ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]],                  [axis_bounds[0] * 2 , axis_bounds[1]* 2], [axis_bounds[2], axis_bounds[3]])

#Visualize the hand and the joints in 3D
uvd = convert_depth_to_uvd(synthdepth)
xyz = convert_uvd_to_xyz(uvd)
points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)
colors = rgb.reshape(rgb.shape[0]*rgb.shape[1], 3)

hand_points = numpy.squeeze(convert_uvd_to_xyz(jnt_uvd))
#% Collect the points within the AABBOX of the hand
axis_bounds = numpy.array([numpy.min(hand_points[:, 0]), numpy.max(hand_points[:, 0]),
                           numpy.min(hand_points[:, 1]), numpy.max(hand_points[:, 1]),
                           numpy.min(hand_points[:, 2]), numpy.max(hand_points[:, 2])])
mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
axis_bounds[mask] -= 20
axis_bounds[~mask] += 20
mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])

index1 = numpy.where(mask1 == True)
index2 = numpy.where(mask2 == True)
index3 = numpy.where(mask3 == True)
inputs = mask1 & mask2 & mask3
points0 = points[inputs]
colors = numpy.asarray(colors[inputs], dtype='float32') / 255
fig = pylab.figure()
ax = fig.add_subplot(111, projection='3d')
print numpy.max(points0[:, 2]), numpy.min(points0[:, 2])
print numpy.max(points0[:, 1]), numpy.min(points0[:, 1])
print numpy.max(points0[:, 0]), numpy.min(points0[:, 0])

#ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5, c=colors, marker='.')
joint_xyz= keypoints['joint_xyz']
ax.scatter(joint_xyz[image_index,:,0], joint_xyz[image_index,:,1], joint_xyz[image_index,:,2], c = 'r',marker='o',s=40)


ax.plot_surface(points0[:, 2],points0[:, 0], points0[:, 1],rstride=1, cstride=1,color='b')
ax.set_xlim(axis_bounds[5], axis_bounds[4])
ax.set_ylim(axis_bounds[0], axis_bounds[1])
ax.set_zlim(axis_bounds[2], axis_bounds[3])
ax.azim = -90
ax.elev = 0
ax.set_autoscale_on(True)
ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]],                  [axis_bounds[0] * 2 , axis_bounds[1]* 2], [axis_bounds[2], axis_bounds[3]])
pylab.show()


