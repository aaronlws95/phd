__author__ = 'QiYE'

import matplotlib.pyplot as plt
from src.SphereHandModel.utils.xyz_uvd import *
from mpl_toolkits.mplot3d import Axes3D

cmap = plt.cm.rainbow
colors_map = cmap(numpy.arange(cmap.N))
rng = numpy.random.RandomState(0)
num = rng.randint(0,256,(21,))
jnt_colors = colors_map[num]
# print jnt_colors.shape
markersize = 7
linewidth=2
azim =  -177
elev = -177

def show_hand_skeleton(refSkeleton,Skeleton,tranSkeleton):
    fig = plt.figure()

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5,  marker='.')
    ax = fig.add_subplot(111, projection='3d')
    dot = refSkeleton*1000
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='r',markersize=markersize+3)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='r',linewidth=linewidth,marker='o',markersize=markersize+3)
    dot = Skeleton*1000
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='b',markersize=markersize+2)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='b',linewidth=linewidth,marker='o',markersize=markersize+2)

    dot = tranSkeleton*1000
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='g',markersize=markersize)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='g',linewidth=linewidth,marker='o',markersize=markersize)


    # ax.set_xlim(axis_bounds[5], axis_bounds[4])
    # ax.set_ylim(axis_bounds[0], axis_bounds[1])
    # ax.set_zlim(axis_bounds[2], axis_bounds[3])
    ax.azim = azim
    ax.elev = elev
    # ax.set_autoscale_on(True)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # ax.set_axis_off()
    plt.show()

def show_hand_skeleton_overlaid_by_sphere(refSkeleton,initSphere,Skeleton,Sphere,tranSkeleton,transSphere):
    fig = plt.figure()

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5,  marker='.')
    ax = fig.add_subplot(111, projection='3d')
    dot = refSkeleton*1000
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='r',markersize=markersize+3)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='r',linewidth=linewidth,marker='o',markersize=markersize+3)
    FINGER_SPHERE = [30,1,0,3,2,5,4,32,7,6,9,8,11,10,35,13,12,15,14,17,16,38,19,18,21,20,23,22,41,25,24,27,26,29,28]
    tmpShpere = initSphere[FINGER_SPHERE]*1000
    ax.scatter3D(tmpShpere[:,3],tmpShpere[:,1],tmpShpere[:,2],s=200,  marker='*',c='g')
    PALM_SPHERE = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    # PALM_SPHERE = [44,45,46,47]
    tmpShpere = initSphere[PALM_SPHERE]*1000
    ax.scatter3D(tmpShpere[:,3],tmpShpere[:,1],tmpShpere[:,2],s=180,  marker='*',c='b')


    dot = Skeleton*1000
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='b',markersize=markersize+2)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='b',linewidth=linewidth,marker='o',markersize=markersize+2)
    FINGER_SPHERE = [30,1,0,3,2,5,4,32,7,6,9,8,11,10,35,13,12,15,14,17,16,38,19,18,21,20,23,22,41,25,24,27,26,29,28]
    tmpShpere = Sphere[FINGER_SPHERE]*1000
    ax.scatter3D(tmpShpere[:,3],tmpShpere[:,1],tmpShpere[:,2],s=200,  marker='*',c='g')
    PALM_SPHERE = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    # PALM_SPHERE = [44,45,46,47]
    tmpShpere = Sphere[PALM_SPHERE]*1000
    ax.scatter3D(tmpShpere[:,3],tmpShpere[:,1],tmpShpere[:,2],s=180,  marker='*',c='b')

    dot = tranSkeleton*1000
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='g',markersize=markersize)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='g',linewidth=linewidth,marker='o',markersize=markersize)
    FINGER_SPHERE = [30,1,0,3,2,5,4,32,7,6,9,8,11,10,35,13,12,15,14,17,16,38,19,18,21,20,23,22,41,25,24,27,26,29,28]
    tmpShpere = transSphere[FINGER_SPHERE]*1000
    ax.scatter3D(tmpShpere[:,3],tmpShpere[:,1],tmpShpere[:,2],s=200,  marker='*',c='g')
    PALM_SPHERE = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    # PALM_SPHERE = [44,45,46,47]
    tmpShpere = transSphere[PALM_SPHERE]*1000
    ax.scatter3D(tmpShpere[:,3],tmpShpere[:,1],tmpShpere[:,2],s=180,  marker='*',c='b')


    # ax.set_xlim(axis_bounds[5], axis_bounds[4])
    # ax.set_ylim(axis_bounds[0], axis_bounds[1])
    # ax.set_zlim(axis_bounds[2], axis_bounds[3])
    ax.azim = azim
    ax.elev = elev
    # ax.set_autoscale_on(True)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # ax.set_axis_off()
    plt.show()


def show_single_hand_skeleton_overlaid_by_sphere(refSkeleton,initSphere):
    fig = plt.figure()

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5,  marker='.')
    ax = fig.add_subplot(111, projection='3d')
    dot = refSkeleton*1000
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='r',markersize=markersize+3)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='r',linewidth=linewidth,marker='o',markersize=markersize+3)
    # tmpShpere = initSphere[INTER_SPHERE]*1000
    # ax.scatter3D(tmpShpere[:,3],tmpShpere[:,1],tmpShpere[:,2],s=100,  marker='*')
    # tmpShpere = initSphere[SADDLE_SPHERE]*1000
    # ax.scatter3D(tmpShpere[:,3],tmpShpere[:,1],tmpShpere[:,2],s=100,  marker='*')
    FINGER_SPHERE = [30,1,0,3,2,5,4,32,7,6,9,8,11,10,35,13,12,15,14,17,16,38,19,18,21,20,23,22,41,25,24,27,26,29,28]
    tmpShpere = initSphere[FINGER_SPHERE]*1000
    ax.scatter3D(tmpShpere[:,3],tmpShpere[:,1],tmpShpere[:,2],s=200,  marker='*',c='g')
    PALM_SPHERE = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    # PALM_SPHERE = [44,45,46,47]
    tmpShpere = initSphere[PALM_SPHERE]*1000
    ax.scatter3D(tmpShpere[:,3],tmpShpere[:,1],tmpShpere[:,2],s=180,  marker='*',c='b')
    plt.show()


def show_single_hand_skeleton(Skeleton):
    fig = plt.figure()

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5,  marker='.')
    ax = fig.add_subplot(111, projection='3d')
    dot = Skeleton
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='g',markersize=markersize)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='g',linewidth=linewidth,marker='o',markersize=markersize)


    # ax.set_xlim(axis_bounds[5], axis_bounds[4])
    # ax.set_ylim(axis_bounds[0], axis_bounds[1])
    # ax.set_zlim(axis_bounds[2], axis_bounds[3])
    ax.azim = azim
    ax.elev = elev
    # ax.set_autoscale_on(True)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # ax.set_axis_off()
    plt.show()




def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = numpy.mgrid[0:2*numpy.pi:20j, 0:numpy.pi:10j]
    x=numpy.cos(u)*numpy.sin(v)
    y=numpy.sin(u)*numpy.sin(v)
    z=numpy.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)
#
# def drawMultiShpere(x,y,z,r,ax):
#
#     # draw a sphere for each data point
#     for (xi,yi,zi,ri) in zip(x,y,z,r):
#         (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
#         ax.plot_wireframe(xs, ys, zs, color="r")
#
#
#     plt.show()

facecolor = numpy.random.rand(21,3)
def ShowPointCloudFromDepth(setname,depth,hand_points,Sphere):

    #Visualize the hand and the joints in 3D
    uvd = convert_depth_to_uvd(depth)
    xyz = uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)


    axis_bounds = numpy.array([numpy.min(hand_points[:, 0]), numpy.max(hand_points[:, 0]),
                               numpy.min(hand_points[:, 1]), numpy.max(hand_points[:, 1]),
                               numpy.min(hand_points[:, 2]), numpy.max(hand_points[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= 20
    axis_bounds[~mask] += 20
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]


    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(depth,'gray')

    ax = fig.add_subplot(122, projection='3d')
    # print numpy.max(points0[:, 2]), numpy.min(points0[:, 2])
    # print numpy.max(points0[:, 1]), numpy.min(points0[:, 1])
    # print numpy.max(points0[:, 0]), numpy.min(points0[:, 0])

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5, c=colors, marker='.')
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5, marker='.')
    # for (xi,yi,zi,ri) in zip(Sphere[:,1],Sphere[:,2],Sphere[:,3],Sphere[:,4],):
    #     (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
    #     ax.plot_wireframe(zs,xs, ys, color="r")
    #
    #
    # ax.set_xlim(axis_bounds[5], axis_bounds[4])
    # ax.set_ylim(axis_bounds[0], axis_bounds[1])
    # ax.set_zlim(axis_bounds[2], axis_bounds[3])
    # ax.azim = -30
    # ax.elev = 10
    # pylab.show()

    ax.scatter3D(points0[:, 0], points0[:, 1], points0[:, 2], s=1.5, marker='.')
    ax.scatter3D(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], s=150, marker='*',c='g')
    for (xi,yi,zi,ri) in zip(Sphere[:,1],Sphere[:,2],Sphere[:,3],Sphere[:,4],):
        (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
        ax.plot_wireframe(xs, ys, zs,rstride=4,cstride=4,color="r")

    for idx in range(21):
        for (xi,yi,zi,ri) in zip(Sphere[idx,:,1],Sphere[idx,:,2],Sphere[idx,:,3],Sphere[idx,:,4]):
            (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
            ax.plot_wireframe(xs, ys, zs,color=facecolor[idx],alpha=1)
    plt.grid(b='on',which='both')
    plt.show()
    # ax.set_xlim(axis_bounds[5], axis_bounds[4])
    # ax.set_ylim(axis_bounds[0], axis_bounds[1])
    # ax.set_zlim(axis_bounds[2], axis_bounds[3])
    # ax.azim = -30
    # ax.elev = 10

    plt.show()



def ShowPointCloud(points0,hand_points,Sphere):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # print numpy.max(points0[:, 2]), numpy.min(points0[:, 2])
    # print numpy.max(points0[:, 1]), numpy.min(points0[:, 1])
    # print numpy.max(points0[:, 0]), numpy.min(points0[:, 0])

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5, c=colors, marker='.')
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5, marker='.')
    # for (xi,yi,zi,ri) in zip(Sphere[:,1],Sphere[:,2],Sphere[:,3],Sphere[:,4],):
    #     (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
    #     ax.plot_wireframe(zs,xs, ys, color="r")
    #
    #
    # ax.set_xlim(axis_bounds[5], axis_bounds[4])
    # ax.set_ylim(axis_bounds[0], axis_bounds[1])
    # ax.set_zlim(axis_bounds[2], axis_bounds[3])
    # ax.azim = -30
    # ax.elev = 10
    # pylab.show()

    ax.scatter3D(points0[:, 0], points0[:, 1], points0[:, 2], s=1.5, marker='.')
    ax.scatter3D(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], s=150, marker='*')
    # for (xi,yi,zi,ri) in zip(Sphere[:,1],Sphere[:,2],Sphere[:,3],Sphere[:,4],):
    #     (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
    #     ax.plot_wireframe(xs, ys, zs,rstride=4,cstride=4,color="r")


    # ax.set_xlim(axis_bounds[5], axis_bounds[4])
    # ax.set_ylim(axis_bounds[0], axis_bounds[1])
    # ax.set_zlim(axis_bounds[2], axis_bounds[3])
    # ax.azim = -30
    # ax.elev = 10
    plt.show()


def ShowPointCloudFromDepth2(setname,depth,hand_points,Sphere):

    #Visualize the hand and the joints in 3D
    uvd = convert_depth_to_uvd(depth)
    xyz = uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)


    axis_bounds = numpy.array([numpy.min(hand_points[:, 0]), numpy.max(hand_points[:, 0]),
                               numpy.min(hand_points[:, 1]), numpy.max(hand_points[:, 1]),
                               numpy.min(hand_points[:, 2]), numpy.max(hand_points[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= 20
    axis_bounds[~mask] += 20
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]


    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(depth,'gray')

    ax = fig.add_subplot(122, projection='3d')
    # print numpy.max(points0[:, 2]), numpy.min(points0[:, 2])
    # print numpy.max(points0[:, 1]), numpy.min(points0[:, 1])
    # print numpy.max(points0[:, 0]), numpy.min(points0[:, 0])

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5, c=colors, marker='.')
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5, marker='.')
    # for (xi,yi,zi,ri) in zip(Sphere[:,1],Sphere[:,2],Sphere[:,3],Sphere[:,4],):
    #     (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
    #     ax.plot_wireframe(zs,xs, ys, color="r")
    #
    #
    # ax.set_xlim(axis_bounds[5], axis_bounds[4])
    # ax.set_ylim(axis_bounds[0], axis_bounds[1])
    # ax.set_zlim(axis_bounds[2], axis_bounds[3])
    # ax.azim = -30
    # ax.elev = 10
    # pylab.show()

    ax.scatter3D(points0[:, 0], points0[:, 1], points0[:, 2], s=6, marker='.')
    ax.scatter3D(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], s=150, marker='*',c='g')
    for idx in range(21):
        for (xi,yi,zi,ri) in zip(Sphere[idx,:,1],Sphere[idx,:,2],Sphere[idx,:,3],Sphere[idx,:,4]):
            (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
            ax.plot_wireframe(xs, ys, zs,color=facecolor[idx],alpha=1)
    plt.grid(b='on',which='both')
    plt.show()
    # ax.set_xlim(axis_bounds[5], axis_bounds[4])
    # ax.set_ylim(axis_bounds[0], axis_bounds[1])
    # ax.set_zlim(axis_bounds[2], axis_bounds[3])
    # ax.azim = -30
    # ax.elev = 10

    plt.show()


