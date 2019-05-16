__author__ = 'QiYE'

# constants for palm center

NUM_JNTS=1



OUT_DIM =3

Num_Class=60
Data_Path='C:/Proj/Proj_Hier_Hand_eccv2016_07042016/Proj_CNN_Hier/'

# ##CNN_Model_multi3 in the initial of holi_derot_recur, a whole CNN the kernels
# ## and the MID_ROOT is used for rotation.py
# """def recur_derot(r0,r1,r2,pred_uvd,gr_uvd,batch_size):
#     rotation = get_rot(pred_uvd,0,constants.MID_ROOT) """
# conv00_kern_size=5
# conv00_pool_size=4
# conv10_kern_size=3
# conv10_pool_size=2
# conv20_kern_size=3
# conv20_pool_size=2
# MID_ROOT=9

#CNN_Model_multi3_conv3 in the initial of hier_derot_recur, a bw CNN the kernels are
MID_ROOT=3
conv00_kern_size=5
conv00_pool_size=4
conv10_kern_size=5
conv10_pool_size=2
conv20_kern_size=5
conv20_pool_size=2

