__author__ = 'QiYE'


from xyz_base_iter0_test_r0r1r2  import get_base_patch_by_iter0
from xyz_base_iter1_test_r0r1r2i import get_bw_xyz_err
from xyz_mid_test_r0r1r2 import get_mid_loc_err
from xyz_top_test_r0r1r2 import get_top_loc_err
from xyz_tip_test_r0r1r2 import get_tip_loc_err

from src.utils import constants


setname='icvl'
# setname='nyu'
# setname='msrc' #have not trained the model for msrc use the model of Hier_Derot instead
dataset_path_prefix=constants.Data_Path
"""change the NUM_JNTS in src/constants.py to 6"""

'''get patches in new veiwpoint by the model of the bw inital stage'''
get_base_patch_by_iter0(dataset_path_prefix=constants.Data_Path,setname=setname)
# """change the NUM_JNTS in src/constants.py to 1"""

# get_bw_xyz_err(dataset_path_prefix=constants.Data_Path, setname=setname)
# #
#
# file_format='mat'
# get_mid_loc_err(setname=setname,dataset_path_prefix=dataset_path_prefix,file_format=file_format)
# get_top_loc_err(setname=setname,dataset_path_prefix=dataset_path_prefix,file_format=file_format)
# get_tip_loc_err(setname=setname,dataset_path_prefix=dataset_path_prefix,file_format=file_format)