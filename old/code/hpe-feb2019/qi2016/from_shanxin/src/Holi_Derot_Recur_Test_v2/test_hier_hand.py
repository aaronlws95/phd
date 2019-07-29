__author__ = 'QiYE'


from xyz_base_iter0_test_r0r1r2  import get_bw_initial_xyz_err
from xyz_base_iter1_test_r0r1r2i import get_bw_xyz_err
from xyz_fingers_iter0_test_r0r1r2 import get_finger_jnt_loc_err

from src.utils import constants


# setname='icvl'
setname='nyu'
# setname='msrc' #have not trained the model for msrc use the model of Hier_Derot instead
dataset_path_prefix=constants.Data_Path
"""change the NUM_JNTS in src/constants.py to 21"""

'''get patches in new veiwpoint by the model of the bw inital stage'''
get_bw_initial_xyz_err(dataset_path_prefix=constants.Data_Path,setname=setname)
# """change the NUM_JNTS in src/constants.py to 1"""

get_bw_xyz_err(dataset_path_prefix=constants.Data_Path, setname=setname)
file_format='mat'

get_finger_jnt_loc_err(setname=setname,dataset_path_prefix=dataset_path_prefix,file_format=file_format)
