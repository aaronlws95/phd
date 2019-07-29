__author__ = 'QiYE'

#############input model###########################
icvl_source_name ='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200'
icvl_initial_model = 'param_cost_uvd_whl_r012_21jnts_64_96_128_1_2_adam_lm9'

icvl_bw_model=   ['param_cost_egoff_adam_iter1_bw0_r012_24_48_1_1_adam_lm29',
'param_cost_egoff_adam_iter1_bw1_r012_24_48_1_1_adam_lm0',
'param_cost_egoff_adam_iter1_bw5_r012_24_48_1_1_adam_lm3',
'param_cost_egoff_adam_iter1_bw9_r012_24_48_1_1_adam_lm29',
'param_cost_egoff_adam_iter1_bw13_r012_24_48_1_1_adam_lm3',
'param_cost_egoff_adam_iter1_bw17_r012_24_48_1_1_adam_lm3']


# idx=[2,3,4,6,7,8,10,11,12,14,15,16,18,19,20]
icvl_finger_jnt_model = [
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',

        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',

        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',

        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',

        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000'
                       ]
##########output xyz########################



#############################################################


nyu_source_name ='_nyu_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300'
nyu_initial_model = 'param_cost_uvd_bw_r012_21jnts_64_96_128_1_2_adam_lm9'
nyu_bw_model=   ['param_cost_egoff_iter1_bw0_r012_24_48_1_1_adam_lm3',
'param_cost_egoff_iter1_bw1_r012_24_48_1_1_adam_lm3',
'param_cost_egoff_iter1_bw5_r012_24_48_1_1_adam_lm3',
'param_cost_egoff_iter1_bw9_r012_24_48_1_1_adam_lm3',
'param_cost_egoff_iter1_bw13_r012_24_48_1_1_adam_lm3',
'param_cost_egoff_iter1_bw17_r012_24_48_1_1_adam_lm3']



nyu_finger_jnt_model = [
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_24_48_1_1_adam_lm1000']



##########################################################



msrc_source_name ='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300'
msrc_initial_model = 'param_cost_uvd_whl_r012_21jnts_64_96_128_1_2_adam_lm9'
#
#
#
msrc_bw_model=   ['param_cost_egoff_iter1_bw0_r012_24_48_1_1_adam_lm29',
'param_cost_egoff_iter1_bw1_r012_24_48_1_1_adam_lm29',
'param_cost_egoff_iter1_bw5_r012_24_48_1_1_adam_lm29',
'param_cost_egoff_iter1_bw9_r012_24_48_1_1_adam_lm29',
'param_cost_egoff_iter1_bw13_r012_24_48_1_1_adam_lm29',
'param_cost_egoff_iter1_bw17_r012_24_48_1_1_adam_lm29']

msrc_finger_jnt_model = [
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000',
        '_r012_21jnts_derot_patch40_8_16_1_1_adam_lm1000']


icvl_iter0_whlimg_derot = 'D:\\icvl_tmp\\icvl_iter0_whlimg_derot.h5'
icvl_iter1_whlimg_derot = 'D:\\icvl_tmp\\icvl_iter1_whlimg_derot.h5'

icvl_xyz_bw_jnt_save_path =   ['D:\\icvl_tmp\\jnt0_xyz.mat',
                       'D:\\icvl_tmp\\jnt1_xyz.mat',
                       'D:\\icvl_tmp\\jnt5_xyz.mat',
                       'D:\\icvl_tmp\\jnt9_xyz.mat',
                       'D:\\icvl_tmp\\jnt13_xyz.mat',
                       'D:\\icvl_tmp\\jnt17_xyz.mat']
icvl_xyz_finger_jnt_save_path=['D:\\icvl_tmp\\jnt2_xyz.mat',
                            'D:\\icvl_tmp\\jnt3_xyz.mat',
                            'D:\\icvl_tmp\\jnt4_xyz.mat',

                  'D:\\icvl_tmp\\jnt6_xyz.mat',
                  'D:\\icvl_tmp\\jnt7_xyz.mat',
                  'D:\\icvl_tmp\\jnt8_xyz.mat',


                  'D:\\icvl_tmp\\jnt10_xyz.mat',
                  'D:\\icvl_tmp\\jnt11_xyz.mat',
                  'D:\\icvl_tmp\\jnt12_xyz.mat',


                  'D:\\icvl_tmp\\jnt14_xyz.mat',
                  'D:\\icvl_tmp\\jnt15_xyz.mat',
                  'D:\\icvl_tmp\\jnt16_xyz.mat',

                  'D:\\icvl_tmp\\jnt18_xyz.mat',
                  'D:\\icvl_tmp\\jnt19_xyz.mat',
                  'D:\\icvl_tmp\\jnt20_xyz.mat']



msrc_iter0_whlimg_derot = 'D:\\msrc_tmp\\msrc_iter0_whlimg_derot.h5'
msrc_iter1_whlimg_derot = 'D:\\msrc_tmp\\msrc_iter1_whlimg_derot.h5'
msrc_xyz_bw_jnt_save_path =   ['D:\\msrc_tmp\\jnt0_xyz.mat',
                       'D:\\msrc_tmp\\jnt1_xyz.mat',
                       'D:\\msrc_tmp\\jnt5_xyz.mat',
                       'D:\\msrc_tmp\\jnt9_xyz.mat',
                       'D:\\msrc_tmp\\jnt13_xyz.mat',
                       'D:\\msrc_tmp\\jnt17_xyz.mat']

msrc_xyz_finger_jnt_save_path=['D:\\msrc_tmp\\jnt2_xyz.mat',
                            'D:\\msrc_tmp\\jnt3_xyz.mat',
                            'D:\\msrc_tmp\\jnt4_xyz.mat',

                  'D:\\msrc_tmp\\jnt6_xyz.mat',
                  'D:\\msrc_tmp\\jnt7_xyz.mat',
                  'D:\\msrc_tmp\\jnt8_xyz.mat',


                  'D:\\msrc_tmp\\jnt10_xyz.mat',
                  'D:\\msrc_tmp\\jnt11_xyz.mat',
                  'D:\\msrc_tmp\\jnt12_xyz.mat',


                  'D:\\msrc_tmp\\jnt14_xyz.mat',
                  'D:\\msrc_tmp\\jnt15_xyz.mat',
                  'D:\\msrc_tmp\\jnt16_xyz.mat',

                  'D:\\msrc_tmp\\jnt18_xyz.mat',
                  'D:\\msrc_tmp\\jnt19_xyz.mat',
                  'D:\\msrc_tmp\\jnt20_xyz.mat']




nyu_iter0_whlimg_derot = 'D:\\nyu_tmp\\nyu_iter0_whlimg_derot.h5'
nyu_iter1_whlimg_derot = 'D:\\nyu_tmp\\nyu_iter1_whlimg_derot.h5'
nyu_xyz_bw_jnt_save_path =   ['D:\\nyu_tmp\\jnt0_xyz.mat',
                       'D:\\nyu_tmp\\jnt1_xyz.mat',
                       'D:\\nyu_tmp\\jnt5_xyz.mat',
                       'D:\\nyu_tmp\\jnt9_xyz.mat',
                       'D:\\nyu_tmp\\jnt13_xyz.mat',
                       'D:\\nyu_tmp\\jnt17_xyz.mat']
nyu_xyz_finger_jnt_save_path =['D:\\nyu_tmp\\jnt2_xyz.mat',
                            'D:\\nyu_tmp\\jnt3_xyz.mat',
                            'D:\\nyu_tmp\\jnt4_xyz.mat',

                  'D:\\nyu_tmp\\jnt6_xyz.mat',
                  'D:\\nyu_tmp\\jnt7_xyz.mat',
                  'D:\\nyu_tmp\\jnt8_xyz.mat',

                  'D:\\nyu_tmp\\jnt10_xyz.mat',
                  'D:\\nyu_tmp\\jnt11_xyz.mat',
                  'D:\\nyu_tmp\\jnt12_xyz.mat',

                  'D:\\nyu_tmp\\jnt14_xyz.mat',
                  'D:\\nyu_tmp\\jnt15_xyz.mat',
                  'D:\\nyu_tmp\\jnt16_xyz.mat',

                  'D:\\nyu_tmp\\jnt18_xyz.mat',
                  'D:\\nyu_tmp\\jnt19_xyz.mat',
                  'D:\\nyu_tmp\\jnt20_xyz.mat']



