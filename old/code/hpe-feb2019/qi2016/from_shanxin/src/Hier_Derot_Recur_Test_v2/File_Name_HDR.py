__author__ = 'QiYE'

#############input model###########################
icvl_source_name ='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200'
icvl_bw_initial_model = 'param_cost_uvd_bw_r012_21jnts_64_96_128_1_1_adam_lm9_ep33'

icvl_bw_model= ['param_cost_egoff_iter1_bw0_beta0_24_48_1_1_adam_lm300',
                 'param_cost_egoff_iter1_bw1_beta0_24_48_1_1_adam_lm300',
                 'param_cost_egoff_iter1_bw5_beta0_24_48_1_1_adam_lm300',
                 'param_cost_egoff_iter1_bw9_beta0_24_48_1_1_adam_lm300',
                 'param_cost_egoff_iter1_bw13_beta0_24_48_1_1_adam_lm300',
                 'param_cost_egoff_iter1_bw17_r012_24_48_1_1_adam_lm300']


icvl_mid_model =   ['param_cost_offset_mid2_patch40_beta6_24_48_1_1_adam_lm3',
                       'param_cost_offset_mid6_patch40_beta6_24_48_1_1_adam_lm3',
                       'param_cost_offset_mid10_patch40_beta6_24_48_1_1_adam_lm3',
                       'param_cost_offset_mid14_patch40_beta6_24_48_1_1_adam_lm3',
                       'param_cost_offset_mid18_patch40_beta6_24_48_1_1_adam_lm3']

icvl_top_model =   ['param_cost_offset_top3_patch40_beta6_24_48_1_1_adam_lm3',
                       'param_cost_offset_top7_patch40_beta6_24_48_1_1_adam_lm3',
                       'param_cost_offset_top11_patch40_beta6_24_48_1_1_adam_lm3',
                       'param_cost_offset_top15_patch40_beta6_24_48_1_1_adam_lm3',
                       'param_cost_offset_top19_patch40_beta6_24_48_1_1_adam_lm3']

icvl_tip_model =   ['param_cost_offset_tip4_patch40_beta6_24_48_1_1_adam_lm3',
                       'param_cost_offset_tip8_patch40_beta6_24_48_1_1_adam_lm3',
                       'param_cost_offset_tip12_patch40_beta6_24_48_1_1_adam_lm3',
                       'param_cost_offset_tip16_patch40_beta6_24_48_1_1_adam_lm3',
                       'param_cost_offset_tip20_patch40_beta6_24_48_1_1_adam_lm3']

##########output xyz########################
icvl_iter0_whlimg_derot = 'D:\\icvl_tmp\\icvl_iter0_whlimg_derot.h5'
icvl_iter1_whlimg_derot = 'D:\\icvl_tmp\\icvl_iter1_whlimg_derot.h5'

icvl_xyz_bw_jnt_save_path =   ['D:\\icvl_tmp\\jnt0_xyz.mat',
                       'D:\\icvl_tmp\\jnt1_xyz.mat',
                       'D:\\icvl_tmp\\jnt5_xyz.mat',
                       'D:\\icvl_tmp\\jnt9_xyz.mat',
                       'D:\\icvl_tmp\\jnt13_xyz.mat',
                       'D:\\icvl_tmp\\jnt17_xyz.mat']
icvl_xyz_mid_jnt_save_path=['D:\\icvl_tmp\\jnt2_xyz.mat',
                  'D:\\icvl_tmp\\jnt6_xyz.mat',
                  'D:\\icvl_tmp\\jnt10_xyz.mat',
                  'D:\\icvl_tmp\\jnt14_xyz.mat',
                  'D:\\icvl_tmp\\jnt18_xyz.mat']


icvl_xyz_top_jnt_save_path=['D:\\icvl_tmp\\jnt3_xyz.mat',
              'D:\\icvl_tmp\\jnt7_xyz.mat',
              'D:\\icvl_tmp\\jnt11_xyz.mat',
              'D:\\icvl_tmp\\jnt15_xyz.mat',
              'D:\\icvl_tmp\\jnt19_xyz.mat']

icvl_xyz_tip_jnt_save_path=['D:\\icvl_tmp\\jnt4_xyz.mat',
              'D:\\icvl_tmp\\jnt8_xyz.mat',
              'D:\\icvl_tmp\\jnt12_xyz.mat',
              'D:\\icvl_tmp\\jnt16_xyz.mat',
              'D:\\icvl_tmp\\jnt20_xyz.mat']


#############################################################
nyu_xyz_bw_jnt_save_path =   ['D:\\nyu_tmp\\jnt0_xyz.mat',
                       'D:\\nyu_tmp\\jnt1_xyz.mat',
                       'D:\\nyu_tmp\\jnt5_xyz.mat',
                       'D:\\nyu_tmp\\jnt9_xyz.mat',
                       'D:\\nyu_tmp\\jnt13_xyz.mat',
                       'D:\\nyu_tmp\\jnt17_xyz.mat']
nyu_xyz_mid_jnt_save_path =['D:\\nyu_tmp\\jnt2_xyz.mat',
              'D:\\nyu_tmp\\jnt6_xyz.mat',
              'D:\\nyu_tmp\\jnt10_xyz.mat',
              'D:\\nyu_tmp\\jnt14_xyz.mat',
              'D:\\nyu_tmp\\jnt18_xyz.mat']

nyu_xyz_top_jnt_save_path =['D:\\nyu_tmp\\jnt3_xyz.mat',
              'D:\\nyu_tmp\\jnt7_xyz.mat',
              'D:\\nyu_tmp\\jnt11_xyz.mat',
              'D:\\nyu_tmp\\jnt15_xyz.mat',
              'D:\\nyu_tmp\\jnt19_xyz.mat']

nyu_xyz_tip_jnt_save_path=['D:\\nyu_tmp\\jnt4_xyz.mat',
              'D:\\nyu_tmp\\jnt8_xyz.mat',
              'D:\\nyu_tmp\\jnt12_xyz.mat',
              'D:\\nyu_tmp\\jnt16_xyz.mat',
              'D:\\nyu_tmp\\jnt20_xyz.mat']




nyu_bw_initial_patch = 'D:\\nyu_tmp\\nyu_bw_initial_patch.h5'
nyu_iter1_whlimg_derot = 'D:\\nyu_tmp\\nyu_iter_whlimg_derot.h5'
# nyu_source_name ='_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300'
nyu_source_name ='_nyu_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300'
nyu_bw_initial_model = 'param_cost_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm9900_lm1038_yt0_ep2020'


nyu_bw_model=   ['param_cost_uvd_bw0_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep610',
'param_cost_uvd_bw1_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep655',
'param_cost_uvd_bw2_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep450',
'param_cost_uvd_bw3_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep405',
'param_cost_uvd_bw4_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep450',
'param_cost_uvd_bw5_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep440']

nyu_mid_model =   ['param_cost_offset_mid2_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep2000',
                       'param_cost_offset_mid6_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep1970',
                       'param_cost_offset_mid10_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep2000',
                       'param_cost_offset_mid14_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep2000',
                       'param_cost_offset_mid18_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep2000']

nyu_top_model =   ['param_cost_offset_top3_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep2000',
                       'param_cost_offset_top7_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep1585',
                       'param_cost_offset_top11_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep2000',
                       'param_cost_offset_top15_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep1975',
                       'param_cost_offset_top19_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep1285']

nyu_tip_model =   ['param_cost_offset_tip4_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep1600',
                       'param_cost_offset_tip8_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep730',
                       'param_cost_offset_tip12_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep1600',
                       'param_cost_offset_tip16_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep1995',
                       'param_cost_offset_tip20_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep965']



##########################################################
msrc_xyz_bw_jnt_save_path =   ['D:\\msrc_tmp\\jnt0_xyz.mat',
                       'D:\\msrc_tmp\\jnt1_xyz.mat',
                       'D:\\msrc_tmp\\jnt5_xyz.mat',
                       'D:\\msrc_tmp\\jnt9_xyz.mat',
                       'D:\\msrc_tmp\\jnt13_xyz.mat',
                       'D:\\msrc_tmp\\jnt17_xyz.mat']

msrc_xyz_mid_jnt_save_path=['D:\\msrc_tmp\\jnt2_xyz.mat',
              'D:\\msrc_tmp\\jnt6_xyz.mat',
              'D:\\msrc_tmp\\jnt10_xyz.mat',
              'D:\\msrc_tmp\\jnt14_xyz.mat',
              'D:\\msrc_tmp\\jnt18_xyz.mat']


msrc_xyz_top_jnt_save_path =['D:\\msrc_tmp\\jnt3_xyz.mat',
              'D:\\msrc_tmp\\jnt7_xyz.mat',
              'D:\\msrc_tmp\\jnt11_xyz.mat',
              'D:\\msrc_tmp\\jnt15_xyz.mat',
              'D:\\msrc_tmp\\jnt19_xyz.mat']

msrc_xyz_tip_jnt_save_path=['D:\\msrc_tmp\\jnt4_xyz.mat',
              'D:\\msrc_tmp\\jnt8_xyz.mat',
              'D:\\msrc_tmp\\jnt12_xyz.mat',
              'D:\\msrc_tmp\\jnt16_xyz.mat',
              'D:\\msrc_tmp\\jnt20_xyz.mat']


msrc_bw_initial_patch = 'D:\\msrc_tmp\\msrc_bw_initial_patch.h5'
msrc_iter1_whlimg_derot = 'D:\\msrc_tmp\\msrc_iter_whlimg_derot.h5'


msrc_source_name ='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300'
msrc_bw_initial_model = 'param_cost_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm2000_yt0_ep1500'



msrc_bw_model=   ['param_cost_uvd_bw0_r012_egoff_c0064_h11_h22_gm0_lm3000_yt0_ep155',
'param_cost_uvd_bw1_r012_egoff_c0064_h11_h22_gm0_lm3000_yt0_ep170',
'param_cost_uvd_bw2_r012_egoff_c0064_h11_h22_gm0_lm3000_yt0_ep105',
'param_cost_uvd_bw3_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep75',
'param_cost_uvd_bw4_r012_egoff_c0064_h11_h22_gm0_lm3000_yt0_ep200',
'param_cost_uvd_bw5_r012_egoff2_c0064_h11_h22_gm0_lm3000_yt0_ep185']
#
# msrc_mid_model =   ['param_cost_offset_mid2_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep630',
#                        'param_cost_offset_mid6_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep2000',
#                        'param_cost_offset_mid10_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep255',
#                        'param_cost_offset_mid14_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep265',
#                        'param_cost_offset_mid18_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep735']
#
# msrc_top_model =   ['param_cost_offset_top3_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep535',
#                        'param_cost_offset_top7_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep545',
#                        'param_cost_offset_top11_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep970',
#                        'param_cost_offset_top15_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep550',
#                        'param_cost_offset_top19_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep1215']
#
# msrc_tip_model =   ['param_cost_offset_tip4_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep290',
#                        'param_cost_offset_tip8_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep310',
#                        'param_cost_offset_tip12_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep285',
#                        'param_cost_offset_tip16_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep635',
#                        'param_cost_offset_tip20_r012_21jnts_derot_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep1490']
