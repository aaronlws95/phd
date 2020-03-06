__author__ = 'QiYE'
import numpy
WRIST=0
PALM=[1,5,9,13,17]
MCP=[2,6,10,14,18]
DIP=[3,7,11,15,19]
TIP=[4,8,12,16,20]

THUMB=[1,2,3,4]
INDEX=[5,6,7,8]
MIDDLE=[9,10,11,12]
RING=[13,14,15,16]
PINKY=[17,18,19,20]

# store in euler angle
# globIdx4EulMot = [0,1,2]
# mcpIdx4EulMot=[[6,7,8],[11,12,13],[16,17,18],[21,22,23],[26,27,28]]
# dipIdx4EulMot=[[9],[14],[19],[24],[29]]
# tipIdx4EulMot=[[10],[15],[20],[25],[30]]

# store in euler quaternion
globIdx4QuatMot = [0,1,2,3]
globIdx4QuatTrans=[4,5,6]
mcpIdx4QuatMot=[[7,8,9,10],[13,14,15,16],[19,20,21,22],[25,26,27,28],[31,32,33,34]]
dipIdx4QuatMot=[[11],[17],[23],[29],[35]]
tipIdx4QuatMot=[[12],[18],[24],[30],[36]]



        # self.meanIdx = xrange(0,34,1)
        # self.palmMeanIdx=[0,1,2,3]
        # self.thumMcpMeanIdx=[ 4,  5,  6,  7]
        # self.indMcpMeanIdx = [8,  9, 10, 11]
        # self.midMcpMeanIdx=[12, 13, 14, 15]
        # self.ringMcpMeanIdx=[16,17, 18, 19]
        # self.pinkMcpMeanIdx=[20, 21, 22, 23]
        #
        # self.thumDTipMeanIdx=[24, 25]
        # self.indDTipMeanIdx=[26, 27]
        # self.midDTipMeanIdx=[28, 29]
        # self.ringDTipMeanIdx=[30, 31]
        # self.pinkDTipMeanIdx=[32, 33]
        #
        # self.meanDTipIdx=[self.thumDTipMeanIdx,self.indDTipMeanIdx,self.midDTipMeanIdx,self.ringDTipMeanIdx,self.pinkDTipMeanIdx]
        #
        # self.palmDivIdx=[34,35,36,37]
        # self.thumMcpDivIdx=[38, 39, 40, 41]
        # self.indMcpDivIdx=[42, 43, 44, 45]
        # self.midMcpDivIdx=[46, 47, 48, 49]
        # self.ringMcpDivIdx=[50,51, 52, 53]
        # self.pinkMcpDivIdx=[54, 55, 56, 57]
        #
        # # self.divMcpIdx=[self.palmDivIdx,self.thumMcpDivIdx,self.indMcpDivIdx,self.midMcpDivIdx,self.ringMcpDivIdx,self.pinkMcpDivIdx]
        # self.divIdx=[34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
        # self.thumDTipDivIdx=[58, 59]
        # self.indDTipDivIdx=[60, 61]
        # self.midDTipDivIdx=[62, 63]
        # self.ringDTipDivIdx=[64, 65]
        # self.pinkDTipDivIdx=[66, 67]
        #
        # self.divDTipDivIdx=[self.thumDTipDivIdx,self.indDTipDivIdx,self.midDTipDivIdx,self.ringDTipDivIdx,self.pinkDTipDivIdx]
        #
        # self.thumDTipCorIdx=[68]
        # self.indDTipCorIdx=[69]
        # self.midDTipCorIdx=[70]
        # self.ringDTipCorIdx=[71]
        # self.pinkDTipCorIdx=[72]
        # self.corDTipDivIdx=[self.thumDTipCorIdx,self.indDTipCorIdx,self.midDTipCorIdx,self.ringDTipCorIdx,self.pinkDTipCorIdx]

def absolute2jointoffset(label):
    label.shape=(label.shape[0],21,3)
    new_label=numpy.empty_like(label)
    new_label[:,WRIST,:] = label[:,WRIST,:]
    new_label[:,PALM,:]=label[:,PALM,:]
    new_label[:,MCP,:]=label[:,MCP,:]-label[:,PALM,:]
    new_label[:,DIP,:]=label[:,DIP,:]-label[:,MCP,:]
    new_label[:,TIP,:]=label[:,TIP,:]-label[:,DIP,:]
    new_label.shape = (new_label.shape[0],63)
    return new_label


def jointoffset2absolute(label):
    label.shape=(label.shape[0],21,3)
    new_label=numpy.empty_like(label)
    new_label[:,WRIST,:] = label[:,WRIST,:]
    new_label[:,PALM,:]=label[:,PALM,:]
    new_label[:,MCP,:]=label[:,MCP,:]+new_label[:,PALM,:]
    new_label[:,DIP,:]=label[:,DIP,:]+new_label[:,MCP,:]
    new_label[:,TIP,:]=label[:,TIP,:]+new_label[:,DIP,:]
    new_label.shape = (new_label.shape[0],63)
    return new_label