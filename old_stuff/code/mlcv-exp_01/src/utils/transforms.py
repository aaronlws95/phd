import numpy as np
import cv2

def _get_corners(bboxes):
    """ Get corners of bounding boxes
    Args:
        bboxes: Numpy array containing bounding boxes of shape `N X 4` where N
                is the number of bounding boxes and the bounding boxes are
                represented in the format `x1 y1 x2 y2`
    Out:
        corners: Numpy array of shape `N x 8` containing N bounding boxes each
                 described by their corner co-ordinates
                 `x1 y1 x2 y2 x3 y3 x4 y4`
    """
    width   = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height  = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)

    x1      = bboxes[:,0].reshape(-1,1)
    y1      = bboxes[:,1].reshape(-1,1)

    x2      = x1 + width
    y2      = y1

    x3      = x1
    y3      = y1 + height

    x4      = bboxes[:,2].reshape(-1,1)
    y4      = bboxes[:,3].reshape(-1,1)

    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    return corners

def _rotate_box(corners, degrees, cx, cy, w, h):
    """ Rotate the bounding box.
    Args:
        corners : Numpy array of shape `N x 8` containing N bounding boxes
                  each described by their corner co-ordinates
                  `x1 y1 x2 y2 x3 y3 x4 y4`
        degrees : angle by which the image is to be rotated in degrees
        cx      : x coordinate of the center of image
                  (about which the box will be rotated)
        cy      : y coordinate of the center of image
                  (about which the box will be rotated)
        h       : height of the image
        w       : width of the image
    Out:
        calc    : Numpy array of shape `N x 8` containing N rotated bounding
                  boxes each described by their corner co-ordinates
                  `x1 y1 x2 y2 x3 y3 x4 y4`
    """
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1),
                                          dtype = type(corners[0][0]))))

    M       = cv2.getRotationMatrix2D((cx, cy), degrees, 1.0)

    cos     = np.abs(M[0, 0])
    sin     = np.abs(M[0, 1])

    nW      = int((h * sin) + (w * cos))
    nH      = int((h * cos) + (w * sin))

    # Prepare the vector to be transformed
    calc    = np.dot(M,corners.T).T
    calc    = calc.reshape(-1,8)
    return calc

def _get_enclosing_box(corners):
    """ Get an enclosing box for rotated corners of a bounding box
    Args:
        corners : Numpy array of shape `N x 8` containing N bounding boxes
                  each described by their corner co-ordinates
                  `x1 y1 x2 y2 x3 y3 x4 y4`
    Out:
        final   : Numpy array containing enclosing bounding boxes of shape
                  `N X 4` where N is the number of bounding boxes and the
                  bounding boxes are represented in the format `x1 y1 x2 y2`
    """
    x_      = corners[:,[0,2,4,6]]
    y_      = corners[:,[1,3,5,7]]

    xmin    = np.min(x_,1).reshape(-1,1)
    ymin    = np.min(y_,1).reshape(-1,1)
    xmax    = np.max(x_,1).reshape(-1,1)
    ymax    = np.max(y_,1).reshape(-1,1)

    final   = np.hstack((xmin, ymin, xmax, ymax))
    return final

def update_rotate_bbox(bboxes, degrees, cx, cy, w, h):
    """ Rotate bbox points
    Args:
        bboxes  : [x_min, y_min, x_max, y_max] (4)
        degrees : angle in degrees
        cx      : x coordinate of the center of image
                  (about which the box will be rotated)
        cy      : y coordinate of the center of image
                  (about which the box will be rotated)
        h       : height of the image
        w       : width of the image
    Out:
        new_bbox: Bbox moved to new location
    """
    corners         = _get_corners(bboxes)
    corners         = np.hstack((corners, bboxes[:, 4:]))
    corners[:, :8]  = _rotate_box(corners[:, :8], degrees, cx, cy, w, h)
    new_bbox        = _get_enclosing_box(corners)
    return new_bbox