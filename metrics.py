import cv2
import numpy as np

def meanIoU(pred, gt, threshold):
    '''meanIoU
    Description: Calculates the mean intersection over uniona for a 2 
                    dimensional numpy array.
    Args: pred - 2D numpy array of values from 0 to 1, the same dimensions as the ground
                    truth array.
          gt   - 2D numpy array of 0s and 1s.  The expected result to which
                    pred is compared.
          threshold - The value above which the values of pred are converted to 1, and
                    below which values are converted to 0.
    Return: 1D numpy array
    '''
    results = np.zeros(len(pred))
    for i in range(len(pred)): 
        thresh = ((pred[i] > threshold) * 1).astype(dtype=np.float32)
        union = cv2.bitwise_or(gt[i], thresh)
        intersection = cv2.bitwise_and(gt[i], thresh)
        results[i] = np.sum(intersection)/np.sum(union)
    return np.mean(results)

def meanIoU_thresholds(pred, gt):
    '''meanIoU_thresholds
    Description: Generates a list of mean IoU values for a range of threshold values.
    Args: pred - 2D numpy array of values from 0 to 1, the same dimensions as the ground
                    truth array.
          gt   - 2D numpy array of 0s and 1s.  The expected result to which
                    pred is compared.
    Return: List of mean IoU values
    '''
    meanIoU_by_thresh = []
    for i in range(100):
        meanIoU_by_thresh.append(meanIoU(np.squeeze(pred), 
                                        np.squeeze(gt), 
                                        (i/(100.0))))
    return meanIoU_by_thresh