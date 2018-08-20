import cv2
img1 = cv2.imread('/home/yingkai/Data/newdata/train/item68.jpg')          # queryImage
img2 = cv2.imread('/home/yingkai/Data/newdata/train/WIN_20180803_01_46_09_Pro.jpg') # trainImage

#%% BF+ORB method
def BF_ORB():
  # Initiate ORB detector
  orb = cv2.ORB_create()

  # find the keypoints and descriptors with SIFT
  kp1, des1 = orb.detectAndCompute(img1,None)
  kp2, des2 = orb.detectAndCompute(img2,None)

   # create BFMatcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # Match descriptors.
  matches = bf.match(des1,des2)

  # Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)

  # Draw first 10 matches.
  img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[0:10], None,flags=2)

  cv2.imshow('result.jpg',img3)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.waitKey(1)
  cv2.waitKey(1)
  cv2.waitKey(1)
  cv2.waitKey(1)  

#%%BF+SIFT
def BF_SIFT():

   # Initiate SIFT detector
  sift = cv2.xfeatures2d.SIFT_create()
  # find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(img1,None)
  kp2, des2 = sift.detectAndCompute(img2,None)
  # BFMatcher with default params
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2, k=2)
  # Apply ratio test
  good = []
  for m,n in matches:
    if m.distance < 0.75*n.distance:
      good.append([m])
  # cv2.drawMatchesKnn expects list of lists as matches.
  img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:10],None, flags=2)

  cv2.imshow('result.jpg',img3)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.waitKey(1)
  cv2.waitKey(1)
  cv2.waitKey(1)
  cv2.waitKey(1)  

def FLANN():
  '''
  FLANN stands for Fast Library for Approximate Nearest Neighbors. 
  It contains a collection of algorithms optimized for fast nearest 
  neighbor search in large datasets and for high dimensional features. 
  It works more faster than BFMatcher for large datasets. 
  We will see the second example with FLANN based matcher.

  For FLANN based matcher, we need to pass two dictionaries which specifies 
  the algorithm to be used, its related parameters etc. First one is IndexParams. 
  For various algorithms, the information to be passed is explained 
  in FLANN docs. As a summary, for algorithms like SIFT, SURF etc. 
  you can pass following: 

     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

  While using ORB, you can pass the following. The commented values 
  are recommended as per the docs, but it didn't provide required results
  in some cases. Other values worked fine:
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2
  Second dictionary is the SearchParams. It specifies the number of 
  times the trees in the index should be recursively traversed. 
  Higher values gives better precision, but also takes more time.
  If you want to change the value, pass search_params = dict(checks=100).


  '''
  sift = cv2.xfeatures2d.SIFT_create()
  # find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(img1,None)
  kp2, des2 = sift.detectAndCompute(img2,None)

  # FLANN parameters
  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks=50)   # or pass empty dictionary

  flann = cv2.FlannBasedMatcher(index_params,search_params)

  matches = flann.knnMatch(des1,des2,k=2)

  # Need to draw only good matches, so create a mask
  matchesMask = [[0,0] for i in range(len(matches))]

  # ratio test as per Lowe's paper
  for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

  draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                  matchesMask = matchesMask,
                  flags = 0)

  img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
  cv2.imshow('result.jpg',img3)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.waitKey(1)
  cv2.waitKey(1)
  cv2.waitKey(1)
  cv2.waitKey(1)  

#%%   
if __name__ == '__main__':
  FLANN()
