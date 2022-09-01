## Unsupervised Face Clustering Pipeline
Live face recognition is still a challenge for automated security divisions. It has been established that with the improvements in convolutional neural networks, particularly the innovative methods of region-CNN, we can use supervised learning solutions like FaceNet, YOLO for quick and live face recognition in a real-world setting.
It is still a laborious process to gather datasets of our target labels before we can train a supervised model. For dataset generation with minimal user labelling work, we require an effective and automated approach.

## Data
For this project a short news repoting video clip is been used.

## Approach
We are putting forth a dataset creation pipeline that uses a video clip as its source, removes all the faces, and accurately groups them into sets of photos that each represent a unique individual. Each set can easily be labeled by human input with ease. 
For the purpose of extracting frames per second from the input video clip, we will use the opencv library. One second appears plenty for processing time and covering pertinent info. To extract the faces from the frames and align them for feature extraction, we will utilise the face recognition package (supported by dlib).
The human observable features will then be extracted and clustered using scikit-DBSCAN learn's clustering.

## How to setup
```python
conda create -p ./env python=3.8 -y
```

## Tech used

## Industrial Use-case

## Conclusion