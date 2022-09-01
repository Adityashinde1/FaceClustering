## 🆕 Unsupervised Face Clustering Pipeline
Live face recognition is still a challenge for automated security divisions. It has been established that with the improvements in convolutional neural networks, particularly the innovative methods of region-CNN, we can use supervised learning solutions like FaceNet, YOLO for quick and live face recognition in a real-world setting.
It is still a laborious process to gather datasets of our target labels before we can train a supervised model. For dataset generation with minimal user labelling work, we require an effective and automated approach.

## 💽 Data
For this project a short news reporting video clip is been used.

## 📚 Approach
We are putting forth a dataset creation pipeline that uses a video clip as its source, removes all the faces, and accurately groups them into sets of photos that each represent a unique individual. Each set can easily be labeled by human input with ease. 
For the purpose of extracting frames per second from the input video clip, we will use the opencv library. One second appears plenty for processing time and covering pertinent info. To extract the faces from the frames and align them for feature extraction, we will utilise the face recognition package (supported by dlib).
The human observable features will then be extracted and clustered using scikit-DBSCAN learn's clustering.

## 🧑‍💻 How to setup
create fresh conda environment
```python
conda create -p ./env python=3.6 -y
```
activate conda environment
```python
conda activate ./env
```
Install requirements
```python
pip install -r requirements.txt
```
To run main file
```python
python main.py
```

## 🧑‍💻 Tech used
1. OpenCV
2. Face_recognition library
3. Clustering

## 🏭 Industrial Use-case
1. It create labels and group them in folders for users to adapt them as a dataset for their training use-cases.

## 👋 Conclusion
Collecting a single person's data and manually adding it to that person's image directory is exceedingly difficult. This method takes a lot of time. Instead, we might choose to use an unsupervised face clustering method to quickly create several directories for various people.

