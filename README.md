# Vehicle Tracking and Object Detection<br>
This project provides a script that performs real-time object detection and vehicle tracking using OpenCV and a pre-trained YOLO model. The script detects objects in a video stream, draws bounding boxes around them, and utilizes the centroid tracking algorithm to track vehicles.<br><br>


# Installation<br>
Clone the repository:<br>
git clone https://github.com/shivam-gupta0/Object-Tracking.git<br><br>
Install the required dependencies:<br>
pip install opencv-python numpy<br><br>
Usage <br>
Specify the paths to the configuration file (cfg_path) and the custom weights file (weights_path) in the script.<br>
Update the path to the classes file (classes.txt) in the script.<br><br>
Run the script:<br>
python vehicle_tracking.py<br><br>
The script will open a video stream and perform object detection on the frames. Detected vehicles will be surrounded by bounding boxes, and the corresponding labels and confidence scores will be displayed.<br>

The script will track the vehicles using the centroid tracking algorithm, assigning each vehicle a unique ID and updating their positions over time. The IDs and current positions of the vehicles will be displayed on the video stream.<br>

Press 'q' to exit the script.<br>


# Process 
Perform object detection on frames of input video <br />
Extract bounding box coordinates of detceted objects <br />
Assign unique ID to each bounding box of first frame <br />
Maintain ID relation with objects as they move in frames of video <br />
Assign new ID to the new detcted objects <br /> 

# Contributing<br>
Contributions are welcome! If you have any improvements or new features to add, please follow these steps:<br>

Fork the repository.<br>

Create a new branch:<br>

git checkout -b feature/your-feature-name<br>
Make your changes and commit them:<br>

git commit -m "Add your commit message"<br>
Push the changes to your branch:<br>

git push origin feature/your-feature-name<br>
Open a pull request, and describe the changes you made.<br>

# Method
The centroid tracking algorithm <br />  
![2022-07-06 (1)](https://user-images.githubusercontent.com/85798077/177437097-67af85d9-fa05-4671-875b-cc7890d2209c.png)
![2022-07-06 (2)](https://user-images.githubusercontent.com/85798077/177437123-f15c244f-d9b2-45ac-9b58-4d7e4f76cf40.png)
