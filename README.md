# MonReader

**Background:**

Our company develops innovative Artificial Intelligence and Computer Vision solutions that revolutionize industries.

- <ins>Machines that can see:</ins> We pack our solutions in small yet intelligent devices that can be easily integrated to your existing data flow.
- <ins>Computer vision for everyone:</ins> Our devices can recognize faces, estimate age and gender, classify clothing types and colors, identify everyday objects and detect motion. 
- <ins>Technical consultancy:</ins> We help you identify use cases of artificial intelligence and computer vision in your industry. Artificial intelligence is the technology of today, not the future.

MonReader is a new mobile document digitization experience for the blind, for researchers and for everyone else in need for fully automatic, highly fast and high-quality document scanning in bulk. It is composed of a mobile app and all the user needs to do is flip pages and everything is handled by MonReader: it detects page flips from low-resolution camera preview and takes a high-resolution picture of the document, recognizing its corners and crops it accordingly, and it dewarps the cropped document to obtain a bird's eye view, sharpens the contrast between the text and the background and finally recognizes the text with formatting kept intact, being further corrected by MonReader's ML powered redactor.

![Alt text](P4_MonReader/project_images/project_10_1.jpg?raw=true "Image2")

![Alt text](P4_MonReader/project_images/project_10_2.jpg?raw=true "Image2")

**Data Description:**

We collected page flipping video from smart phones and labelled them as flipping and not flipping.

We clipped the videos as short videos and labelled them as flipping or not flipping. The extracted frames are then saved to disk in a sequential order with the following naming structure: VideoID_FrameNumber

**Download Data:**

https://drive.google.com/file/d/1KDQBTbo5deKGCdVV_xIujscn5ImxW4dm/view?usp=sharing

**Goal(s):**

Predict if the page is being flipped using a single image.

**Success Metrics:**

Evaluate model performance based on F1 score, the higher the better.

**Bonus(es):**

Predict if a given sequence of images contains an action of flipping.

**Notebook user guide:**