##  Code File Descriptions

- **with_pose_3.py**  
  It extracts features with YOLO

- **description_refine.py**  
  It formats data of extracted features

- **summary_1.py**  
  It summarizes content from refined description
  
- **answer_api.py**
  
  It uses the LLM to answer question of users based on the summarized content. e.g. Is a person in the video lying?
  
- **index.html**
  
  UI of the BrainyCam. User can upload the video, ask question and get answer in this UI
  
- **backend.js**

  The pipeline of executing the with_pose_3.py, description_refine.py, summary_1.py and answer_api.py.

## Run Instructions

Please run the code file in the following order：

1.   node backend.js  (with node.js)
   
2.  Host the index.html file as a static page

   Then you can upload the video on the index.html (UI) and ask questions

