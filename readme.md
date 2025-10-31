这是一个基于AI的视频智能分析系统，结合YOLO和LLM，对常见监控视频实现特征提取，支持用户对于视频内容进行提问，并回答目标出现的时间与地点等信息，帮助用户节省视频查询时间。

代码文件说明

with_pose_3.py
使用 YOLO 提取视频特征。

description_refine.py
对提取的特征数据进行格式化处理。

summary_1.py
对格式化后的描述内容进行总结。

answer_api.py
基于总结后的内容，使用 LLM 回答用户问题，例如：“视频中有人在躺着吗？”

index.html
BrainyCam 的用户界面（UI），用户可以上传视频、提问并获取回答。

backend.js
执行 with_pose_3.py、description_refine.py、summary_1.py 和 answer_api.py 的完整处理流程。

运行说明

请按照以下顺序运行代码文件：

使用 Node.js 运行

node backend.js


将 index.html 文件托管为静态页面

然后可以在 UI 中上传视频并提出问题获取回答。

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

