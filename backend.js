const express = require("express");
const multer = require("multer");
const { exec } = require("child_process");
const fs = require("fs");


const app = express();
const cors = require("cors");
app.use(cors());


app.use(express.json());
app.use(cors({
  allowedHeaders: ['Content-Type', 'X-XSRF-TOKEN'],
  exposedHeaders: ['Content-Disposition']
}));
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  console.log(`Receive request: ${req.method} ${req.url}`);
  next();
});

const execPromise = (command) => {
  return new Promise((resolve, reject) => {
    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error executing ${command}:`, stderr);
        reject(stderr);
      } else {
        console.log(`Success executing ${command}:\n${stdout}`);
        resolve(stdout);
      }
    });
  });
};



const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, './');
  },
  filename: (req, file, cb) => {
    cb(null, '_test.mp4'); 
  },
});

const upload = multer({ storage });

app.options('/upload', (req, res) => {
  res.header('Access-Control-Allow-Methods', 'POST');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  res.status(204).send();
});

app.post("/upload", upload.single("video"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: "Uploading file non-exist." });
  }

  const videoPath = req.file.path;
  console.log(`The video file is stored in: ${videoPath}`);

  try {
  
    // await execPromise(`python with_pose_3.py`);
          await execPromise(`python with_pose_2.py`);

    const output1 = await fs.promises.readFile("step_1.json", "utf8");
    await execPromise(`python description_refine.py`);
    await execPromise(`python summary_1.py step_2.json step_3.txt`);

    res.json({ message: "Success in processing video." });

  } catch (err) {
    console.error("Failure in processing pipeline:", err);
    res.status(500).json({ message: "Failure in processing video pipeline." });
  }
});


app.post("/ask", (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ message: "The question cannot be empty" });
  }

  console.log(`Receive user question: ${question}`);
  exec(`python answer_api.py step_3.txt --question="${question}"`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Failure in executing answer_2.py : ${stderr}`);
      return res.status(500).json({ message: "failure in processing the question." });
    }
    try {
      const result = JSON.parse(stdout);
      console.log(result.answer)
      res.json(result.answer); 
    } catch (e) {
      console.error("Parse Failure：", stdout);
      res.status(500).json({ message: "The exception of return value from python", raw: stdout });
    }


  });
})
app.listen(8080, '0.0.0.0', () => {
  console.log("The server is running on cloud platform 8080");
  // console.log("The server is running on localhost:3000");
});
