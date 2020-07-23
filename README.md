The following requirements have been confirmed in the experimental environment, and the same version as ours can guarantee operation.

Before running, you need to confirm python==3.7. For users with GPU, you need to ensure that the CUDA version is CUDA 10, and other basic dependencies are included in the requirements

Use pip install -r requirements.txt to install the basic package

The path in the project needs to be modified according to the experimental environment

Detection:

First run create_STimages.py to generate spatiotemporal feature images of the video

Run detection.py to detect the generated STimages image collection, and the detection results will be generated in TrainingData. We give a model we trained. This model has a good effect on Bactrocera citrus. You need to ensure that the keras version is 2.2. 5 can be used directly

Run check.py and select the generated test results for manual error correction

Training:

First run create_STimages.py to generate spatiotemporal feature images of the video

Categorize and save spatiotemporal features in several folders in TrainingData. The name of the behavior can be named by yourself

Run rename_all.py and create_label.py to make TrainingSet

Run training.py to train the model

The above steps only need to modify the path, and the other steps are performed automatically by the program.

