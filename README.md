Clone the Final Project Template
$ cd ${CFU_ROOT}/proj
$ git clone https://github.com/zakochi/AAML-2025-Project.git
Prepare the Model File
Download the original Wav2Letter tflite model:

$ cd AAML-2025-Project/src/wav2letter/model
$ wget https://github.com/ARM-software/ML-Zoo/raw/master/models/speech_recognition/wav2letter/tflite_pruned_int8/wav2letter_pruned_int8.tflite
Then convert the tflite file into a header file:

$ chmod +x model_convert.sh
$ ./model_convert.sh
