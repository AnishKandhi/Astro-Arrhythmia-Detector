**Astronaut Arrhythmia Detector**

WARNING: THIS IS A PROTOTYPE; DO NOT USE FOR MEDICAL ADVICE

**Problem Overview**
Astronauts in deep space face conditions such as microgravity, which can cause fluid to accumulate in their hearts, changing their size and disrupting normal rhythm. In addition, conditions like radiation, dehydration, and physical stress during extravehicular missions can cause irregular heart rhythms (arrhythmia) that can be mistaken for life-threatening arrhythmias or can become life-threatening if not managed promptly. The issue is that astronauts are frequently monitored by cardiologists on Earth, and communication delays from space can be up to 20 minutes, making instant treatment impossible. Additionally, many arrhythmia detectors today use one-size-fits-all models that use the same rules and patterns to predict arrhythmias across different astronauts. However, different astronauts have different resting heart rhythms and ECG baselines. 

Therefore, I trained a machine learning model on a variety of heart rhythms across different genders and ages in the MIT BIH dataset. I trained a CNN
on both the voltage waveform for each beat as well as its frequency spectrum from Fourier Transform. The model predicts the arrhythmia class of each beat recorded
from 30 minutes of ECG data for each astronaut from 0 to 3 (Class 0: Normal, Class 1: Supraventricular Ectopic Beat, Class 2: Ventricular Ectopic Beat, Class 3: Conduction/Fusion Anomaly).

After training, the CNN was calibrated to each individual's resting heart rhythm baseline in order to ensure predictions were personalized instead of general. To do this, I fine-tuned the trained model's classifier head (decision layer) on 30 randomly selected normal beats from the patient and 125 randomly selected arrhythmias from the training dataset.


**Results**

The calibrated CNN's final overall accuracy across the 6 test records was **83.08%**, a **17%** step up from the uncalibrated model. The CNN also maintained an arrhythmia sensitivity of **76.77%** and a low false positive rate of **15.4%**, which is a **15%** decrease from the uncalibrated model.

<img width="1116" height="141" alt="image" src="https://github.com/user-attachments/assets/491a6183-2159-4d9f-b5ab-00bd17e39f21" />

**Digital Signal Processing Pipeline**
1. The wfdb (Waveform Database) Python library was used to fetch the patient record from the MIT-BIH dataset. 
2. The raw heart rhythm (ECG) waveform showing voltage over time was extracted from the record. 
3. The cardiologist's annotations showing the symbol for each beat and information about the patient (gender, age) were also extracted. 
4. The ECG waveform was cleaned using a bandpass filter selecting for frequencies from .5 Hz to 40 Hz to filter out noise from muscle contractions, electromagnetic interference, and movement. 
5. The beats were extracted from the cleaned waveform using the cardiologist's annotations, noting the peak indices (which sample was a peak), and reserving a 300-sample window around the peak. 
6. The beat was normalized according to its mean and standard deviation. Fast Fourier Transform was performed on the beat to convert it into a 1D frequency spectrum. 
7. The label for the beat (0-3) was determined by mapping the symbol for the beat to an arrhythmia class. 
8. The beat waveform, frequency spectrum, and corresponding label were placed into parallel arrays, and the arrays for all the records were pooled together into a final dataset of 3 master arrays.


**Training Pipeline**
1. The 3 parallel arrays (beat waveforms, frequency spectrums, labels) created and saved in the .npz files by the Digital Signal Processing pipeline are loaded into the script.
2. The arrays are converted into PyTorch tensors and placed in a Torch dataset.
3. The dataset is loaded into a DataLoader with a specified batch size, which holds the data for the model to be trained on it.
4. The CNN is trained on the data using CrossEntropyLoss and the Adam optimizer.


**Obstacles & Challenges**
1. **Class Imbalance** - The MIT-BIH dataset was heavily skewed towards normal (class 0) beats, so my original dataset had over 12000+ beats for class 0 but only around 2000 each arrhythmia class. Therefore, I capped the number of normal beats I selected from each record at 350 to be in the final training dataset. This evened out the distribution.

<img width="516" height="260" alt="image" src="https://github.com/user-attachments/assets/3c8f5da3-dd82-4db2-a91a-f90590214d25" />

2. **Variation in Baselines** - Because every astronaut has slightly different resting ECG baselines, I made sure to use transfer learning to tune the CNN's classifier head (decision layer) so it can learn what the baseline for that specific astronaut looked like before making predictions. I made sure to freeze the weights of its
feature extraction layers to prevent overfitting on the test data during fine-tuning.

4. **Data Leakage** - To prevent data leakage, I made sure that the arrhythmias I selected to fine-tune the classifier head were not part of the test records and were
only in the training data purely to remind the model what an arrhythmia looked like. This ensures the model's performance on its test data was not influenced by
its exposure to the beats beforehand.
