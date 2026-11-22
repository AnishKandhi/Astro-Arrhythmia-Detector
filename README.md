



**Digital Signal Processing Pipeline**
1. The wfdb (Waveform Database) Python library was used to fetch the patient record from the MIT BIH dataset. 
2. The raw heart rhythm (ECG) waveform showing voltage over time was extracted from the record. 
3. The cardiologist's annotations showing the symbol for each beat and information about the patient (gender, age) was also extracted. 
4. The ECG waveform was cleaned using a bandpass filter selecting for frequencies from .5 Hz to 40 Hz to filter out noise from muscle contractions, electromagnetic interference, and movement. 
5. The beats were extracted from the cleaned waveform by using the cardiologist's annotations noting the peak indices (which sample was a peak), and reserving a 300 sample window around the peak. 
6. The beat was normalized according to its mean and standard deviation. Fast Fourier Transform was performed on the beat to convert it into a 1D frequency spectrum. 
7. The label for the beat (0-3) was determined by mapping the symbol for the beat to an arrhythmia class. 
8. The beat waveform, frequency spectrum, and corresponding label were placed into parallel arrays, and the arrays for all the records were pooled together into a final dataset of 3 master arrays.


**Training Pipeline**
1. The 3 parallel arrays (beat waveforms, frequency spectrums, labels) created and saved in the .npz files by the Digital Signal Processing pipeline are loaded into the script.
2. The arrays are converted into PyTorch tensors and placed in a Torch dataset.
3. The dataset is loaded into a DataLoader with a specified batch size, which holds the data for the model to be trained on it.
4. The CNN is trained on the data using CrossEntropyLoss and Adam optimizer.
