**Digital Signal Processing Pipeline**

1. The wfdb (Waveform Database) Python library was used to fetch the patient record from the MIT BIH dataset.
2. The raw heart rhythm (ECG) waveform showing voltage over time was extracted from the record.
3. The cardiologist's annotations showing the symbol for each beat and information about the patient (gender, age) was also extracted.
4. The ECG waveform was cleaned using a bandpass filter selecting for frequencies from .5 Hz to 40 Hz to filter out noise from muscle contractions, electromagnetic interference, and movement.
5. The beats were extracted from the cleaned waveform by using the cardiologist's annotations noting the peak indices (which sample was a peak), and reserving a 300 sample window around the peak.
6. The beat was normalized according to its mean and standard deviation.
7. Fast Fourier Transform was performed on the beat to convert it into a 1D frequency spectrum.
8. The label for the beat (0-3) was determined by mapping the symbol for the beat to an arrhythmia class.
9. The beat waveform, frequency spectrum, and corresponding label were placed into parallel arrays, and the arrays for all the records were pooled together into a final dataset of 3 master arrays.


**Bandpass Filter**

<img width="505" height="105" alt="image" src="https://github.com/user-attachments/assets/fb92a43b-a49e-496d-b350-6a22c7ddd99e" />

The function performs the bandpass filter. The butter() function extracts the coefficients needed for a 4th order filter selecting frequencies from .5 Hz to 40 Hz
using a sampling frequency of 360 Hz. The coefficients a and b are applied to filter the raw heart signal using the filtfilt() function.

**Record Retrieval**

<img width="454" height="153" alt="image" src="https://github.com/user-attachments/assets/46595a49-046a-4283-b4f4-8508b45c2956" />

The record and cardiologist's annotations are fetched using the wfdb library in the _get_record()_ function. The raw signal is retrieved using the p_signal property
in the record. The bandpass() function I created earlier was used to filter it according to the record's specific sampling frequency. 


**Beat Extraction**

<img width="622" height="333" alt="image" src="https://github.com/user-attachments/assets/665084ea-8bc1-4aaa-93be-7a0e3e8d0d2b" />

I reserved a 300 beat window for each beat in the waveform by identifying where the peaks were using the record's sample property (which gives index of each peak),
and slicing the signal to section off a sub-array (the beat) containing 150 beats over and after the peak. The beat was normalized via their z-scores,
where I calculated how many standard deviations each sample was away from the mean of the beat's distribution. 

**Frequency-Domain Conversion**

<img width="450" height="66" alt="image" src="https://github.com/user-attachments/assets/5e29d405-6179-47e6-adce-2f4e738179e5" />

I used NumPy's Fast Fourier Transform function and the Hann window from earlier to convert the beat from the time-domain into a frequency vs. power spectrum. In this way,
the model can analyze the frequency distribution of the beat and calibrate the width and height of the QRS segment (essential for detecting arrhythmia), instead of being limited to a voltage over time
graph.

**Data Loading**

<img width="263" height="94" alt="image" src="https://github.com/user-attachments/assets/b9853c68-b0fb-4a5e-92af-0b29afec0ed1" />

The beat waveform, its frequency spectrum, and its label were added to the 3 parallel arrays that will later be dumped into the final dataset. The label was extracted by mapping
the symbol from the annotation using a Dictionary.
<img width="386" height="141" alt="image" src="https://github.com/user-attachments/assets/ee1e8b38-6b68-476e-94af-5cc0b3ca4a97" />

