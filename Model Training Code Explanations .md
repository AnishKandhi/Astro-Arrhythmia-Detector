**Training Pipeline**

1. The 3 parallel arrays (beat waveforms, frequency spectrums, labels) created and saved in the .npz files by the Digital Signal Processing pipeline 
are loaded into the script.
2. The arrays are converted into PyTorch tensors and placed in a Torch dataset.
3. The dataset is loaded into a DataLoader with a specified batch size, which holds the data for the model to be trained on it.
4. The CNN is trained on the data using CrossEntropyLoss and Adam optimizer.
   
<img width="689" height="422" alt="image" src="https://github.com/user-attachments/assets/09579263-4459-4755-9be9-26fcdfb8686d" />

The snippet above loads the training data parallel arrays, converts them into PyTorch-ready tensors, unsqueezes (unzipping) them, and creates a PyTorch-ready dataset out of them.

<img width="757" height="88" alt="image" src="https://github.com/user-attachments/assets/6ec10a28-c93f-490c-9f2a-d8a3f54c45e5" />

The torch dataset is then loaded into the DataLoader, which will be traversed through during training. The batch size was set to 64 to force the CNN to train
on 64 beats at a time rather than 1 by 1, to maximize efficiency.

<img width="523" height="756" alt="image" src="https://github.com/user-attachments/assets/b59c40da-7d98-4c33-9c90-60cf0025e87f" />


**CNN Architecture**

The CNN is broken into 3 parts:
1. The time branch detects patterns and features in the EKG time-domain voltage waveform.
2. The frequency branch detects features and patterns in the 1D frequency spectrum processed by Fast Fourier Transform (FFT)
3. The classifier branch acts as the decision layer, taking in the results of both the time and frequency branches combined and predicting the arrhythmia class (0 - 3).

<img width="523" height="756" alt="image" src="https://github.com/user-attachments/assets/36caff70-da22-40c3-80c8-977cf4c29271" />

The time branch and frequency (fft) branch have the same structure, with 2 1D convolution layers, followed by a series of ReLU layers to add non-linearity, 
BatchNorm layers for normalization to smooth out feature extraction, and MaxPool layers to condense feature maps at each step. The final produced feature maps
of both layers are flattened  so they can be pooled together into a single tensor for the classifier.
The classifier branch contains 2 linear regression layers, with the first layer processing the combined time and frequency feature inputs using 64 neurons, and the second
taking in the 64 outputs from the first layer, and producing scores for each arrhythmia class (the prediction).
In summary, the outputs of the time and frequency branches are concatenated into a single tensor, and that tensor is passed into the classifer for a prediction.

**Training Segment**

<img width="632" height="308" alt="image" src="https://github.com/user-attachments/assets/a1e5d4ac-e8ef-4a63-b57a-3ee1f8892d82" />

I decided to use CrossEntropyLoss to ensure the loss was calculated based on the probabilities predicted rather than the label itself, to ensure precision
for backpropogation. I then traversed through each input (time and frequency tensors) and output (label tensor) in the DataLoader. For each epoch, 
I fetched the model's output, grabbed the loss (error), performed gradient descent to calculate how much each weight should be updated (_loss.backward()_), and updated
the weights (_optimizer.step()_).
