**Calibration Pipeline**

One issue is that the model struggles to automatically adapt to different individuals resting EKG baselines across factors like age and gender. Therefore, I made sure to calibrate the model
to each person's resting baseline heart rhythm by randomly selecting 30 of those normal beats, and fine-tuning the model's decision (classification) layer on them. I also 
observed that the model's arrhythmia sensitivity decreased after fine-tuning, so I also randomly selected 75 arrhythmia beats to remind it what an arrhythmia looked like.
Both the randomly selected normal and arrhythmia beats were pooled together to tune the model's classifier head. This drastically improved arrhythmia sensitivity and accuracy.



<img width="817" height="527" alt="image" src="https://github.com/user-attachments/assets/e60d9948-4fac-4c87-8fa1-094d559ea0c1" />
The code above loads the training data, and selects 75 random arrhythmias in the label array _tr["y"]_ (where ever _tr["y"] > 0_, that means that beat is an arrythmia). 
<img width="1288" height="479" alt="image" src="https://github.com/user-attachments/assets/93fd635c-8769-4694-a5b2-0f8dadc05f2b" />
In the 3 concatenate lines, these arrhythmia beats were also pooled with the calibration normal beats (30) to create the fine-tuning dataset. The _requires_grad = false_ lines 
freeze the weights of the waveform and frequency domain layers. This is so that only the classifier head is tuned and adapted, as tuning all of the model's parameters would
force it to relearn patterns on the features of the test record, causing overfitting. 
<img width="982" height="81" alt="image" src="https://github.com/user-attachments/assets/305a332d-b99a-495e-8004-a9114378be1b" />
I ensured the optimizer took in the classifier branch's weights because they are the ones that need to be tuned via transfer learning to adapt predicictions to the individual's
resting rhythm baseline.

<img width="580" height="277" alt="image" src="https://github.com/user-attachments/assets/debd892a-035b-4d02-a548-ee4a402fcb31" />
Like the previous training script, this snippet fetches the model's output, calculates loss, and uses backpropogation to update the classification branch's weights for a set number 
of epcochs.

**Results**
<img width="1478" height="140" alt="image" src="https://github.com/user-attachments/assets/af5ee963-030a-4fec-9c31-3e4447be9195" />


