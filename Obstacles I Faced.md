**Obstacle Encountered: Class Imbalance**

One major issue I ran into was that the MIT BIH database was severely class-imbalanced, with substantially more "normal" beats (class 0) than the 3
arryhtmias. This would cause overfitting because the model would learn that constantly predicting class 0 would yield a high accuracy.
Therefore, I made sure to cap the number of normal beats from each record that I allowed into the final dataset at 350 beats.

<img width="637" height="368" alt="image" src="https://github.com/user-attachments/assets/22cd5120-bcb5-4083-96c2-e973117abd3e" />

The code above goes through each record, checks where the label is equal to 0 (meaning a normal beat), and randomly selects only 350 of them. Those 350 normal beats,
alongside all the arrhythmia beats (which don't need to be capped), were pooled together and extended into the final dataset.

<img width="668" height="259" alt="image" src="https://github.com/user-attachments/assets/c62b439b-fd52-4b87-94e9-6d0ad00ab11c" />

Initially, there were over 12000+ class 0 beats in the training dataset. After capping each record, the new distribution was much more even (see below).

<img width="535" height="258" alt="image" src="https://github.com/user-attachments/assets/6a479cb4-df92-43bb-9742-abd68d7c6ea3" />

I chose this method instead of using weighted-loss entropy because it helped ensure that my dataset had even representation from all records' normal beats instead of
just 1 record's normal beats.

**Obstacle Encountered: Natural Variation in Baseline Rhythms**
Because everyone has a different baseline heart rhythm depending on age, gender, and fitness levels, a one-size-fits-all model would not work well. That is why I used calibration
to fine-tune my CNN's decision layer before locking in a prediction (see benchmark.py & explanations for more information).
To prevent data leakage, I made sure that the arrhythmias and normal beats I selected were independent of the test records. I also made sure to split the
test records among males, females, and ages equally for rigorous model generalizability evaluation.

Results before selecting sample arrhythmias during fine-tuning:

<img width="1113" height="135" alt="image" src="https://github.com/user-attachments/assets/cf92a1b0-cfcf-4744-b3b1-016c9bf4d08c" />

Results after selecting sample arrhythmias during fine-tuning:

<img width="1431" height="261" alt="image" src="https://github.com/user-attachments/assets/7e92eb0d-7987-451c-b22d-9c33a1c15f6d" />






