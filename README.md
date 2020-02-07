# Afew_Expressions
Continuous affect prediction in the wild is a very interesting problem and is challenging as continuous prediction involves heavy computation. We make an attempt to predict continuous emotions at frame level. AffWild2 database consists of videos in the wild labelled for valence and arousal at frame level. It also consists of annotations for seven discrete expressions such as neutral, anger, disgust, fear, happiness, sadness and surprise. We used a bi-modal approach by fusing audio and visual features and train a sequence-to-sequence model that is based on Gated Recurrent Units (GRU) and Long Short Term Memory (LSTM) network. We show experimental results on validation data.

# Our experimental results are given below:
BaseLine Model: 37 % </br>
Audio Model(GRU layers): 39%</br>
Video Model(OpenFace) : 39.5%</br>
Multi-Modal(GRU) : 41.5</br>
