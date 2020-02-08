# FG-2020 Competition: Affective Behavior Analysis in-the-wild (ABAW)
 As we exceed upon the procedures for modelling the different aspects of behaviour,Expression recognition has become a key field of research in Human Computer Interactions. Expression recognition in the wild is a very interesting problem and is challenging as it involves  detailed feature extraction and heavy computation. we make an attempt to recognize different expressions i.e.,Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise on extended Affect Wild Net (AffWid2) database. AffWild2 database consists of videos in the wild labelled for seven different expressions at frame level. We used a bi-modal approach by fusing audio and visual features and train a sequence-to-sequence model that is based on Gated Recurrent Units (GRU) and Long Short Term Memory (LSTM) network. We show experimental results on validation data  

# Our experimental results are given below:
BaseLine Model: 37 % </br>
Audio Model(GRU layers): 39%</br>
Video Model(OpenFace) : 39.5%</br>
Multi-Modal(GRU) : 41.5</br>
