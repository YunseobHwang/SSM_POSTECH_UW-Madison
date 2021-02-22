# IIoT- and AI-based Smart Substainable Manufacturing

- My part: Development of AI-based fault diagnosis system in cold forging process

## Outline

<p align="center">
     <b> CNN-based fault diagnosis in cold forging process</b> <br>
     <img alt="Outline" src="./images/outline.png"
          width=80% />
</p>

## Experimental Data Collection

- Collaborated with Semblex Co. in Chicago, Illinois, US
- Data was acquisited under expermental condition
- 2 Acceleometer Sensors (Sampling Rate: 1650 Hz / One near the die side where forging happens, another one close to the material side where a wire is fed)
- 6 fault scenarioes (Heavy Oil, Die Punch, Scrapped Wire, Die Chip, Die Internal, and Pin)

<p align="center">
     <b> Examples of Fault Cases </b> <br>
     <img alt="Defect" src="./images/defect_cases.png"
          width=80% />
</p>

## Preprocessing

<p align="center">
     <b> Clipping</b> <br>
     <img alt="PRE1" src="./images/preprocessing1.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> Segmentation</b> <br>
     <img alt="PRE2" src="./images/preprocessing2.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> Wavelet Transform</b> <br>
     <img alt="PRE3" src="./images/preprocessing3.png"
          width=80% />
</p>

## 2D Convolutional Neural Network

<p align="center">
     <b> CNN architecture</b> <br>
     <img alt="CNN" src="./images/CNN_architecture.png"
          width=80% />
</p>

## Studies

### Overlapping

<p align="center">
     <b> 10 % Overlap and Non-overlap </b> <br>
     <img alt="SEG" src="./images/segmentations.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> Confusion Matrices according to overlapping</b> <br>
     <img alt="CM" src="./images/cm_comparison.png"
          width=80% />
</p>

### Sample Length

### Number of Training Samples
