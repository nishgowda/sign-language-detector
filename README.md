# Sign Language Detector

Classify sign language letters using a Convolutional Neural Network made in PyTorch and uses OpenCV for real time camera applications. The jupyter notebook for this project can also be found in this repository.

## Training
First download the MNSIT sign language training and testing datasets and place it in the data/input directory. Run ```train.py``` along with the name for the model that will be saved.

```
$ python3 train.py sign_language.pt
```

Afterwards, your model should be saved directly in the *models* directory.

## Real time applications
Once your model is saved, you can run it with your video camera turned on to try it out. Run ```detect.py``` along with the name of the model you saved previously.

```
$ python3 detect.py sign_language.pt
```
A box will appear in your video feed which is where you should be directing your hand. Below, it should show the letter predicted and the probability of it as well.
