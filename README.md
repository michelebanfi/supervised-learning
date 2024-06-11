Upload updates
```bash
git status
git add -A
git commit -m "messagge"
git push
```

Download updates
> NOTE! first check to have a clean working tree using ``git status``
```bash
git pull
```

# Road map:  
- [x] Data preprocessing: Process the images by resizing them to a common size,
converting them to grayscale or RGB, and normalizing the pixel values. 
- [x] Create the validation test from the train test. 
- [x] Convolutional neural network: train a convolutional neural network on the preprocessed 
images to classify them into different categories.

- [x] Cast the problem as self-supervised learning 
- [x] Model evaluation: evaluate the performance of the models using metrics such
as accuracy, precision, recall, and F1-score. 
- [x] Hyperparameter Tuning: fine-tune the hyperparameters of the model 
to achieve a better performance. 