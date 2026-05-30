# Advanced NLP Exercise 1: Fine Tuning

This is the codebase for Exercise 1 of the Advanced NLP HUJI course (67664), fine tuning pretrained models to perform sequence classification (paraphrase detection) on the MRPC dataset from the GLUE benchmark.

# Install
```bash
pip install -r requirements.txt```

# Fine-Tune and Predict on Test Set
Run:

``` python ex1.py --do_train --num_train_epochs <number of training epochs> --batch_size <batch size> --lr <learning rate> ```

If you use --do_predict, a prediction.txt file will be generated, containing prediction results for all test samples.
