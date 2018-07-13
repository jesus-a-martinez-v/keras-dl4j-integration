# Keras + DL4J

Example of a simple integration between Keras (running with a TensorFlo backend) and Deeplearning4J. The model is developed, tested and saved using Keras (details here). Then, it's loaded in DeepLearning4J from the corresponding model and weights files that resulted from the Keras export process, and run.

# Installation

The easiest way to get started is to load this project in IntelliJ. You'll also need `conda`.

### Create conda environment

```
conda env create -f env.yml
source activate data
```

### Run Keras model

`cd src/main`
`python keras_model.py`

You should see how to model gets trained, and also two files should appear at the same level of the script: `iris_model_json` and `iris_model_save`.

### Load model in DL4J

Just press play in the Dl4JModel class ;)

# NOTE:

- As of July of 2018, DL4J does not support Keras 2 export API. Hence, we're using Keras 1 API, which requires versions a bit older of Python and TensorFlow, but for the purposes of this example it shouldn't be an issue.
