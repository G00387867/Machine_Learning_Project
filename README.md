# Machine_Learning_Project


This repository contains a project work for implementing machine-learning web service.

### Project requirements

Create a web service that uses machine learning to make predictions based on the data set powerproduction. The goal is;

- Produce a model that accurately predicts wind turbine power output from wind speed values, as in the data set with analysis of its accuracy.

- Develop a web service that will respond with predicted power values based on speed values sent as HTTP requests:

1. Python script that runs a web service based on the model.
2. Dockerfile to build and run the web service in a container.
  
***

### Methodology

- Importing the data set

- Explore the data set

- pre-process and cleanse the data set

- Perform Analysis

- Create python script that runs the web service

- Dockerfile to build and run the web service in a container. 


***

### To run the Jupyter Notebook

Download and Install Anaconda's latest release from it's official and licensed source. The anaconda package includes the python code and the packages (libraries) needed for the computation and visualisation of the contetnts of the `power-production.ipynb` file.

1. Click on **`Code`** icon on the upper right corner and copy the `link` under `HTTPS`

2. Open the `CLI` or `cmd` on the machine navigate to the required directory insert `$ git clone` `link of the repository`

3. Insert `$ jupyter notebook`

4. A web browser is automatically initiated, where you can see the file `power-production.ipynb`

***
### To run the web-service

Insert the following in the CLI:

```
$ <span style=“color:green;”>set</span> FLASK_APP=app.py
```
then;

```
$ python -m flask run
```

***

### License

This repository was enrolled under the MIT license. Please click [**Here**](https://github.com/G00387867/FDA-project/blob/main/LICENSE) for further information.
