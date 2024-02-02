# JBI100-example-app

## About this app

Welcome to The Credit Canvas: a new opportunity to look at the 
finance scene in a straightforward and yet very insightful way!

This file will walk you through two files of code:
- The data.py file which contains all the code for cleaning and preprocessing
- And the app.py file which contains all the code for the dashboard

Let's get started!

## Requirements

* Python 3 (add it to your path (system variables) to make sure you can access it from the command prompt)
* Git (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

## Getting the app to run

To make sure that everything will run smoothly without errors, it is important that all the proper packages are installed.

Please install all required packages by running:
```
> pip install -r requirements.txt
```

Let's start off by taking a look at the file that cleans and preprocesses all the data. DO to the directory and find the data.py file.
There is one big function inside this file, which loads the all_data.csv (which is the orignal dataset from Kaggle) and applies cleaning and preprocessing to it.
If you scroll to the bottom, you can see that, just before the return function, the final_credit_0_60 dataframe is created. 
If you uncomment the line that makes this dataframe into a csv, you can save it locally onto your computer. 
To make it easy for you, we have already included this file here in this virtual environment. We have also done this with the Kaggle dataset. 

Now onto the most important part: the dashboard! The code for the dashboard is inside the app.py file, when you run the file, a link shows up.
Open this in your browser to see the results. You can edit the code in any editor (e.g. Visual Studio Code) and if you save it you will see the results in the browser.

### Code Writing
We used Dash and Python to write all the code.
All of the code that we have written has been done by ourselves, we have not gotten any of it from other sources.

## Resources

* [GitHub](https://github.com/giacomograzia/JBI100\_VisualizationProject)
* [Dash](https://dash.plot.ly/)
