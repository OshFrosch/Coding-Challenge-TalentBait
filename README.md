# Coding-Challenge-TalentBait


This code challenge is meant to evaluate your problem-solving skills in areas of text
preprocessing, data manipulation and ML modeling in Python.

Here you'll find a data set (JSON file) with about 10.000 sentences. Each entry has a category (key: label) and a sentence (key: text). 

## For example:
* {label: “tech”, text: “Gute Kenntnisse in Oracle PL SQL sowie Java und/oder Python”}
* {label: “informative”, text: “Kreativität und schnelle Auffassungsgabe”}
* {label: “none”, text: “Starttermin: Je früher, desto besser.”}

## There are three labels: 
* soft: for soft skills like “team player”
* tech: for tech skills like “python experience”
* none: for all other type of sentences

## Tasks:
1. Read the file and prepare the data for text classification (take decisions on how to clean the data, what to do with punctuation signs or special characters, etc.)
2. Train a classification model that takes a sentence and predicts the class it belongs to. You are free to use any type of model or libraries for this task.
3. Provide evaluation metrics about the trained model, eventually visualize the results
4. Explain your results: Is it a good model, what would you do to improve the model. 
5. Bonus 1 (optional): Share the trained model as an application. It can be a command line application or any other format that can be executed (it doesn't matter the OS, I have access to Linux, MacOS, and Windows)

6. Bonus 2 (optional): Suggest some improvements for the future


#### How to deliver the code challenge and some information
* Create a GitHub repo to host your code
* Please send us the challenge within 2 days after receiving it
* Use a Jupyter Notebook (Optional)
* Add starting instructions to a README.md
* Use Python 3.6 (I used python 3.8; I hope this was fine)

## Command line tool

I decided to write a python skript which can be run in the terminal with the input as the arguments:

`python classify_job_requirements.py 'input'`
