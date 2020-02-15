## Content

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model

- README.md

## Getting Started

Clone the repository, to get our Notebooks, Presentation and Project Report.

```
git clone https://github.com/Mavengence/Kaggle-Seattle-Airbnb-Analysis.git
```

### Prerequisites

- Of course you need git to get the source
- If you want to compile the report or the presentation by ur self u need a LaTex Compiler for your OS and maybe an IDE which makes things easier
- If you want to compile, train and play with our Code you need a python working environment. We used Jupyter Notebooks. The requiered packeges you can see in the Notebooks itself.
- Get the [Dataset](https://www.kaggle.com/airbnb/seattle) from Kaggle.com

### Run the Notebook
Open a Terminal Window and type

```
cd/Disaster-Response-Unit-Figure_Eight python run.py
```

Now, open another Terminal Window.

Type

```
env|grep WORK
```

### Deployment

Just pull the repo and change anything you want

## Authors

* **Tim Löhr** - [GitHub Mavengence](https://github.com/Mavengence)

## License

Pretty much the BSD license, just don't repackage it and call it your own please!
Also if you do make some changes, feel free to make a pull request and help make things more awesome!

## Acknowledgments

Thanks Udacity Data Scientist Nanodegree program for providing the boiler template and the data and especially the knowledge for solving this project
