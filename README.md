# Spotify-Data-Analysis

CS 542 Machine Learning - Course Project

Zuoqi Zhang, Charles Huang

File List:

* data_us.csv - dataset downloaded from [Kaggle](https://www.kaggle.com/edumucelli/spotifys-worldwide-daily-song-ranking/data).
* features_df.csv - dataset we used for data analysis.
* datasetwithstreams.csv - dataset with stream counts used for data analysis.
* dataset.csv - dataset we used for model training.
* process_data.py - generates two data frames.
* analyze_data.py - analyzes data.
* train_models.py - uses different models to do classification.
* find_best.py - finds best predictors (feature subsets).

process_data.py is included, but it would be best not to run it since it requires an automically generated Spotify API access token, which we cannot provide as it expires every hour and needs to be re-obtained.

find_best.py may take a few minutes to run depending on your computer's processing speed.
