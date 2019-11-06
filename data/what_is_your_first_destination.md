# What's your first destination?

Founed in 2008, Airbnb till now has hosted more that 60 milion of travelers across 34 thousand of cities around the world. Milion of hosts and travelers decide to creat an account to publish a listing for their rooms, flat and apartments. This is all done to create an unique experience that make travelers feel like their at home.
However, a common issue to new travelers is to make the first booking. Often they surf through thousand of listing trying to make reservations and waiting for responses. The goal of this challenge is to build a model able to predict what will be the next destination of a traveler. 

## Data Description

The dataset we are researching is provided by Airbnb and contains a list of users along with their demographics, web session records, and some summary statistics. The whole dataset contains 5 csv files: train-users, test-users, sessions, countries and age-gender-bkts.

You are asked to predict which country a new user's first booking destination will be. All the users in this dataset are from the USA.

There are 12 possible outcomes of the destination country: 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF' (no destination found), and 'other'. Please note that 'NDF' is different from 'other' because 'other' means there was a booking, but is to a country not included in the list, while 'NDF' means there wasn't a booking.
The training and test sets are split by dates. In the test set, you will predict all the new users with first activities after 7/1/2014. In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010


- *train_users.csv*: contains 171239 training examples with 16 properties: id, date-account-crreated, date-first-booking, gender, age, signup-method, signup-flow, language, affiliate-channel, affiliate-provider, first-affiliate-tracked, signup-app, first-device-type, first-browser, country-destination and timestamp-firstactive

2. *test_users.csv*: has 43673 items and 15 properties. The field country-destination (target variable) is missing and it si the one you have to predict.

3. *sessions.csv*: contains 5600850 elements with 6 properties: user-id, action, action-type, action-detail, device-type and secs-elapsed.

4. *countries.csv*: contains statistics of destination countries in this dataset and their geopraphic information. It has information for 10 countries and their 7 different
properties, such as longitude and latitude

5. *age_gender_bkts.csv*: this file contains statistics of users’ age group, gender, country of destination. It consists 420 examples and 5 properties

6. *sample_submission.csv*: a template for submitting your predictions.

## Submission

The intermediate submissions have to contain:
- a csv with your predictions

The final submission has to contain:
- a csv with your final prediction
- notebooks with an [Exploratory Data Analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis) of the datasets. The goal is to show what you have understood about the datasets, please provide useful visualizations for the evaluator
- all the code used for: data preprocessing, data transformation and model development
- a checkpoint of the trained model that can be reloaded to check the results.
- a final notebook and or presentation that let non-data-scientist to understand your findings and evauate the results obtained by your model

