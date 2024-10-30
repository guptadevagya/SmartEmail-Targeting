import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

class UserPredictor:
    def __init__(self):
        # define feature lists and transformations
        num_feats = [
            'past_purchase_amt', 'total_minutes', 'avg_session_duration',
            'days_since_last_visit', 'visits_to_tv', 'time_on_tv',
            'unique_pages', 'visited_tv'
        ]
        num_transformer = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
        ])

        cat_feats = ['badge', 'part_of_day']
        cat_transformer = OneHotEncoder(handle_unknown='ignore')

        # compose preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_feats),
                ('cat', cat_transformer, cat_feats)
            ]
        )

        # set up the pipeline with preprocessing and logistic regression
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', LogisticRegression(
                solver='liblinear', penalty='l1', C=0.1, class_weight='balanced'
            ))
        ])

    def _add_log_features(self, users, logs):
        # predefined categories for 'part_of_day'
        times = ['Night', 'Morning', 'Afternoon', 'Evening', 'Unknown']

        # convert 'date' to datetime and drop rows with errors
        logs['date'] = pd.to_datetime(logs['date'], errors='coerce')
        logs.dropna(subset=['date'], inplace=True)

        # compute 'time' as hour of day
        logs['time'] = logs['date'].dt.hour

        # assign 'part_of_day' based on 'time'
        logs['part_of_day'] = pd.cut(
            logs['time'],
            bins=[0, 6, 12, 18, 24],
            labels=times[:-1],
            right=False,
            include_lowest=True
        )
        logs['part_of_day'] = logs['part_of_day'].cat.add_categories(['Unknown']).fillna('Unknown')

        # calculate total_minutes per user
        logs['total_minutes'] = logs['duration'] / 60.0
        total_minutes = logs.groupby('id')['total_minutes'].sum().reset_index()

        # calculate session_count per user
        session_count = logs.groupby('id')['date'].nunique().reset_index(name='session_count')

        # calculate avg_session_duration per user
        avg_session = total_minutes.merge(session_count, on='id', how='left')
        avg_session['avg_session_duration'] = avg_session['total_minutes'] / avg_session['session_count']
        avg_session = avg_session[['id', 'avg_session_duration']]

        # calculate days_since_last_visit per user
        max_date = logs['date'].max()
        last_visit = logs.groupby('id')['date'].max().reset_index(name='last_visit')
        last_visit['days_since_last_visit'] = (max_date - last_visit['last_visit']).dt.days
        last_visit = last_visit[['id', 'days_since_last_visit']]

        # determine most frequent 'part_of_day' per user
        freq_time = logs.groupby('id')['part_of_day'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
        ).reset_index()

        # calculate visits_to_tv per user
        logs_tv = logs[logs['url'] == 'tv.html']
        visits_tv = logs_tv.groupby('id').size().reset_index(name='visits_to_tv')

        # calculate time_on_tv per user
        time_tv = logs_tv.groupby('id')['duration'].sum().reset_index(name='time_on_tv')
        time_tv['time_on_tv'] = time_tv['time_on_tv'] / 60.0  # convert to minutes

        # calculate visited_tv flag per user
        visited_tv = logs_tv[['id']].drop_duplicates()
        visited_tv['visited_tv'] = 1

        # calculate unique_pages per user
        unique_pages = logs.groupby('id')['url'].nunique().reset_index(name='unique_pages')

        # merge all features into users
        users = users.merge(total_minutes, on='id', how='left')
        users = users.merge(avg_session, on='id', how='left')
        users = users.merge(last_visit, on='id', how='left')
        users = users.merge(freq_time, on='id', how='left')
        users = users.merge(visits_tv, on='id', how='left')
        users = users.merge(time_tv, on='id', how='left')
        users = users.merge(visited_tv, on='id', how='left')
        users = users.merge(unique_pages, on='id', how='left')

        # fill missing values
        users['part_of_day'] = users['part_of_day'].fillna('Unknown')
        users['total_minutes'] = users['total_minutes'].fillna(0)
        users['avg_session_duration'] = users['avg_session_duration'].fillna(0)
        users['days_since_last_visit'] = users['days_since_last_visit'].fillna(0)
        users['visits_to_tv'] = users['visits_to_tv'].fillna(0)
        users['time_on_tv'] = users['time_on_tv'].fillna(0)
        users['visited_tv'] = users['visited_tv'].fillna(0)
        users['unique_pages'] = users['unique_pages'].fillna(0)

        # ensure correct data types
        users['visited_tv'] = users['visited_tv'].astype(int)
        users['part_of_day'] = users['part_of_day'].astype('category').cat.set_categories(times)

        return users

    def fit(self, train_users, train_logs, train_y):
        # fit the model using training data
        train_users = self._add_log_features(train_users, train_logs)
        x_train = train_users[
            [
                'past_purchase_amt', 'total_minutes', 'avg_session_duration',
                'days_since_last_visit', 'badge', 'part_of_day',
                'visits_to_tv', 'time_on_tv', 'unique_pages',
                'visited_tv'
            ]
        ]
        y_train = train_y['clicked']
        self.model.fit(x_train, y_train)

    def predict(self, test_users, test_logs):
        # predict using the model on new data
        test_users = self._add_log_features(test_users, test_logs)
        x_test = test_users[
            [
                'past_purchase_amt', 'total_minutes', 'avg_session_duration',
                'days_since_last_visit', 'badge', 'part_of_day',
                'visits_to_tv', 'time_on_tv', 'unique_pages',
                'visited_tv'
            ]
        ]
        predictions = self.model.predict(x_test)
        return predictions