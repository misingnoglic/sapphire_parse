import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.predictor.interpretable_predictor import InterpretableTabularPredictor
from sklearn.preprocessing import OneHotEncoder


train = False
# Simple Model:
model_path = "AutogluonModels/ag-20241128_213837"

# Complex Model:
# model_path = 'AutogluonModels/ag-20241128_214225'

intensities_ranked = ['Colorless', 'Near Colorless', 'Faint', 'Very Light', 'Light', 'Medium Light', 'Medium', 'Medium Intense', 'Intense', 'Vivid']
categorical_columns = ['Color', 'Shape', 'Clarity', 'Cut', 'Color Intensity', 'Origin', 'Treatments']

def get_one_hot_columns(df, categorical_cols):
    column_values = {}
    for col in categorical_cols:
        column_values[col] = sorted(df[col].unique())
    return column_values


def clean_df(df, one_hot_columns):
    df["length_width_ratio"] = df["Length"]/df["Width"]
    # If Total Price is a string, convert to float
    if df['Total Price'].dtype == 'O':
        df['Total Price'] = df['Total Price'].str.replace(',', '').str.replace('$', '').astype(float)
    df = df.drop(columns=["name", "url", "image_url", "Item ID"], errors='ignore')
    # Stuff shehean does not have
    df = df.drop(columns=['Shape', 'Clarity', 'Height', 'Cut'], errors='ignore')
    df = df.drop(columns=["Per Carat Price", "Price per Length"], errors='ignore')
    # df = df[['Total Price', 'Weight', 'Color Intensity', 'Origin', 'Treatments', 'Length', 'Width', 'Height', 'Color', 'Shape', 'Clarity', 'Cut']]
    df = df[df['Total Price'] < 15000]
    df = df[df['Weight'] < 12]
    # Drop ones where color is padparadscha
    # df = df[df["Color"] != "Padparadscha (Pinkish-Orange / Orangish-Pink)"]
    for cat_col in categorical_columns:
        if cat_col not in df.columns:
            continue
        unique_columns = one_hot_columns[cat_col]
        for col in unique_columns:
            df[f'{cat_col}_{col}'] = df[cat_col] == col
            df[f'{cat_col}_{col}'] = df[f'{cat_col}_{col}'].astype(int)
        df = df.drop(columns=[cat_col])

    return df

if __name__ == '__main__':
    df = pd.read_csv("sapphires.csv")
    one_hot_columns = get_one_hot_columns(df, categorical_columns)
    df = clean_df(df, one_hot_columns)

    # Define features and target
    # X = df.drop('Total Price', axis=1)
    # y = df['Total Price']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # df['Color Intensity'] = df['Color Intensity'].apply(lambda x: intensities_ranked.index(x))
    # Split df into 80 20 split
    train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)
    if train:
        predictor = InterpretableTabularPredictor(label='Total Price', problem_type='regression').fit(
            train_data=train_data,
            time_limit=180,
            hyperparameters={'IM_FIGS': [
                {'max_rules': 50},
                # {'max_rules': 6}, {'max_rules': 10}, {'max_rules': 15}
            ]}

        )
    else:
        predictor = InterpretableTabularPredictor.load(model_path)

    scores = predictor.evaluate(test_data)
    print(scores)

    print(predictor.leaderboard_interpretable(verbose=True))
    predictor.print_interpretable_rules(complexity_threshold=1000)

    # Get plot of tree
    imodel = predictor._trainer.load_model('Figs').model
    imodel.plot(dpi=1000, filename='Figs.png')

    print('Evaluate prices of third party')

    df_third_party = pd.read_csv('INTA.csv')
    df_third_party = clean_df(df_third_party, one_hot_columns)

    # Predict price of df
    y_pred = predictor.predict(df_third_party)
    print('Predicted minus actual')
    print(y_pred - df_third_party['Total Price'])

    print('Discount percentage')
    print((y_pred - df_third_party['Total Price'])/y_pred)