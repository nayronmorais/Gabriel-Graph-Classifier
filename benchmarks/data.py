import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


PATH_RAWDATA = '../rawdatasets/'

_conversors = {
        'ord': OrdinalEncoder(handle_unknown='error'),
        'onehot': OneHotEncoder(handle_unknown='ignore', sparse=False)
    }

def categoric_to_numeric(data, conv_type='ord', remove_nan=True):
    """ Convert (by columns) a categorical data to numeric. The `data` must be a pandas.DataFrame. """

    try:
        if not issubclass(data.__class__, pd.DataFrame):
            raise ValueError('')

        if remove_nan:
            data = data.dropna(axis=0, how='any', inplace=False)

        subcols = data.select_dtypes(include=(object, pd.CategoricalDtype)).columns.values

        # There are not non numeric columns.
        if subcols.shape[0] == 0:
            return data

        datasub = data[subcols]

        conversor = _conversors[conv_type]
        datasub = conversor.fit_transform(datasub)

        if conv_type == 'ord':
             data.loc[:, subcols] =  datasub + 1 # Start from 1 instead 0.
        else:
            new_colnames = []
            [
                new_colnames.extend(
                                [BASE_COL_NAME + '_' + category for category in conversor.categories_[i]]
                ) for i, BASE_COL_NAME in enumerate(subcols)
            ]
            data = data.drop(columns=subcols, inplace=False)
            data = pd.concat((data, pd.DataFrame(datasub, columns=new_colnames)), axis=1)

        return data
    except KeyError:
        raise ValueError('')
