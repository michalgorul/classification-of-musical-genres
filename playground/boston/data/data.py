from keras.datasets import boston_housing


# Boston suburb dataset in the mid-1970s, given data points about the suburb at the time, such as the
# crime rate, the local property tax rate, and so on.  It has relatively few data points: only 506,
# split between 404 training samples and 102 test samples. And each feature in the input data
# (for example, the crime rate) has a different scale. For instance, some values are proportions, which take
# values between 0 and 1; others take values between 1 and 12, others between 0 and 100, and so on.


# The targets are the median values of owner-occupied homes, in thousands of dollars:
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Normalizing data

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# Note that the quantities used for normalizing the test data are computed using the
# training data. You should never use in your workflow any quantity computed on the
# test data, even for something as simple as data normalization.
