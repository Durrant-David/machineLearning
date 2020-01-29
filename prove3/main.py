import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor


# function to get the NN
def knn_display(x, y, nn=3, reg=0):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    if reg:
        knn = KNeighborsRegressor(n_neighbors=nn)
    else:
        knn = KNeighborsClassifier(n_neighbors=nn)
    knn.fit(x_train, y_train)
    print(knn.score(x_test, y_test))


#
# Start the car data training
#
# Define the headers since the data does not have any
car_header = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
car_cleanup_data = {"buying": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                    "maint": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                    "doors": {"5more": 5},
                    "persons": {"more": 6},
                    "lug_boot": {"small": 1, "med": 2, "big": 3},
                    "safety": {"low": 1, "med": 2, "high": 3},
                    "class": {"vgood": 4, "good": 3, "acc": 2, "unacc": 2}}

car = pd.read_csv("data/car.data", names=car_header)

obj_car = car.select_dtypes(include=['object']).copy()
obj_car.replace(car_cleanup_data, inplace=True)

x = obj_car.drop(["class"], axis=1)
y = obj_car["class"]

knn_display(x, y, 8)

#
# Start the mpg data training
#
mpg_header = ["mpg", "cylinders", "displacement",
              "horsepower", "weight", "acceleration",
              "model_year", "origin", "car_name"]
mpg_cleanup_data = {"horsepower": {"?": np.NaN}}

mpg = pd.read_csv("data/auto-mpg.data", names=mpg_header, delim_whitespace=True)

obj_mpg = mpg
obj_mpg.replace(mpg_cleanup_data, inplace=True)
obj_mpg['car_name'] = obj_mpg['car_name'].astype('category')
obj_mpg["car_name_cat"] = obj_mpg['car_name'].cat.codes
obj_mpg.dropna(inplace=True)

x = obj_mpg.drop(["mpg", "car_name"], axis=1)
y = obj_mpg["mpg"]
knn_display(x, y, 10, 1)

#
# Start the mpg with mean missing data
#
obj_mpg['horsepower'] = obj_mpg['horsepower'].astype(float)
mean_cylinders = obj_mpg['cylinders'].mean()
mean_displacement = obj_mpg['displacement'].mean()
mean_horsepower = obj_mpg['horsepower'].mean()
mean_weight = obj_mpg['weight'].mean()
mean_acceleration = obj_mpg['acceleration'].mean()
mean_model_year = obj_mpg['model_year'].mean()
mean_origin = obj_mpg['origin'].mean()
obj_mpg['cylinders'] = obj_mpg['cylinders'] - mean_cylinders
obj_mpg['displacement'] = obj_mpg['displacement'] - mean_displacement
obj_mpg['horsepower'] = obj_mpg['horsepower'] - mean_horsepower
obj_mpg['weight'] = obj_mpg['weight'] - mean_weight
obj_mpg['acceleration'] = obj_mpg['acceleration'] - mean_acceleration
obj_mpg['model_year'] = obj_mpg['model_year'] - mean_model_year
obj_mpg['origin'] = obj_mpg['origin'] - mean_origin
x = obj_mpg.drop(["mpg", "car_name"], axis=1)
y = obj_mpg["mpg"]
knn_display(x, y, 10, 1)

#
# Start the math data training
#
math_cleanup_data = {"school": {"GP": 0, "MS": 1},
                     "sex": {"F": 0, "M": 1},
                     "address": {"R": 0, "U": 1},
                     "famsize": {"LE3": 0, "GT3": 1},
                     "Pstatus": {"A": 0, "T": 1},
                     "schoolsup": {"no": 0, "yes": 1},
                     "famsup": {"no": 0, "yes": 1},
                     "paid": {"no": 0, "yes": 1},
                     "activities": {"no": 0, "yes": 1},
                     "nursery": {"no": 0, "yes": 1},
                     "higher": {"no": 0, "yes": 1},
                     "internet": {"no": 0, "yes": 1},
                     "romantic": {"no": 0, "yes": 1}}
math = pd.read_csv("data/student/student-mat.csv", sep=";")

obj_math = math
obj_math.replace(math_cleanup_data, inplace=True)

obj_math["Mjob"] = obj_math["Mjob"].astype('category')
obj_math["Fjob"] = obj_math["Fjob"].astype('category')
obj_math["reason"] = obj_math["reason"].astype('category')
obj_math["guardian"] = obj_math["guardian"].astype('category')
obj_math["Mjob_cat"] = obj_math["Mjob"].cat.codes
obj_math["Fjob_cat"] = obj_math["Fjob"].cat.codes
obj_math["reason_cat"] = obj_math["reason"].cat.codes
obj_math["guardian_cat"] = obj_math["guardian"].cat.codes

x = obj_math.drop(["Mjob", "Fjob", "reason", "guardian", "G3"], axis=1)
y = obj_math["G3"]
knn_display(x, y, 9, 1)

#
# Start the math one hot
#
obj_math2 = math
obj_math2 = pd.get_dummies(obj_math, columns=["Mjob", "Fjob", "reason", "guardian"],
                          prefix=["Mjob", "Fjob", "reason", "guardian"]).head()
# obj_math.replace(math_cleanup_data, inplace=True)

# print(obj_math)
x = obj_math2.drop(["G3"], axis=1)
y = obj_math2["G3"]
knn_display(x, y, 3, 1)
