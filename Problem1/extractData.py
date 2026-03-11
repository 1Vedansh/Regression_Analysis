from datetime import datetime
import math
import random

# Initialize dataset list with headers
dataset = []
headers = "Year,Month,Day,HourSin,HourCos,Holiday,WorkingDay,Season_1,Season2,Season_3,Weather_1,Weather_2,Weather_3,Temperature,Atemp,Humidity,WindSpeed,Casual,Registered,Count\n"
dataset.append(headers)

# Read the original train.csv file and partially preprocess each line
with open(file="train.csv", mode="r") as file:
    lines = file.readlines()
    print(f"Headers: {lines[0].strip()}")
    for index, line in enumerate(lines):
        row = ""

        if index == 0:
            continue
        
        print(f"Partially Preprocessed line: {line.strip()}")

        columns = line.strip().split(",")
        
        date = columns[0]
        season = columns[1]
        holiday = columns[2]
        workingday = columns[3]
        weather = columns[4]
        temp = columns[5]
        atemp = columns[6]
        humidity = columns[7]
        windspeed = columns[8]
        casual = columns[9]
        registered = columns[10]
        count = columns[11]

        dt_object = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

        # Compute hour sine and cosine for cyclical representation
        hour_sin = math.sin(2 * math.pi * dt_object.hour / 24)
        hour_cos = math.cos(2 * math.pi * dt_object.hour / 24)   

        row += f"{dt_object.year},{dt_object.month},{dt_object.day},{hour_sin},{hour_cos},{holiday},{workingday}"
        
        # One hot encode season and weather
        season = []
        for i in range(1, 4):
            if int(columns[1]) == i:
                season.append("1")
            else:
                season.append("0")

        weather = []
        for i in range(1, 4):
            if int(columns[4]) == i:
                weather.append("1")
            else:
                weather.append("0")

        for i in season:
            row += f",{i}"

        for i in weather:
            row += f",{i}"

        row += f",{temp},{atemp},{humidity},{windspeed}"
        row += f",{casual},{registered},{count}\n"

        dataset.append(row)

# Split dataset into training and testing sets
option = int(input("1 for random split, 2 for predefined split: "))

with open("processedTrain.csv","w") as train, open("processedTest.csv","w") as test:
    for index,data in enumerate(dataset):
        if(index==0):
            train.write(data)
            test.write(data)
            continue

        if(option == 1):
            if(random.random() < 0.75):
                train.write(data)
            else:
                test.write(data)
        else:
            if(index < 7500):
                train.write(data)
            else:
                test.write(data)
