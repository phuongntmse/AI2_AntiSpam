import csv
import math
import time
from tkinter import *

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils import np_utils
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg)


class BayesModel:
    def __init__(self, mode):
        self.mode = mode
        print(mode)
        if mode == 0:
            self.training_data = 'Spam detection - For model creation.csv'
        else:
            self.training_data = 'Spam detection - For model creation_Processed_data.csv'
        self.test_data = 'Spam detection - For prediction.csv'
        self.class_column = 0

    def run(self):
        training_set = self.loadCsv(self.training_data)
        self.changeClassName(training_set)
        trainingData = self.loadCsv(self.training_data)
        testData = self.loadCsv(self.test_data)
        for xi in testData:
            xi.pop(0)
        for xi in trainingData:
            xi.pop(0)

        beginT = time.time()
        classified_data2 = self.classify_data(training_set, trainingData)
        tData = self.loadCsv(self.training_data)
        self.changeClassName(tData)
        rs = [(self.accuracy(classified_data2, tData), time.time() - beginT)]
        beginT = time.time()
        classified_data = self.classify_data(training_set, testData)
        rs.append((self.accuracy(classified_data, self.loadCsv(self.test_data)), time.time() - beginT))
        return rs

    def loadCsv(self, filename):
        lines = csv.reader(open(filename, "r"))
        dataset = list(lines)
        dataset.pop(0)  # cut name column row0
        return dataset

    def getClassList(self, dataset):
        classes = []
        for i in range(len(dataset)):
            if dataset[i][self.class_column] not in classes:
                classes.append(dataset[i][self.class_column])
        return classes

    def changeClassName(self, dataset):
        classes = self.getClassList(dataset)
        for i in range(len(classes)):
            for j in range(len(dataset)):
                if dataset[j][self.class_column] == classes[i]:  # convert No yes to 0 1
                    dataset[j][self.class_column] = i
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]

    def count_by_classes(self, dataset):
        classes = self.getClassList(dataset)
        count = {}
        for i in range(len(classes)):
            count[classes[i]] = 0
            for j in range(len(dataset)):
                if dataset[j][self.class_column] == classes[i]:
                    count[classes[i]] += 1
        return count

    def class_prob(self):
        prob_by_class = []
        count = countByClasses
        total_count = 0
        for c_i in (range(len(count))):
            total_count += count[c_i]
        for c_i in count:
            prob_by_class.append(count[c_i] / total_count)
        return prob_by_class

    def get_info(self, dataset):
        info = []
        for column in zip(*dataset):
            info.append((self.mean(column), self.std_deviation(column)))
        del info[self.class_column]  # cut class_column
        return info

    def get_info_class(self, dataset):
        separated_data = self.separate_data(dataset)
        info_class = {}
        for class_ci, matrix in separated_data.items():
            info_class[class_ci] = self.get_info(matrix)
        return info_class

    def separate_data(self, dataset):
        separated_data = dict()
        for i in range(len(dataset)):
            row = dataset[i]
            class_ci = row[self.class_column]
            if class_ci not in separated_data:
                separated_data[class_ci] = list()
            separated_data[class_ci].append(row)
        return separated_data

    def mean(self, column):
        return sum(column) / float(len(column))

    def std_deviation(self, column):
        average = self.mean(column)
        variance = sum([(x - average) ** 2 for x in column]) / float(len(column) - 1)
        return math.sqrt(variance)

    def calculate_Gauss_prob(self, x, mean, std_deviation):
        e = -(math.pow(x - mean, 2) / (2 * math.pow(std_deviation, 2)))
        prob = math.exp(e) / (math.sqrt(2 * math.pi) * std_deviation)
        return prob

    def calculate_class_prob(self, xi):
        info_class = infoClasses
        info_x_i = {}
        for class_ci in info_class:
            info_x_i[class_ci] = 1
            for j in range(len(info_class[class_ci])):
                mean, std_dev = info_class[class_ci][j]
                if mean == 0 and std_dev == 0 and float(xi[j]) == 0:
                    prob = 1
                elif mean == 0 and std_dev == 0 and float(xi[j]) != 0:
                    prob = 0
                else:
                    prob = self.calculate_Gauss_prob(float(xi[j]), mean, std_dev)
                info_x_i[class_ci] = info_x_i[class_ci] * prob  # cal p(xi1/c1)*p(xi2/c1)...
        return info_x_i

    def clasify_xi(self, x_i):
        # calulate p(ci)* p(xi1/c1)*p(xi2/c1)...
        info_x_i = self.calculate_class_prob(x_i)
        prob_by_class = probByClasses
        idx = 0
        prob_x_i = {}
        for label in info_x_i:
            prob = info_x_i[label] * prob_by_class[idx]
            idx += 1
            prob_x_i[label] = prob

        max_prob = -1
        lable_xi = None
        for label, prob in prob_x_i.items():
            if lable_xi is None or prob > max_prob:
                max_prob = prob
                lable_xi = label
        return lable_xi

    def classify_data(self, dataset, validation_data):
        classified_data = []
        # excute some common functions
        global countByClasses, probByClasses, infoClasses
        countByClasses = self.count_by_classes(dataset)
        probByClasses = self.class_prob()
        infoClasses = self.get_info_class(dataset)
        # ---end---
        for i in range(len(validation_data)):
            label = self.clasify_xi(validation_data[i])
            classified_data.append(label)
        return classified_data

    def accuracy(self, classified_data, validation_data):
        check = 0
        for i in range(len(validation_data)):
            if classified_data[i] == float(validation_data[i][self.class_column]):
                check += 1
        percent = float(check / float(len(validation_data)) * 100.0)
        return percent


class NeuronModel:
    def __init__(self, have_hidden_layer, enter_epoch, enter_batch_size):
        self.df = pd.read_csv('Spam detection - For model creation.csv', encoding='latin-1')
        self.df['GOAL-Spam'] = self.df['GOAL-Spam'].map({'Yes': 1, 'No': 0})
        self.have_hidden_layer = have_hidden_layer
        if enter_epoch != 0:
            self.enter_epoch = enter_epoch
        else:
            if have_hidden_layer:
                self.enter_epoch = 300
            else:
                self.enter_epoch = 180
        if enter_batch_size != 0:
            self.enter_batch_size = enter_batch_size
        else:
            self.enter_batch_size = 20

    def run(self):
        beginT = time.time()
        # standardization
        target_column = ['GOAL-Spam']
        predictors = list(set(list(self.df.columns)) - set(target_column))
        self.df[predictors] = self.df[predictors] / self.df[predictors].max()
        X = self.df[predictors].values
        y = self.df[target_column].values
        y_train = y
        X_train = X
        y_train = np_utils.to_categorical(y_train)

        # Model Network
        if self.have_hidden_layer:
            model = Sequential()
            model.add(Dense(500, activation='sigmoid', input_dim=57))
            model.add(Dropout(0.2))
            model.add(Dense(2, activation='softmax'))
        else:
            model = Sequential()
            model.add(Dense(2, activation='softmax', input_dim=57))

        # Compile the model, loss measure
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Validation dataset
        data_eval = pd.read_csv('Spam detection - For prediction.csv', encoding='latin-1')

        # standardization
        target_column = ['Spam']
        predictors = list(set(list(data_eval.columns)) - set(target_column))
        data_eval[predictors] = data_eval[predictors] / data_eval[predictors].max()
        X_eval = data_eval[predictors].values
        y_eval = data_eval[target_column].values
        y_eval = np_utils.to_categorical(y_eval)

        # build the model
        graph_data = model.fit(X_train, y_train, epochs=self.enter_epoch, batch_size=self.enter_batch_size,
                               validation_data=(X_eval, y_eval),
                               verbose=0)

        # draw graph
        fig_plt = plt.Figure()
        graph_plt = fig_plt.add_subplot(111)
        graph_plt.plot(graph_data.history['accuracy'])
        graph_plt.plot(graph_data.history['val_accuracy'])
        if self.have_hidden_layer:
            graph_plt.set_title('Model M2')
        else:
            graph_plt.set_title('Model M1')
        graph_plt.set_ylabel('accuracy')
        graph_plt.set_xlabel('epoch')
        graph_plt.legend(['train', 'test'], loc='lower right')
        # plt.show()

        # Accuracy on data
        rs = []
        pred_train = model.predict(X_train)
        scores = model.evaluate(X_train, y_train, verbose=0)
        rs.append((scores[1], 1 - scores[1]))
        pred_eval = model.predict(X_eval)
        scores2 = model.evaluate(X_eval, y_eval, verbose=0)
        rs.append((scores2[1], 1 - scores2[1]))
        rs.append(time.time() - beginT)
        return fig_plt, rs


# import matplotlib.pyplot as plt
def start():
    result.delete('1.0', END)
    selected_model = model.get()
    if selected_model == 0 or selected_model == 1:
        model_object = BayesModel(selected_model)
        final_result = model_object.run()
        result.insert(END, "--------BEGIN--------\n")
        result.insert(END, "1) Accuracy on training data: {0:.8f}%\nError on training data: {1:.8f}%\n".format(
            final_result[0][0], 100.0 - final_result[0][0]))
        result.insert(END,
                      "2) Accuracy on eval data: {0:.8f}%\nError on eval data: {1:.8f}%\n".format(final_result[1][0],
                                                                                                  100.0 -
                                                                                                  final_result[1][0]))
        result.insert(END, "3) Total process time: {0:.8f}s\n".format(final_result[0][1] + final_result[1][1]))
        result.insert(END, "--------END--------\n")
        print(final_result)
        return
    else:
        enter_epoch = int(entry_epoch.get())
        enter_batch_size = int(entry_batch_size.get())
        if selected_model == 2:
            model_object = NeuronModel(False, enter_epoch, enter_batch_size)
        else:
            model_object = NeuronModel(True, enter_epoch, enter_batch_size)
        final_plt, final_result = model_object.run()
        result.insert(END, "--------BEGIN--------\n")
        result.insert(END, "1) Accuracy on training data: {0:.8f}%\nError on training data: {1:.8f}%\n".format(
            final_result[0][0] * 100, final_result[0][1] * 100))
        result.insert(END, "2) Accuracy on eval data: {0:.8f}%\nError on eval data: {1:.8f}%\n".format(
            final_result[1][0] * 100,
            final_result[1][
                1] * 100))
        result.insert(END, "3) Total process time: {0:.8f}s\n".format(final_result[2]))
        result.insert(END, "--------END--------\n")
        print(final_result)
        # open image window
        simulate_window = Toplevel(root)
        simulate_window.title("Result")
        simulate_window.geometry("700x500")
        canvas_frame = Frame(simulate_window, width=700, height=500)
        canvas_frame.grid_propagate(False)
        canvas_frame.grid(row=0, column=1, padx=30)
        fig = final_plt
        in_canvas = FigureCanvasTkAgg(fig, canvas_frame)
        in_canvas.get_tk_widget().pack(pady=10)
        in_canvas.draw()
        return


# -----------------------------------UI-----------------------------------
root = Tk()
root.geometry("700x400")
root.title("Anti-Spam Construction")
frame = Frame(root, width=700, height=400)
frame.pack(padx=10)

# Config --Start
setup_frame = Frame(frame, width=250, height=400)
setup_frame.grid_propagate(False)
setup_frame.grid(row=0, column=0, padx=10)
# Radio group button
modelGroup = LabelFrame(setup_frame, text="Select algorithm")
modelGroup.pack()
model = IntVar()
bayes_full_data = Radiobutton(modelGroup, text="Bayes Model with Full Data", variable=model, value=0)
bayes_full_data.pack(anchor=W)
bayes_full_data = Radiobutton(modelGroup, text="Bayes Model with Partly Data", variable=model, value=1)
bayes_full_data.pack(anchor=W)
neuron_without_hidden_layers = Radiobutton(modelGroup, text="Neuron Network without hidden layer", variable=model,
                                           value=2)
neuron_without_hidden_layers.pack(anchor=W)
neuron_with_hidden_layers = Radiobutton(modelGroup, text="Neuron Network with one hidden layer", variable=model,
                                        value=3)
neuron_with_hidden_layers.pack(anchor=W)
# Enter epoch (Only used for neuron model)
epoch_frame = Frame(setup_frame)
epoch_frame.pack(pady=20)
label_epoch = Label(epoch_frame, text="Epoch (Only used for neuron model):")
label_epoch.grid(row=0, column=0)
entry_epoch = Entry(epoch_frame, width=10)
entry_epoch.insert(0, '0')
entry_epoch.grid(row=0, column=1)
# Enter epoch (Only used for neuron model)
batch_size_frame = Frame(setup_frame)
batch_size_frame.pack(pady=20)
label_batch_size = Label(batch_size_frame, text="Batch_size (Only used for neuron model):")
label_batch_size.grid(row=0, column=0)
entry_batch_size = Entry(batch_size_frame, width=10)
entry_batch_size.insert(0, '0')
entry_batch_size.grid(row=0, column=1)
# Start button
startbutton = Button(setup_frame, text="Start", command=start, width=20)
startbutton.pack(pady=20)
# Config --End

# Result information --Start
rs_frame = Frame(frame, width=450, height=400)
rs_frame.grid_propagate(False)
rs_frame.grid(row=0, column=1, padx=10)
rs_label = Label(rs_frame, text="Result").pack()
result = Text(rs_frame, width=40, height=20, wrap="word")
result.pack()
scrollb_y = Scrollbar()
scrollb_y.place(in_=result, relx=1.0, relheight=1.0, bordermode="outside")
scrollb_y.configure(command=result.yview)
# Result information --End

# other params
root.mainloop()
