import math
import sys

def neuralnetworklearning(dataset, number_of_iterations,attribute_names,learning_rate):
    weight=0
    for x in range(number_of_iterations):
        list_output = []
        attributes=[]
        counter = 0
        output = 0
        idx = 0
        for names in attribute_names:
            attributes.append(dataset.get_column(attribute_names)[idx])

        for attribute in attributes:
            estimate_output = weight*attribute
            error = calculate_error(attributes[counter+1],estimate_output)
            updated_weight = update_weights(weight,learning_rate,error,attribute,estimate_output)
            output = output+estimate_output
            display_pattern = ''.join("w(",attribute_names[counter], ") = ","%.2f" % updated_weight)
            counter = counter + 1
            list_output.append(display_pattern)

        final_output = ''.join("Output = " ,sigmoid(output))
        list_output.append(final_output)
        idx=(idx+1)%number_of_iterations



        print("After iteration {}".format(x+1))
        for contents in list_output:
            print(contents, ",", end = "")



def sigmoid(w_prod):
    sigmoid = 1/(1+math.exp(-w_prod))
    return sigmoid

def calculate_error(exp_output,w_prod):
    error = exp_output-sigmoid(w_prod)
    return error

def update_weights(initial_weight,learning_rate,computed_error,input_x,w_prod):
    updated_weight = initial_weight+ learning_rate*computed_error*input_x*sigmoid(w_prod)*(1-sigmoid(w_prod))
    return updated_weight



class DataFrame(object):
    def __init__(self, column_names, column_values):
        '''
        column_names - list of column/feature names
        column_values - list of list of feature values
        '''
        self.columns = column_names
        self.column_values = column_values
        self.data_dict = dict(zip(self.column_names, self.column_values))

    def __init__(self, data_dict):
        self.columns = data_dict.keys()
        self.column_values = data_dict.values()
        self.data_dict = data_dict

    def get_count(self):
        return (len(list(self.column_values)[0]))

    def get_column(self, column_name):
        return self.data_dict[column_name]

def main():

    if sys.argv[1]:
        training_file_name = sys.argv[1]

    if sys.argv[2]:
        test_file_name = sys.argv[2]

    if sys.argv[3]:
        learning_rate=sys.arv[3]

    if sys.argv[4]:
        number_of_iterations=sys.argv[4]

    else:
        sys.exit()

    with open(training_file_name) as train_file:
        names_train = train_file.readline().split()
        lines_train = train_file.readlines()


    with open(test_file_name) as test_file:
        names_test=test_file.readline().split()
        lines_test=test_file.readlines()





    attribute_values_train = [list() for _ in range(len(names_train))]
    attribute_values_test  = [list() for _ in range(len(names_test))]




    for line_idx, line in enumerate(lines_train):
        data_row=line.split()
        for col_idx, value in enumerate(data_row):
            attribute_values_train[col_idx].append(int(value))

    for line_idx, line in enumerate(lines_test):
        data_row=line.split()
        for col_idx, value in enumerate(data_row):
            attribute_values_test[col_idx].append(int(value))



    training_dataset = DataFrame(dict(zip(names_train,attribute_values_train)))
    testing_dataset = DataFrame(dict(zip(names_test,attribute_values_test)))

    neuralnetworklearning(training_dataset,number_of_iterations,names_test,learning_rate)
    neuralnetworklearning(testing_dataset,number_of_iterations,names_train,learning_rate)






