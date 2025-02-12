import matplotlib.pyplot as plt


def draw_plot(percentage_used,Algorithm_name,ac,f1,precision,recall):

    fig, axis = plt.subplots(2, 2)

    axis[0,0].set_title("                                                                                                                             "+Algorithm_name)
    axis[0,0].plot(percentage_used,ac,'ro-',label="Accuracy %")
    axis[0,0].set_ylabel("Accuracy %")
    axis[0,0].set_xlabel("Train set size %")

    axis[0,1].plot(percentage_used,f1,'go-',label="F1")
    axis[0,1].set_ylabel("F1")
    axis[0,1].set_xlabel("Train set size %")

    axis[1,0].plot(percentage_used,precision,'bo-',label="Precision")
    axis[1,0].set_ylabel("Precision")
    axis[1,0].set_xlabel("Train set size %")

    axis[1,1].plot(percentage_used,recall,'o-',label="Recall")
    axis[1,1].set_ylabel("Recall")
    axis[1,1].set_xlabel("Train set size %")

    plt.show()

#############################################################################################################
'''
DATA
'''
#############################################################################################################

percentage_used = [10,20,30,40,50]

RNN_ac = [53,81,80,83,79]
RNN_f1 = [0.53,0.81,0.8,0.83,0.79]
RNN_precision = [0.53,0.8,0.79,0.826,0.79]
RNN_recall = [0.53,0.82,0.81,0.83,0.78]

epochs=[1,2,3,4,5,6,7,8,9,10]
train_losses=[0.6714918613433838, 0.3878242075443268, 0.11695915460586548, 0.022239843383431435, 0.005675839725881815, 0.0020590955391526222, 0.001192908501252532, 0.0008008991135284305, 0.0005718410830013454, 0.00042375936754979193]
validation_losses=[0.5580642819404602, 0.48639732599258423, 0.5714244246482849, 0.5852997303009033, 0.6188271045684814, 0.6398758888244629, 0.6697278618812561, 0.6910997629165649, 0.7190591096878052, 0.7381474375724792]

#############################################################################################################
'''
DRAW PLOTS
'''
#############################################################################################################
draw_plot(percentage_used,"RNN",RNN_ac,RNN_f1,RNN_precision,RNN_recall)

fig, axis = plt.subplots(1,2)

axis[0].set_title("                                                                                                                             "+"RNN loss")
axis[0].plot(epochs,train_losses,'ro-',label="Train Losses")
axis[0].set_ylabel("Train Losses")
axis[0].set_xlabel("Epochs")

axis[1].plot(epochs,validation_losses,'go-',label="Validation Losses")
axis[1].set_ylabel("Validation Losses")
axis[1].set_xlabel("Epochs")

plt.show()