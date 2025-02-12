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
percentage_used=[10,20,30,40,50]

LogReg_ac = [74.8,76.35,77.35,76.9,79]
LogReg_f1 = [0.74,0.75,0.77,0.76,0.79]
LogReg_precision = [0.76,0.77,0.79,0.8,0.78]
LogReg_recall = [0.72,0.74,0.75,0.73,0.79]

NaiveBayes_ac=[74,68.57,84,86,84.6]
NaiveBayes_f1=[0.72,0.56,0.85,0.8,0.82]
NaiveBayes_precision=[0.9,0.78,1,0.75,0.82]
NaiveBayes_recall=[0.6,0.43,0.73,0.86,0.82]

RandomForest_ac=[55,52.3,52.7,50.4,54.34]
RandomForest_f1=[0.68,0.67,0.68,0.66,0.68]
RandomForest_precision=[0.53,0.51,0.51,0.49,0.52]
RandomForest_recall=[0.95,0.99,0.99,0.998,0.97]

#############################################################################################################
'''
DRAW PLOTS
'''
#############################################################################################################
draw_plot(percentage_used,"Random Forest",RandomForest_ac,RandomForest_f1,RandomForest_precision,RandomForest_recall)
draw_plot(percentage_used,"Naive Bayes",NaiveBayes_ac,NaiveBayes_f1,NaiveBayes_precision,NaiveBayes_recall)
draw_plot(percentage_used,"Logistic Regression",LogReg_ac,LogReg_f1,LogReg_precision,LogReg_recall)